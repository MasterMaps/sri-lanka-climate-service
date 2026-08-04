"""MODIS MCD12Q1 Land Cover Classification plugin.

Yearly land cover at 500m from the MODIS Terra+Aqua Combined product (v061).
Source: Copernicus Data Space Ecosystem (CDSE) OData API.

IMPORTANT: CDSE requires explicit activation of the TerraAqua/MODIS collection
in the user portal (https://dataspace.copernicus.eu/) before downloads work.
Without activation, both S3 and OData requests return HTTP 403. Go to the
Collections or Data Access section in the portal and subscribe/activate MODIS.

Credentials: set environment variables before starting the service:

    CDSE_USERNAME=your@email.com
    CDSE_PASSWORD=yourpassword

Variable: LC_Type1 — IGBP global vegetation classification (17 classes).
Available years: 2001–present.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin

logger = logging.getLogger(__name__)

# 500m ≈ 1/240 degree at equator (MODIS sinusoidal nominal)
_RES_DEG = 1.0 / 240.0

_COLLECTION = "modis-terraaqua-mcd12q1"
_ASSET_KEY = "data"
_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
_FIRST_YEAR = 2001
_VAR = "land_cover"
_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"


def _cdse_token(username: str, password: str) -> str:
    import requests
    r = requests.post(_TOKEN_URL, data={
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def _odata_download(url: str, dest: str, token: str) -> None:
    import requests
    with requests.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


class ModisLandCoverPlugin(BaseDatasetPlugin):
    """IngestionPlugin for MODIS MCD12Q1 yearly land cover via CDSE STAC.

    Downloads HDF tiles for the bbox, reprojects from sinusoidal to WGS84,
    mosaics tiles, extracts LC_Type1, and commits one timestep per year.
    """

    max_concurrency = 1
    commit_batch_size = 1
    rechunk_time = 5
    pyramid: bool = True

    async def periods(self, start: str, end: str) -> list[str]:
        sy = max(int(start[:4]), _FIRST_YEAR)
        ey = int(end[:4])
        return [str(y) for y in range(sy, ey + 1)]

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import tempfile
        import pystac_client
        import rasterio
        import rioxarray as rxr
        from rasterio.merge import merge as rio_merge

        xmin, ymin, xmax, ymax = map(float, bbox)
        year = period_id[:4]

        catalog = pystac_client.Client.open(_STAC_URL)
        search = catalog.search(
            collections=[_COLLECTION],
            bbox=[xmin, ymin, xmax, ymax],
            datetime=f"{year}-01-01/{year}-12-31",
        )
        items = list(search.items())
        if not items:
            raise ValueError(f"No MODIS MCD12Q1 items found for {year}")
        logger.info("Found %d MODIS tile(s) for %s", len(items), year)

        import os
        import requests

        token = _cdse_token(os.environ["CDSE_USERNAME"], os.environ["CDSE_PASSWORD"])

        tile_arrays: list[xr.DataArray] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for item in items:
                odata_url = item.assets["Product"].href
                local = f"{tmpdir}/{item.id}.hdf"
                logger.info("Downloading %s", item.id)
                _odata_download(odata_url, local, token)

                # Open the LC_Type1 subdataset inside the HDF4 file
                with rasterio.open(local) as src:
                    lc_sds = next(
                        (s[0] for s in src.subdatasets if "LC_Type1" in s[0]), None
                    )
                if lc_sds is None:
                    raise ValueError(f"LC_Type1 subdataset not found in {item.id}")

                da = rxr.open_rasterio(lc_sds, chunks=None, masked=True, lock=False)
                # Reproject from MODIS sinusoidal to WGS84 and clip
                da = da.rio.reproject("EPSG:4326")
                da = da.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)
                tile_arrays.append(da.squeeze("band", drop=True).load())

        # Mosaic tiles (take first valid value per pixel)
        if len(tile_arrays) == 1:
            da = tile_arrays[0]
        else:
            da = tile_arrays[0].copy()
            for tile in tile_arrays[1:]:
                # Reindex tile to match first tile's grid, fill gaps
                tile_aligned = tile.reindex_like(da, method="nearest", tolerance=_RES_DEG)
                da = da.where(da.notnull(), tile_aligned)

        da = da.where(da < 255).astype("uint8")

        # Ensure y ascending (south → north)
        if da.y.values[0] > da.y.values[-1]:
            da = da.isel(y=slice(None, None, -1))

        ds = da.to_dataset(name=_VAR)
        ds = ds.expand_dims(t=[np.datetime64(f"{year}-01-01")])
        return ds.load()
