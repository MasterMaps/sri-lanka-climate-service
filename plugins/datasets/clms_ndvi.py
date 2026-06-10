"""CLMS Normalized Difference Vegetation Index (NDVI) plugin.

10-daily (dekadal) NDVI at 300m from the Copernicus Land Monitoring Service.
Source: Copernicus Data Space Ecosystem (CDSE) — requires S3 credentials.

Data is available on the 1st, 11th and 21st of each month from 2014 onwards.
Derived from PROBA-V (2014–2020) and Sentinel-3/OLCI (2020–present).

Credentials: register at https://dataspace.copernicus.eu/, generate S3 keys at
https://eodata-s3keysmanager.dataspace.copernicus.eu/, then add to ~/.aws/credentials:

    [cdse]
    aws_access_key_id = <ACCESS-KEY>
    aws_secret_access_key = <SECRET-KEY>
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.streaming.protocol import GridSpec
from .clms_gpp import _dekadal_dates

logger = logging.getLogger(__name__)

# 300m ≈ 1/360 degree
_RES_DEG = 1.0 / 360.0

_COLLECTION = "clms_ndvi_global_300m_10daily_v3_cog"
_ASSET_KEY = "ndvi300_ndvi"
_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
_FIRST_YEAR = 2014
_VAR = "ndvi"


class ClmsNdviPlugin:
    """IngestionPlugin for CLMS NDVI 300m 10-daily data via CDSE STAC.

    Fetches Cloud-Optimised GeoTIFFs from the Copernicus Data Space Ecosystem
    S3 storage, clips to the instance bbox, and commits one timestep per dekad.
    """

    max_concurrency = 2
    commit_batch_size = 1
    rechunk_time = 30
    pyramid: bool = True

    async def probe(self, bbox: list[float], **_: Any) -> GridSpec:
        xmin, ymin, xmax, ymax = map(float, bbox)
        nx = max(1, round((xmax - xmin) / _RES_DEG))
        ny = max(1, round((ymax - ymin) / _RES_DEG))
        return GridSpec(
            shape=(ny, nx),
            crs=4326,
            dtype=np.dtype("float32"),
            nodata=float("nan"),
            time_dim="t",
        )

    async def periods(self, start: str, end: str) -> list[str]:
        import pystac_client
        clamped_start = max(start[:10], f"{_FIRST_YEAR}-01-01")
        catalog = pystac_client.Client.open(_STAC_URL)
        search = catalog.search(collections=[_COLLECTION], max_items=1, sortby="-datetime")
        items = list(search.items())
        if not items:
            return []
        latest = items[0].datetime.date().isoformat()
        clamped_end = min(end[:10], latest)
        return _dekadal_dates(clamped_start, clamped_end)

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import os
        import boto3
        import pystac_client
        import rioxarray as rxr

        xmin, ymin, xmax, ymax = map(float, bbox)
        period_date = date.fromisoformat(period_id)
        window_end = period_date + timedelta(days=9)

        catalog = pystac_client.Client.open(_STAC_URL)
        search = catalog.search(
            collections=[_COLLECTION],
            bbox=[xmin, ymin, xmax, ymax],
            datetime=f"{period_id}/{window_end.isoformat()}",
            max_items=1,
        )
        items = list(search.items())
        if not items:
            raise ValueError(f"No CLMS NDVI item found for {period_id}")

        item = items[0]
        s3_url = item.assets[_ASSET_KEY].href
        s3_key = s3_url.removeprefix("s3://eodata/")
        logger.info("Fetching CLMS NDVI %s: %s", period_id, s3_url)

        # Use GDAL VSIS3 range requests (COG) — only downloads tiles covering the bbox, not the full global file
        session = boto3.Session(profile_name="cdse")
        creds = session.get_credentials().get_frozen_credentials()
        env = {
            "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
            "AWS_HTTPS": "YES",
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "AWS_ACCESS_KEY_ID": creds.access_key,
            "AWS_SECRET_ACCESS_KEY": creds.secret_key,
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_CHUNK_SIZE": "1048576",
        }
        old_env = {k: os.environ.get(k) for k in env}
        try:
            os.environ.update(env)
            da = rxr.open_rasterio(f"/vsis3/eodata/{s3_key}", chunks=None, masked=True, lock=False)
            da = da.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)
            da = da.squeeze("band", drop=True)
            # Mask flag values (252=Unknown, 253=Snow, 254=Water, 255=Missing)
            da = da.where(da <= 250)
            scale = float(da.attrs.get("scale_factor", 1.0))
            offset = float(da.attrs.get("add_offset", 0.0))
            da = (da * scale + offset).astype("float32")
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        # Ensure y is ascending (south → north)
        if da.y.values[0] > da.y.values[-1]:
            da = da.isel(y=slice(None, None, -1))

        ds = da.to_dataset(name=_VAR)
        ds = ds.expand_dims(t=[np.datetime64(period_id)])
        ds = ds.load()
        return ds
