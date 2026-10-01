"""Copernicus DEM GLO-30 elevation plugin.

Static dataset — single ingest, no time dimension.
Source: Copernicus DEM GLO-30 COG tiles from AWS Open Data (public, no auth).
Variable: elevation (Digital Surface Model) at ~30m (1 arc-second) resolution.

Tiles are fetched per 1°×1° grid cell; only the bytes covering the bbox
window are downloaded per tile (HTTP range requests via GDAL VSI).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin

logger = logging.getLogger(__name__)

# GLO-30: 1 arc-second per pixel = 1/3600 degree
_RES_DEG = 1.0 / 3600

_S3_BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def _tile_url(lat: int, lon: int) -> str:
    lat_str = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
    lon_str = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"
    name = f"Copernicus_DSM_COG_10_{lat_str}_00_{lon_str}_00_DEM"
    return f"{_S3_BASE}/{name}/{name}.tif"


class CopernicusDemPlugin(BaseDatasetPlugin):
    """IngestionPlugin for Copernicus DEM GLO-30.

    Effectively static: a single nominal timestep is written and not extended.
    Tiles covering the bbox are fetched via HTTP range requests from the
    AWS Open Data bucket — no authentication required.
    """

    max_concurrency = 1
    commit_batch_size = 1
    rechunk_time: int | None = None
    pyramid: bool = True

    async def periods(self, start: str, end: str) -> list[str]:
        return ["static"]

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import rioxarray
        from rioxarray.merge import merge_arrays

        xmin, ymin, xmax, ymax = map(float, bbox)

        lon_min_tile = int(math.floor(xmin))
        lon_max_tile = int(math.floor(xmax))
        lat_min_tile = int(math.floor(ymin))
        lat_max_tile = int(math.floor(ymax))

        arrays: list[xr.DataArray] = []
        for lat in range(lat_min_tile, lat_max_tile + 1):
            for lon in range(lon_min_tile, lon_max_tile + 1):
                url = _tile_url(lat, lon)
                logger.info("Fetching GLO-30 tile lat=%d lon=%d", lat, lon)
                try:
                    da = rioxarray.open_rasterio(url, chunks=None, masked=True, lock=False)
                    da = da.squeeze("band", drop=True)
                    da = da.rio.clip_box(minx=xmin, miny=ymin, maxx=xmax, maxy=ymax)
                    da = da.load()
                    arrays.append(da)
                except Exception:
                    # Tiles over ocean or at coverage edge may not exist
                    logger.warning("GLO-30 tile not available: %s", url, exc_info=True)

        if not arrays:
            raise ValueError(f"No GLO-30 tiles found for bbox {bbox}")

        merged = merge_arrays(arrays) if len(arrays) > 1 else arrays[0]
        merged = merged.astype(np.float32)

        ds = merged.to_dataset(name="elevation")
        # OCS coverage/publish requires a time dimension; carry one nominal step.
        ds = ds.expand_dims(t=[np.datetime64("2010-01-01")])
        return ds
