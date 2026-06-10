"""WorldPop Global2 1 km yearly population (global mosaic, clipped to the extent).

WorldPop publishes a global 1 km constrained/UA mosaic per year (2015-2030). The
data server advertises ``Accept-Ranges`` but ignores actual ``Range`` GETs
(returns 200, not 206), so partial/windowed reads over ``/vsicurl`` do not work
(see dhis2/open-climate-service#269). This plugin therefore downloads the ~290 MB
yearly mosaic once, caches it per process, and reads the instance bounding box as
a windowed clip from the *local* file (cheap, no full raster in memory).
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import requests
import rioxarray  # noqa: F401  # registers .rio
import xarray as xr

from open_climate_service.streaming.protocol import GridSpec

logger = logging.getLogger(__name__)

_FIRST_YEAR, _LAST_YEAR = 2015, 2030
_PROBE_YEAR = 2020
_CACHE: dict[int, str] = {}


def _mosaic_url(year: int) -> str:
    return (
        "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/"
        f"{year}/0_Mosaicked/v1/1km_ua/constrained/global_pop_{year}_CN_1km_R2025A_UA_v1.tif"
    )


def _download_mosaic(year: int) -> str:
    cached = _CACHE.get(year)
    if cached and Path(cached).exists():
        return cached
    url = _mosaic_url(year)
    logger.info("Downloading WorldPop 1km mosaic %d (~290 MB) from %s", year, url)
    fd, path = tempfile.mkstemp(suffix=f"_worldpop_1km_{year}.tif")
    os.close(fd)
    with requests.get(url, stream=True, timeout=900) as resp:
        resp.raise_for_status()
        with open(path, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    _CACHE[year] = path
    return path


def _read_clip(year: int, bbox: list[float]) -> xr.Dataset:
    path = _download_mosaic(year)
    xmin, ymin, xmax, ymax = map(float, bbox)
    da = rioxarray.open_rasterio(path, masked=True).rio.clip_box(
        minx=xmin, miny=ymin, maxx=xmax, maxy=ymax
    )
    da = da.squeeze("band", drop=True).astype("float32")
    if float(da.y.values[0]) > float(da.y.values[-1]):
        da = da.isel(y=slice(None, None, -1))
    return da.to_dataset(name="pop_total")


class WorldPop1kmYearlyPlugin:
    """Streaming plugin for yearly WorldPop Global2 1 km population over the extent."""

    max_concurrency = 1
    commit_batch_size = 1

    def __init__(self, variable: str = "pop_total", **_: object) -> None:
        self.variable = variable

    async def probe(self, bbox: list[float], **_: Any) -> GridSpec:
        dataset = await asyncio.to_thread(_read_clip, _PROBE_YEAR, bbox)
        try:
            return GridSpec(
                shape=(int(dataset.sizes["y"]), int(dataset.sizes["x"])),
                crs=4326,
                dtype=np.dtype("float32"),
                nodata=float("nan"),
                time_dim="t",
            )
        finally:
            dataset.close()

    async def periods(self, start: str, end: str) -> list[str]:
        start_year = max(int(str(start)[:4]), _FIRST_YEAR)
        end_year = min(int(str(end)[:4]), _LAST_YEAR)
        if start_year > end_year:
            return []
        return [str(year) for year in range(start_year, end_year + 1)]

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        year = int(str(period_id)[:4])
        dataset = await asyncio.to_thread(_read_clip, year, bbox)
        if self.variable != "pop_total":
            dataset = dataset.rename({"pop_total": self.variable})
        dataset = dataset.expand_dims(t=[np.datetime64(f"{year}-01-01")])
        return dataset.load()
