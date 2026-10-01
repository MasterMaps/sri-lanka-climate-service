"""Cicero PM2.5 surface concentration plugin.

Daily PM2.5 (μg m⁻³) for Sri Lanka from a locally-stored NetCDF file produced
by the CICERO Center for International Climate Research.

File: pm_final_srilanka_linearp.nc
Period: 2020-03-01 – 2023-12-31
Resolution: 0.01° (~1 km)
Projection: geographic (EPSG:4326)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin

logger = logging.getLogger(__name__)

_VAR = "pm25"
_SOURCE_VAR = "__xarray_dataarray_variable__"
_RES_DEG = 0.01


class CiceroPm25Plugin(BaseDatasetPlugin):
    """IngestionPlugin for locally-stored Cicero PM2.5 NetCDF data.

    Opens the file lazily on first access and slices one day at a time.
    No network access required — all data is read from disk.
    """

    max_concurrency = 1
    commit_batch_size = 30
    rechunk_time = 365
    pyramid: bool = True

    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        self._ds: xr.Dataset | None = None

    def _open(self) -> xr.Dataset:
        if self._ds is None:
            logger.info("Opening Cicero PM2.5 file: %s", self._file_path)
            self._ds = xr.open_dataset(self._file_path, chunks={})
        return self._ds

    async def periods(self, start: str, end: str) -> list[str]:
        ds = self._open()
        times = ds.time.values
        start_dt = np.datetime64(start[:10], "D")
        end_dt = np.datetime64(end[:10], "D")
        mask = (times.astype("datetime64[D]") >= start_dt) & (
            times.astype("datetime64[D]") <= end_dt
        )
        return [str(t)[:10] for t in times[mask]]

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        ds = self._open()
        xmin, ymin, xmax, ymax = map(float, bbox)

        da = ds[_SOURCE_VAR].sel(time=period_id, method="nearest")
        da = da.sel(
            lon=slice(xmin, xmax),
            lat=slice(ymin, ymax),
        )
        da = da.rename({"lon": "x", "lat": "y"})
        da = da.astype("float32")
        da.attrs["units"] = "μg m⁻³"

        result = da.rename(_VAR).to_dataset()
        result = result.expand_dims(t=[np.datetime64(period_id)])
        result = result.load()
        return result
