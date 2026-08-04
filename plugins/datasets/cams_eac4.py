"""CAMS EAC4 3-hourly atmospheric composition plugin.

Global reanalysis at 0.75° from the Copernicus Atmosphere Monitoring Service.
Dataset: cams-global-reanalysis-eac4. Downloads one calendar month per API
request and caches the result in memory to serve all 3-hourly periods in
that month without redundant re-downloads.

CAMS EAC4 covers 2003-01-01 onwards with ~9 months of publication lag.

Authentication: register at https://ads.atmosphere.copernicus.eu/ and copy
your API key. Store it in ~/.ecmwfdatastoresrc:

    url: https://ads.atmosphere.copernicus.eu/api
    key: <your-api-key>

Or set environment variables ECMWF_DATASTORES_URL and ECMWF_DATASTORES_KEY.
Falls back to reading the key from ~/.cdsapirc if neither is found.
"""

from __future__ import annotations

import calendar
import logging
import shutil
import tempfile
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin

logger = logging.getLogger(__name__)

_ADS_URL = "https://ads.atmosphere.copernicus.eu/api"
_DATASET = "cams-global-reanalysis-eac4"
_TIMES_3H = [f"{h:02d}:00" for h in range(0, 22, 3)]  # 00:00…21:00, 8 per day
_RES_DEG = 0.75
_FIRST_DATE = "2003-01-01"
_LAG_MONTHS = 9


def _get_ads_key() -> str:
    import os

    if key := os.getenv("ECMWF_DATASTORES_KEY"):
        return key
    for rc_file in ("~/.ecmwfdatastoresrc", "~/.cdsapirc"):
        rc_path = Path(rc_file).expanduser()
        if rc_path.exists():
            for line in rc_path.read_text().splitlines():
                if line.strip().startswith("key:"):
                    return line.split(":", 1)[1].strip()
    raise RuntimeError(
        "No ADS API key found. Set ECMWF_DATASTORES_KEY or add\n"
        "  url: https://ads.atmosphere.copernicus.eu/api\n"
        "  key: <your-key>\n"
        "to ~/.ecmwfdatastoresrc"
    )


class CamsEac4Plugin(BaseDatasetPlugin):
    """IngestionPlugin for CAMS EAC4 3-hourly data via the ADS API.

    Downloads one calendar month per API request and caches it in memory;
    subsequent fetch_period calls for the same month read from the cache.

    Args:
        api_variable: CAMS EAC4 variable name for the ADS request
                      (e.g. 'particulate_matter_2.5um').
        variable:     Name to use for the output dataset variable
                      (e.g. 'pm2p5'). Must match the YAML variable: field.
    """

    max_concurrency = 1  # ADS allows only 1 active job per account
    commit_batch_size = 248  # ≥ max 3-hourly periods in one month (31 × 8)
    rechunk_time = 56  # one week of 3-hourly

    def __init__(self, api_variable: str, variable: str) -> None:
        self.api_variable = api_variable
        self.variable = variable
        self._cached_month: tuple[int, int] | None = None
        self._cached_ds: xr.Dataset | None = None

    async def periods(self, start: str, end: str) -> list[str]:
        if start[:10] < _FIRST_DATE:
            start = _FIRST_DATE
        cutoff = date.today() - timedelta(days=30 * _LAG_MONTHS)
        start_str = start if "T" in start else start + "T00"
        end_str = end if "T" in end else end + "T23"
        start_dt = datetime.strptime(start_str[:13], "%Y-%m-%dT%H")
        end_dt = datetime.strptime(end_str[:13], "%Y-%m-%dT%H")
        cutoff_dt = datetime(cutoff.year, cutoff.month, cutoff.day, 21)
        effective_end = min(end_dt, cutoff_dt)
        if start_dt.hour % 3 != 0:
            start_dt = start_dt.replace(hour=(start_dt.hour // 3 + 1) * 3)
        periods = []
        current = start_dt
        while current <= effective_end:
            periods.append(current.strftime("%Y-%m-%dT%H"))
            current += timedelta(hours=3)
        return periods

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        dt = datetime.fromisoformat(period_id)
        year, month = dt.year, dt.month

        if self._cached_month != (year, month):
            self._cached_ds = self._download_month(year, month, bbox)
            self._cached_month = (year, month)

        ds_month = self._cached_ds
        assert ds_month is not None

        # Select the single 3-hourly timestep
        time_coord = "valid_time" if "valid_time" in ds_month.coords else "time"
        target = np.datetime64(f"{year}-{month:02d}-{dt.day:02d}T{dt.hour:02d}:00")
        da = ds_month[self.variable].sel({time_coord: target}, method="nearest")
        da = da.drop_vars(time_coord, errors="ignore")

        # Ensure y ascending (south → north)
        lat_dim = "latitude" if "latitude" in da.dims else "y"
        if float(da[lat_dim].values[0]) > float(da[lat_dim].values[-1]):
            da = da.isel({lat_dim: slice(None, None, -1)})

        # Rename latitude/longitude → y/x
        rename = {k: v for k, v in (("latitude", "y"), ("longitude", "x")) if k in da.dims}
        if rename:
            da = da.rename(rename)

        da = da * 1e9  # kg/m³ → μg/m³
        ds = da.to_dataset(name=self.variable)
        ds = ds.expand_dims(t=[np.datetime64(period_id)])
        return ds.load()

    def _download_month(self, year: int, month: int, bbox: list[float]) -> xr.Dataset:
        from ecmwf.datastores import Client

        xmin, ymin, xmax, ymax = map(float, bbox)
        _, last_day = calendar.monthrange(year, month)
        logger.info(
            "Downloading CAMS EAC4 %s %d-%02d from ADS",
            self.api_variable, year, month,
        )

        client = Client(url=_ADS_URL, key=_get_ads_key())
        params = {
            "variable": [self.api_variable],
            "date": [f"{year}-{month:02d}-01/{year}-{month:02d}-{last_day:02d}"],
            "time": _TIMES_3H,
            "area": [ymax, xmin, ymin, xmax],  # N, W, S, E
            "data_format": "netcdf_zip",
        }
        remote = client.submit(_DATASET, params)

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "cams.zip"
            remote.download(str(zip_path))
            with zipfile.ZipFile(zip_path) as archive:
                first_name = archive.namelist()[0]
                nc_path = Path(tmpdir) / "cams.nc"
                with archive.open(first_name) as src, open(nc_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            ds = xr.open_dataset(nc_path).load()

        # Drop expver (experiment version tag, not a data variable)
        if "expver" in ds.dims:
            ds = ds.isel(expver=0, drop=True)
        elif "expver" in ds.coords:
            ds = ds.drop_vars("expver")

        # Rename data variable to the configured output name if needed
        actual_vars = [v for v in ds.data_vars if v != "expver"]
        if actual_vars and actual_vars[0] != self.variable:
            ds = ds.rename({actual_vars[0]: self.variable})

        return ds
