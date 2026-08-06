"""ESA CCI Land Cover plugin.

Yearly land cover classification at 300m from the ESA CCI Land Cover v2 dataset.
Source: Copernicus Climate Data Store (CDS) — requires ~/.cdsapirc credentials.
Register at: https://cds.copernicus.eu/how-to-api

Available years: 1992–2022
Version scheme: v2_0_7cds for ≤2015, v2_1_1 for ≥2016.
"""

from __future__ import annotations

import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.streaming import BaseDatasetPlugin

logger = logging.getLogger(__name__)

# 300m resolution ≈ 10 arc-seconds = 10/3600 degrees
_RES_DEG = 10.0 / 3600

_FIRST_YEAR = 1992
_LAST_YEAR = 2022
_CDS_DATASET = "satellite-land-cover"


def _cds_version(year: int) -> str:
    return "v2_0_7cds" if year <= 2015 else "v2_1_1"


class EsaLandCoverPlugin(BaseDatasetPlugin):
    """IngestionPlugin for ESA CCI Land Cover yearly data via CDS API.

    Downloads per-year NetCDF files clipped to the bbox, extracts the
    lccs_class variable, and commits one timestep per year.

    CDS credentials must be present in ~/.cdsapirc before first use.
    """

    max_concurrency = 1  # CDS processes one job at a time per account
    commit_batch_size = 1
    rechunk_time = 5
    pyramid: bool = True

    async def periods(self, start: str, end: str) -> list[str]:
        sy = max(int(start[:4]), _FIRST_YEAR)
        ey = min(int(end[:4]), _LAST_YEAR)
        return [str(y) for y in range(sy, ey + 1)]

    async def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        import cdsapi

        year = int(period_id[:4])
        xmin, ymin, xmax, ymax = map(float, bbox)

        request = {
            "variable": "all",
            "year": [str(year)],
            "version": [_cds_version(year)],
            "area": [ymax, xmin, ymin, xmax],  # [N, W, S, E]
        }
        logger.info("Fetching ESA CCI Land Cover %d from CDS (bbox %s)", year, bbox)

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "landcover.zip"
            cdsapi.Client().retrieve(_CDS_DATASET, request, str(zip_path))

            with zipfile.ZipFile(zip_path) as zf:
                nc_names = [n for n in zf.namelist() if n.endswith(".nc")]
                if not nc_names:
                    raise ValueError(f"No .nc file in CDS download for {year}: {zf.namelist()}")
                zf.extract(nc_names[0], tmpdir)
                nc_path = Path(tmpdir) / nc_names[0]

            ds = xr.open_dataset(nc_path, decode_coords="all")

            if "lccs_class" in ds:
                ds = ds[["lccs_class"]]
            else:
                first_var = next(iter(ds.data_vars))
                logger.warning("lccs_class not found; using %r", first_var)
                ds = ds[[first_var]]

            # Normalise dimension names to x/y
            rename = {}
            for dim in list(ds.dims):
                if dim.lower() in ("lat", "latitude"):
                    rename[dim] = "y"
                elif dim.lower() in ("lon", "longitude"):
                    rename[dim] = "x"
            if rename:
                ds = ds.rename(rename)

            # Ensure a single-element time dimension
            if "t" not in ds.dims:
                ds = ds.expand_dims(dim="t").assign_coords(
                    t=[np.datetime64(f"{year}-01-01", "ns")]
                )

            ds = ds.load()
        return ds
