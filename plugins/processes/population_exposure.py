"""Population exposure raster: a climate variable multiplied by population.

Produces a gridded "population-weighted exposure" surface — for each pixel and
time step, ``value(x, y, t) x population(x, y)`` — e.g. PM2.5 x WorldPop. The
population is reduced to a single surface (its spatial pattern is stable year to
year) and aligned to the climate grid by nearest-neighbour interpolation
(preserving counts), then multiplied. Output keeps the climate cube's (t, y, x)
shape and is published as a new GeoZarr dataset.
"""

from __future__ import annotations

from typing import Any

import xarray as xr

from open_climate_service.process import process


def _as_dataarray(cube: Any) -> xr.DataArray:
    if isinstance(cube, xr.Dataset):
        return cube[list(cube.data_vars)[0]]
    if "bands" in getattr(cube, "dims", ()):
        return cube.isel(bands=0, drop=True)
    return cube


def _find_dim(obj: Any, candidates: list[str]) -> str | None:
    dims = obj.dims if isinstance(obj, xr.DataArray) else set(obj.dims)
    for c in candidates:
        if c in dims:
            return c
    return None


@process(
    summary="Population exposure raster: climate value x population",
    description=(
        "Multiplies a climate cube by a population surface, per pixel and time step, "
        "to produce a gridded population-exposure raster (e.g. PM2.5 x WorldPop, units "
        "person*ug/m3). The population is reduced to one surface and aligned to the "
        "climate grid by nearest-neighbour interpolation."
    ),
)
def population_exposure(data: Any, weights: Any) -> xr.DataArray:
    data = _as_dataarray(data)
    weights = _as_dataarray(weights)

    x_dim = _find_dim(data, ["x", "longitude", "lon"])
    y_dim = _find_dim(data, ["y", "latitude", "lat"])
    wx = _find_dim(weights, ["x", "longitude", "lon"])
    wy = _find_dim(weights, ["y", "latitude", "lat"])

    # Reduce population to a single (y, x) surface ...
    for d in list(weights.dims):
        if d not in (wx, wy):
            weights = weights.mean(dim=d) if weights.sizes[d] > 1 else weights.isel({d: 0}, drop=True)
    if (wx, wy) != (x_dim, y_dim):
        weights = weights.rename({wx: x_dim, wy: y_dim})
    # ... then align to the climate grid (nearest preserves population counts).
    weights = weights.interp({x_dim: data[x_dim], y_dim: data[y_dim]}, method="nearest")

    exposure = data * weights  # broadcasts the (y, x) surface across time
    exposure.name = data.name or "exposure"
    exposure.attrs = {"long_name": "population exposure (climate value x population)"}
    return exposure
