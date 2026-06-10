"""Population-weighted spatial aggregation of a climate cube to org units.

Replicates the DHIS2 Climate Tools "population weighting" guide
(https://climate-tools.dhis2.org/guides/aggregation/population-weighting/):

    weighted mean per org unit = sum(value * population) / sum(population)

`data` is the climate cube (t, y, x), `weights` is a population cube. The
population surface is aligned to the climate grid by nearest-neighbour interp
(preserving counts), multiplied, and summed per polygon, then normalised by the
summed population. Output matches `aggregate_spatial`: a dataset with
(geometry, t) dims that feeds the DHIS2 JSON / CHAP CSV exporters.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.process import process


def _parse_geometries(geometries: Any) -> tuple[list[Any], list[str]]:
    from shapely.geometry import shape

    if isinstance(geometries, dict):
        if geometries.get("type") == "FeatureCollection":
            feats = geometries.get("features", [])
            return [shape(f["geometry"]) for f in feats], [str(f.get("id", i)) for i, f in enumerate(feats)]
        if geometries.get("type") == "Feature":
            return [shape(geometries["geometry"])], [str(geometries.get("id", 0))]
        return [shape(geometries)], ["0"]
    items = [shape(g) if isinstance(g, dict) else g for g in geometries]
    return items, [str(i) for i in range(len(items))]


def _find_dim(obj: Any, candidates: list[str]) -> str | None:
    dims = obj.dims if isinstance(obj, xr.DataArray) else set(obj.dims)
    for c in candidates:
        if c in dims:
            return c
    return None


def _as_dataarray(cube: Any) -> xr.DataArray:
    if isinstance(cube, xr.Dataset):
        return cube[list(cube.data_vars)[0]]
    if "bands" in getattr(cube, "dims", ()):
        return cube.isel(bands=0, drop=True)
    return cube


@process(
    summary="Population-weighted spatial aggregation to geometries",
    description=(
        "For each geometry computes sum(data * weights) / sum(weights) per time step — "
        "the population-weighted mean of the climate variable. `weights` (population) is "
        "aligned to the climate grid by nearest-neighbour interpolation; a time dimension "
        "on the weights is averaged to a single surface. Returns one value per geometry per "
        "climate time step."
    ),
)
def aggregate_spatial_weighted(
    data: Any,
    weights: Any,
    geometries: Any,
    target_dimension: str | None = None,
) -> xr.Dataset:
    import rasterio.features
    from rasterio.transform import from_bounds
    from shapely.geometry import mapping

    data = _as_dataarray(data)
    weights = _as_dataarray(weights)

    x_dim = _find_dim(data, ["x", "longitude", "lon"])
    y_dim = _find_dim(data, ["y", "latitude", "lat"])
    t_dim = _find_dim(data, ["t", "time"])
    if x_dim is None or y_dim is None:
        raise ValueError(f"aggregate_spatial_weighted: cannot find x/y dims in {list(data.dims)}")

    # Reduce the population to a single (y, x) surface, then align to the climate grid.
    wx = _find_dim(weights, ["x", "longitude", "lon"])
    wy = _find_dim(weights, ["y", "latitude", "lat"])
    for d in list(weights.dims):
        if d not in (wx, wy):
            weights = weights.mean(dim=d) if weights.sizes[d] > 1 else weights.isel({d: 0}, drop=True)
    weights = weights.rename({wx: x_dim, wy: y_dim}) if (wx, wy) != (x_dim, y_dim) else weights
    weights = weights.interp({x_dim: data[x_dim], y_dim: data[y_dim]}, method="nearest")
    w_vals = weights.values.astype("float64")

    x_coords = data[x_dim].values.astype(float)
    y_coords = data[y_dim].values.astype(float)
    height, width = len(y_coords), len(x_coords)
    dx = float(abs(x_coords[1] - x_coords[0])) if width > 1 else 1.0
    dy = float(abs(y_coords[1] - y_coords[0])) if height > 1 else 1.0
    transform = from_bounds(
        x_coords.min() - dx / 2, y_coords.min() - dy / 2, x_coords.max() + dx / 2, y_coords.max() + dy / 2, width, height
    )

    if t_dim:
        arr = data.transpose(t_dim, y_dim, x_dim).values.astype("float64")
    else:
        arr = data.transpose(y_dim, x_dim).values.astype("float64")[None]
    n_t = arr.shape[0]

    geoms, labels = _parse_geometries(geometries)
    geom_dim = target_dimension or "geometry"

    out = np.full((len(geoms), n_t), np.nan, dtype="float64")
    for gi, geom in enumerate(geoms):
        mask = rasterio.features.geometry_mask(
            [mapping(geom)], out_shape=(height, width), transform=transform, invert=True
        )
        if height > 1 and float(y_coords[1]) > float(y_coords[0]):
            mask = mask[::-1]
        w_here = np.where(mask, w_vals, np.nan)
        for ti in range(n_t):
            v = arr[ti]
            valid = mask & np.isfinite(v) & np.isfinite(w_here) & (w_here > 0)
            denom = w_here[valid].sum()
            if denom > 0:
                out[gi, ti] = float((v[valid] * w_here[valid]).sum() / denom)

    # Emit the period dimension as the canonical "t" (the CHAP/DHIS2 exporters
    # expect "t"), regardless of whether the source cube used "t" or "time".
    coords: dict[str, Any] = {geom_dim: labels}
    if t_dim:
        coords["t"] = data[t_dim].values
        result = xr.DataArray(out, dims=[geom_dim, "t"], coords=coords)
    else:
        result = xr.DataArray(out[:, 0], dims=[geom_dim], coords=coords)
    return result.to_dataset(name=data.name or "value")
