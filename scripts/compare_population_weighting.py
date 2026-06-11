#!/usr/bin/env python
"""Compare org-unit aggregation **with vs without population weighting** as choropleths.

For each district it aggregates a climate variable two ways against a running
Open Climate Service instance, then draws side-by-side choropleths plus their
difference:

  * **Unweighted**  — simple spatial mean per district
                      (built-in ``aggregate_to_chap_csv`` workflow, method=mean).
  * **Population-weighted** — sum(value x population) / sum(population) per district
                      (the ``aggregate_population_weighted_chap`` workflow).

Both run server-side over the *same* climate dataset, extent and geometries, so
the only difference is the weighting — which shifts each district's value toward
where people actually live. Inspired by the DHIS2 Climate Tools guides:
  https://climate-tools.dhis2.org/guides/aggregation/population-weighting/
  https://climate-tools.dhis2.org/guides/aggregation/org-unit-aggregation/

Districts are pulled from GADM ADM2 (override with --geometries to use your own
org-unit GeoJSON, e.g. DHIS2 boundaries). Values are averaged over the period so
each district gets one number to map.

Example:
  python scripts/compare_population_weighting.py \
      --instance http://localhost:8014 \
      --dataset cicero_pm25_daily --population worldpop_population_1km_yearly \
      --temporal-extent 2023-01-01 2023-01-31 --gadm-country LKA \
      --out sri_lanka_pm25_weighting.png
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

_GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_{iso}_2.json.zip"
_NON_VALUE_COLS = {"time_period", "location"}


def load_districts(geometries: str | None, gadm_country: str | None) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of org units with a 'name' column (the choropleth unit)."""
    if geometries:
        gdf = gpd.read_file(geometries)
        name_col = next(
            (c for c in ("name", "shapeName", "NAME_2", "NAME_1", "id") if c in gdf.columns), gdf.columns[0]
        )
    else:
        if not gadm_country:
            raise SystemExit("Provide --geometries or --gadm-country")
        print(f"Fetching GADM ADM2 districts for {gadm_country} ...")
        resp = requests.get(_GADM_URL.format(iso=gadm_country.upper()), timeout=120)
        resp.raise_for_status()
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(tmp)
            (json_path,) = list(Path(tmp).glob("*.json"))
            gdf = gpd.read_file(json_path)
        name_col = "NAME_2"
    gdf = gdf.rename(columns={name_col: "name"})[["name", "geometry"]]
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf


def feature_collection(gdf: gpd.GeoDataFrame) -> dict:
    """GeoJSON FeatureCollection whose feature ids are the district names (= CHAP location)."""
    import shapely.geometry as sgeom

    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": str(row["name"]), "geometry": sgeom.mapping(row.geometry)}
            for _, row in gdf.iterrows()
        ],
    }


def run_workflow(instance: str, process_id: str, arguments: dict) -> pd.DataFrame:
    """POST a CHAP-CSV-producing workflow to /result and return it averaged per location."""
    body = {"process": {"process_graph": {"agg": {"process_id": process_id, "arguments": arguments, "result": True}}}}
    resp = requests.post(f"{instance.rstrip('/')}/result", json=body, timeout=900)
    if resp.status_code != 200:
        raise SystemExit(f"{process_id} failed (HTTP {resp.status_code}): {resp.text[:300]}")
    df = pd.read_csv(io.StringIO(resp.text))
    value_cols = [c for c in df.columns if c not in _NON_VALUE_COLS]
    if not value_cols:
        raise SystemExit(f"{process_id}: no value column in CHAP CSV (got {list(df.columns)})")
    # One value per district: average across the period's time steps.
    return df.groupby("location")[value_cols[0]].mean().rename(process_id)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instance", default="http://localhost:8014")
    p.add_argument("--dataset", default="cicero_pm25_daily", help="Climate dataset id")
    p.add_argument("--population", default="worldpop_population_1km_yearly", help="Population dataset id (weights)")
    p.add_argument("--temporal-extent", nargs=2, metavar=("START", "END"), default=["2023-01-01", "2023-01-31"])
    p.add_argument("--period-type", default="day")
    p.add_argument("--gadm-country", default="LKA", help="ISO3 to fetch GADM ADM2 districts for")
    p.add_argument("--geometries", default=None, help="Org-unit GeoJSON path (overrides --gadm-country)")
    p.add_argument("--label", default=None, help="Variable label for the colour bar (e.g. 'PM2.5 (ug/m3)')")
    p.add_argument("--out", default="population_weighting_comparison.png")
    args = p.parse_args()

    gdf = load_districts(args.geometries, args.gadm_country)
    fc = feature_collection(gdf)
    extent = list(args.temporal_extent)
    print(f"{len(gdf)} org units; {args.dataset} over {extent[0]}..{extent[1]}")

    unweighted = run_workflow(
        args.instance,
        "aggregate_to_chap_csv",
        {"dataset_id": args.dataset, "temporal_extent": extent, "geometries": fc,
         "method": "mean", "period_type": args.period_type},
    )
    weighted = run_workflow(
        args.instance,
        "aggregate_population_weighted_chap",
        {"dataset_id": args.dataset, "population_dataset_id": args.population,
         "temporal_extent": extent, "geometries": fc, "period_type": args.period_type},
    )

    merged = gdf.merge(unweighted, left_on="name", right_index=True).merge(weighted, left_on="name", right_index=True)
    merged["difference"] = merged["aggregate_population_weighted_chap"] - merged["aggregate_to_chap_csv"]

    # Shared colour scale for the two aggregation maps; symmetric scale for the diff.
    lo = float(min(merged["aggregate_to_chap_csv"].min(), merged["aggregate_population_weighted_chap"].min()))
    hi = float(max(merged["aggregate_to_chap_csv"].max(), merged["aggregate_population_weighted_chap"].max()))
    dmax = float(merged["difference"].abs().max()) or 1.0
    label = args.label or args.dataset

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        ("aggregate_to_chap_csv", "Unweighted (simple mean)", "YlOrRd", lo, hi),
        ("aggregate_population_weighted_chap", "Population-weighted", "YlOrRd", lo, hi),
        ("difference", "Weighted - Unweighted", "RdBu_r", -dmax, dmax),
    ]
    for ax, (col, title, cmap, vmin, vmax) in zip(axes, panels):
        merged.plot(column=col, cmap=cmap, vmin=vmin, vmax=vmax, legend=True, ax=ax,
                    edgecolor="0.6", linewidth=0.3, legend_kwds={"shrink": 0.6})
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle(f"{label} by district — {extent[0]} to {extent[1]} (population weighting effect)", fontsize=14)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"Wrote {args.out}")
    # Quick numeric summary of the weighting effect.
    biggest = merged.reindex(merged["difference"].abs().sort_values(ascending=False).index).head(5)
    print("\nLargest weighting shifts (district: unweighted -> weighted, diff):")
    for _, r in biggest.iterrows():
        print(f"  {r['name']:<18} {r['aggregate_to_chap_csv']:.2f} -> "
              f"{r['aggregate_population_weighted_chap']:.2f}  ({r['difference']:+.2f})")


if __name__ == "__main__":
    sys.exit(main())
