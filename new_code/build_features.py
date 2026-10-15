"""TEMPLATE (not run against real data here): turn public geodata into span features.
Inputs: power-line geometries, canopy-height raster (LiDAR CHM or Meta/WRI 1 m), NDVI raster,
optional older CHM for growth, and an asset layer (voltage, conductor height, species, customers...).
Rasters must share one projected CRS in metres (e.g. UTM)."""
import argparse, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterstats import zonal_stats
from shapely.ops import substring


def split_line(g, span_m):
    n = max(1, int(np.ceil(g.length / span_m)))
    return [substring(g, i * g.length / n, (i + 1) * g.length / n) for i in range(n)]


def zs(geoms, raster, stat):
    return [o[stat] for o in zonal_stats(list(geoms), raster, stats=[stat], all_touched=True)]


def main(a):
    with rasterio.open(a.chm) as r:
        crs = r.crs
    lines = gpd.read_file(a.lines).to_crs(crs).explode(index_parts=False).reset_index(drop=True)
    rows = [{"line_id": f"L{j}", "span_id": f"L{j}-{i}", "geometry": s}
            for j, g in enumerate(lines.geometry) for i, s in enumerate(split_line(g, a.span_m))]
    sp = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)

    half = a.row_width / 2
    inner = sp.geometry.buffer(half)                                   # right-of-way
    edge = sp.geometry.buffer(half + a.edge_width).difference(inner)   # danger-tree zone
    sp["canopy_height_max_m"] = zs(inner, a.chm, "percentile_98")
    sp["edge_tree_height_m"] = zs(edge, a.chm, "percentile_95")
    sp["ndvi_mean"] = zs(inner, a.ndvi, "mean")
    sp["row_width_m"] = a.row_width
    if a.chm_prev:                                                     # growth from two LiDAR/CHM epochs
        prev = np.array(zs(inner, a.chm_prev, "percentile_98"), dtype=float)
        sp["growth_rate_m_yr"] = (sp["canopy_height_max_m"] - prev) / a.years
    sp["days_since_survey"] = (pd.Timestamp.today() - pd.Timestamp(a.survey_date)).days

    c = gpd.GeoSeries(sp.geometry.centroid, crs=crs).to_crs(4326)
    sp["lon"], sp["lat"] = c.x.values, c.y.values

    assets = gpd.read_file(a.assets).to_crs(crs)   # voltage_kv, conductor_height_m, species, customers_served, wildfire_zone, slope_deg, precip_mm_yr, wind_gust_p95_ms, ndvi_trend
    sp = gpd.sjoin_nearest(sp, assets, how="left", max_distance=50).drop(columns="index_right")
    sp = sp.drop_duplicates("span_id").drop(columns="geometry")
    if a.labels:                                   # span_id, encroach_12m from trim tickets / outages / LiDAR violations
        sp = sp.merge(pd.read_csv(a.labels), on="span_id", how="left")
    sp.to_csv(a.out, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", required=True); ap.add_argument("--chm", required=True)
    ap.add_argument("--ndvi", required=True); ap.add_argument("--assets", required=True)
    ap.add_argument("--chm_prev"); ap.add_argument("--years", type=float, default=3)
    ap.add_argument("--labels"); ap.add_argument("--survey_date", required=True)
    ap.add_argument("--span_m", type=float, default=300); ap.add_argument("--row_width", type=float, default=30)
    ap.add_argument("--edge_width", type=float, default=15); ap.add_argument("--out", default="data/spans.csv")
    main(ap.parse_args())
