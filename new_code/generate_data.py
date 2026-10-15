"""Synthetic span-level data so the pipeline runs end to end.
REPLACE with real data from build_features.py + your trim/outage records."""
import numpy as np, pandas as pd
from common import SPECIES, required_clearance

rng = np.random.default_rng(42)
L, S = 40, 60
n = L * S
line = np.repeat(np.arange(L), S); k = np.tile(np.arange(S), L)
lat0 = 48.4 + rng.uniform(0, .6, L); lon0 = 14.0 + rng.uniform(0, .9, L)
brg = rng.uniform(0, 2 * np.pi, L)
kv = rng.choice([33, 66, 110, 230, 400], L, p=[.2, .25, .3, .15, .1])
wild = rng.integers(0, 4, L)

lat = lat0[line] + np.cos(brg[line]) * 0.0035 * k + rng.normal(0, 2e-4, n)
lon = lon0[line] + np.sin(brg[line]) * 0.005 * k + rng.normal(0, 2e-4, n)
voltage = kv[line]
species = rng.choice(SPECIES, n, p=[.35, .3, .25, .1])
g0 = pd.Series(species).map({"conifer": .5, "broadleaf": .7, "mixed": .6, "shrub": .3}).values
precip = rng.normal(850, 150, n).clip(400, 1400)
growth = (g0 * precip / 850 * rng.lognormal(0, .25, n)).round(2)
cond_h = 8 + voltage / 400 * 14 + rng.normal(0, 1, n)
req = required_clearance(voltage)
canopy = np.clip(cond_h - req - rng.normal(4, 3.5, n), 1, 40)       # true canopy at survey
margin = cond_h - canopy - req
row_w = np.clip(15 + voltage / 10 + rng.normal(0, 4, n), 12, None)
edge = np.clip(canopy * rng.uniform(.8, 1.3, n), 2, 45)
wind = rng.normal(22, 4, n)
age = rng.integers(30, 1100, n)                                      # days since LiDAR survey

grow_in = margin - growth * (age / 365 + 1) + rng.normal(0, 1, n) < 0
fall_in = rng.random(n) < np.where(edge / (row_w / 2) > 1, .08 * wind / 22, .005)
y = (grow_in | fall_in).astype(int)

df = pd.DataFrame({
    "span_id": [f"L{a}-{b}" for a, b in zip(line, k)], "line_id": [f"L{a}" for a in line],
    "lat": lat, "lon": lon, "voltage_kv": voltage, "species": species,
    "conductor_height_m": cond_h, "canopy_height_max_m": canopy + rng.normal(0, 1.2, n),  # LiDAR noise
    "edge_tree_height_m": edge + rng.normal(0, 1.5, n), "row_width_m": row_w,
    "growth_rate_m_yr": growth,
    "ndvi_mean": np.clip(.3 + .012 * canopy + rng.normal(0, .05, n), .1, .95),
    "ndvi_trend": growth * .02 + rng.normal(0, .01, n),
    "slope_deg": np.abs(rng.normal(8, 5, n)), "precip_mm_yr": precip, "wind_gust_p95_ms": wind,
    "days_since_survey": age, "customers_served": rng.lognormal(6, 1.2, n).astype(int),
    "wildfire_zone": wild[line], "encroach_12m": y})
df.round(3).to_csv("data/spans.csv", index=False)
print(f"{len(df)} spans, positive rate {y.mean():.1%} -> data/spans.csv")
