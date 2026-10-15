"""Shared feature engineering + risk logic (used by train, predict, dashboard)."""
import numpy as np
import pandas as pd

TARGET = "encroach_12m"          # 1 = clearance violation / trim ticket / veg outage within 12 months
SPECIES = ["conifer", "broadleaf", "mixed", "shrub"]
BASE = ["voltage_kv", "conductor_height_m", "canopy_height_max_m", "edge_tree_height_m",
        "row_width_m", "growth_rate_m_yr", "ndvi_mean", "ndvi_trend", "slope_deg",
        "precip_mm_yr", "wind_gust_p95_ms", "days_since_survey", "wildfire_zone"]
DERIVED = ["req_clearance_m", "clearance_margin_m", "projected_margin_m",
           "edge_hazard_ratio", "species_code"]
FEATURES = BASE + DERIVED

TIER_COLORS = {"Critical": [214, 39, 40], "High": [255, 127, 14],
               "Medium": [255, 215, 0], "Low": [44, 160, 44]}
TIER_HEX = {"Critical": "#d62728", "High": "#ff7f0e", "Medium": "#ffd700", "Low": "#2ca02c"}


def required_clearance(kv):
    """ILLUSTRATIVE minimum vegetation clearance (m). Replace with your regulatory standard."""
    return np.interp(kv, [33, 66, 110, 230, 400], [1.5, 2.0, 3.0, 4.5, 6.0])


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["req_clearance_m"] = required_clearance(d["voltage_kv"])
    d["clearance_margin_m"] = d["conductor_height_m"] - d["canopy_height_max_m"] - d["req_clearance_m"]
    horizon_yr = d["days_since_survey"] / 365.0 + 1.0          # survey age + 12-month look-ahead
    d["projected_margin_m"] = d["clearance_margin_m"] - d["growth_rate_m_yr"] * horizon_yr
    d["edge_hazard_ratio"] = d["edge_tree_height_m"] / (d["row_width_m"] / 2).clip(lower=1)
    d["species_code"] = d["species"].map({s: i for i, s in enumerate(SPECIES)})
    return d


def consequence(d: pd.DataFrame) -> pd.Series:
    """0-1 impact score: customers, wildfire zone (0-3), voltage."""
    c = np.log1p(d["customers_served"]) / np.log1p(d["customers_served"].max())
    return (0.4 * c + 0.3 * d["wildfire_zone"].fillna(0) / 3 + 0.3 * d["voltage_kv"] / 400).clip(0, 1)


def risk_score(prob, d):
    return 100 * prob * (0.4 + 0.6 * consequence(d))


def to_tier(score):
    return pd.cut(score, [-1, 15, 30, 50, 101], labels=["Low", "Medium", "High", "Critical"]).astype(str)
