"""Run: streamlit run dashboard.py"""
from pathlib import Path
import pandas as pd, plotly.express as px, pydeck as pdk, streamlit as st
from common import TIER_COLORS, TIER_HEX

st.set_page_config(page_title="Vegetation Encroachment Risk", layout="wide")
st.title("⚡ Vegetation Encroachment Risk Dashboard")

up = st.sidebar.file_uploader("Predictions CSV (from predict.py)", type="csv")
path = Path("outputs/predictions.csv")
if up is None and not path.exists():
    st.error("Run predict.py first or upload a predictions CSV."); st.stop()
df = pd.read_csv(up or path)

tiers = ["Critical", "High", "Medium", "Low"]
sel = st.sidebar.multiselect("Risk tier", tiers, default=tiers)
sel_lines = st.sidebar.multiselect("Line", sorted(df.line_id.unique()))
sel_kv = st.sidebar.multiselect("Voltage (kV)", sorted(df.voltage_kv.unique()))
minp = st.sidebar.slider("Min probability", 0.0, 1.0, 0.0, 0.05)
cap = st.sidebar.slider("Spans crews can treat this cycle", 10, 500, 100, 10)

f = df[df.tier.isin(sel) & (df.prob_encroach_12m >= minp)]
if sel_lines: f = f[f.line_id.isin(sel_lines)]
if sel_kv: f = f[f.voltage_kv.isin(sel_kv)]
if f.empty: st.warning("No spans match the filters."); st.stop()

hi = f[f.tier.isin(["Critical", "High"])]
top = f.nlargest(cap, "risk_score")
c = st.columns(5)
c[0].metric("Spans", f"{len(f):,}")
c[1].metric("Critical + High", f"{len(hi):,}")
c[2].metric("Avg P(encroach 12m)", f"{f.prob_encroach_12m.mean():.1%}")
c[3].metric("Customers on Critical/High", f"{int(hi.customers_served.sum()):,}")
c[4].metric(f"Risk covered by top {cap}", f"{top.risk_score.sum() / f.risk_score.sum():.0%}")

m = f.assign(color=f.tier.map(TIER_COLORS), prob_pct=(f.prob_encroach_12m * 100).round(1))
layer = pdk.Layer("ScatterplotLayer", m, get_position=["lon", "lat"], get_fill_color="color",
                  get_radius=150, radius_min_pixels=3, pickable=True)
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(latitude=m.lat.mean(), longitude=m.lon.mean(), zoom=8),
    map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    tooltip={"html": "<b>{span_id}</b><br/>Tier: {tier}<br/>P(12m): {prob_pct}%<br/>"
                     "Risk: {risk_score}<br/>Projected margin: {projected_margin_m} m"}))

a, b = st.columns(2)
tc = f.tier.value_counts().reindex(tiers).fillna(0).reset_index()
tc.columns = ["tier", "spans"]
a.plotly_chart(px.bar(tc, x="tier", y="spans", color="tier", color_discrete_map=TIER_HEX,
                      title="Spans by risk tier"), use_container_width=True)
ln = f.groupby("line_id").risk_score.mean().nlargest(15).reset_index()
b.plotly_chart(px.bar(ln, x="risk_score", y="line_id", orientation="h",
                      title="Top 15 lines by mean risk").update_yaxes(autorange="reversed"),
               use_container_width=True)
a.plotly_chart(px.scatter(f, x="projected_margin_m", y="growth_rate_m_yr", color="tier",
                          color_discrete_map=TIER_HEX, hover_name="span_id",
                          title="Projected clearance margin vs growth rate"), use_container_width=True)
b.plotly_chart(px.histogram(f, x="prob_encroach_12m", nbins=30,
                            title="Distribution of encroachment probability"), use_container_width=True)

st.subheader(f"Work-order priority list (top {cap})")
cols = ["priority_rank", "span_id", "line_id", "tier", "risk_score", "prob_encroach_12m",
        "projected_margin_m", "clearance_margin_m", "growth_rate_m_yr", "edge_hazard_ratio",
        "customers_served", "wildfire_zone"]
st.dataframe(top[cols], use_container_width=True, hide_index=True)
st.download_button("Download priority list (CSV)", top[cols].to_csv(index=False), "priority_spans.csv")
