import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import pydeck as pdk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. DATA GENERATION & SIMULATION ENGINE
# ---------------------------------------------------------
@st.cache_data
def load_utility_data():
    np.random.seed(42)
    n_lines = 50
    
    # Generate synthetic transmission lines
    lines_data = {
        "line_id": [f"LN-{100+i}" for i in range(n_lines)],
        "start_lat": np.random.uniform(34.0, 34.5, n_lines),
        "start_lon": np.random.uniform(-118.5, -117.8, n_lines),
        "end_lat": np.random.uniform(34.0, 34.5, n_lines),
        "end_lon": np.random.uniform(-118.5, -117.8, n_lines),
        "voltage_kv": np.random.choice([132, 220, 400], n_lines),
        "distance_to_conductor_m": np.random.uniform(0.5, 15.0, n_lines),
        "tree_height_m": np.random.uniform(2.0, 25.0, n_lines),
        "growth_rate_cm_yr": np.random.uniform(10, 50, n_lines),
        "soil_moisture_pct": np.random.uniform(10, 95, n_lines),
        "status": np.random.choice(["Normal", "Hazardous-Storm"], n_lines, p=[0.8, 0.2])
    }
    df_lines = pd.DataFrame(lines_data)
    
    # Substations Data
    sub_data = {
        "substation_id": ["SUB-Alpha", "SUB-Beta", "SUB-Gamma", "SUB-Delta"],
        "lat": [34.42, 34.05, 34.30, 34.15],
        "lon": [-118.40, -118.20, -117.95, -118.00],
        "available_capacity_mw": [120, 80, 150, 95],
        "status": ["Active", "Active", "Active", "Active"]
    }
    df_subs = pd.DataFrame(sub_data)
    
    return df_lines, df_subs

# ---------------------------------------------------------
# 2. AI VEGETATION RISK PREDICTION MODEL
# ---------------------------------------------------------
@st.cache_resource
def train_risk_model(df):
    # Create synthetic target label: Risk of flashover / encroachment failure
    # High risk if distance < 3m and tree height > 10m or high growth rate & soil moisture
    conditions = (df["distance_to_conductor_m"] < 4.0) & (df["tree_height_m"] > 8.0)
    df["target_high_risk"] = conditions.astype(int)
    # Add some noise
    noise_indices = np.random.choice(df.index, size=3, replace=False)
    df.loc[noise_indices, "target_high_risk"] = 1 - df.loc[noise_indices, "target_high_risk"]
    
    X = df[["distance_to_conductor_m", "tree_height_m", "growth_rate_cm_yr", "soil_moisture_pct", "voltage_kv"]]
    y = df["target_high_risk"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# ---------------------------------------------------------
# 3. DISASTER ISOLATION & BACKUP RESTORATION PLANNER
# ---------------------------------------------------------
def run_disaster_isolation_and_backup(df_lines, df_subs, disaster_active):
    G = nx.Graph()
    
    # Add substations as nodes
    for _, sub in df_subs.iterrows():
        G.add_node(sub["substation_id"], type="substation", status=sub["status"], capacity=sub["available_capacity_mw"])
        
    # Add lines as edges
    for _, line in df_lines.iterrows():
        # Assign mock source/target substations for network topology
        sub_sources = df_subs["substation_id"].values
        u = np.random.choice(sub_sources)
        v = np.random.choice(sub_sources)
        while u == v:
            v = np.random.choice(sub_sources)
            
        line_status = "Isolated" if (disaster_active and line["risk_score"] > 0.7) else "Live"
        G.add_edge(u, v, line_id=line["line_id"], weight=line["distance_to_conductor_m"], status=line_status, voltage=line["voltage_kv"])
        
    # Isolate hazardous lines during disasters
    isolated_lines = []
    backup_plan = []
    
    for u, v, data in G.edges(data=True):
        if data["status"] == "Isolated":
            isolated_lines.append(data["line_id"])
            # Generate backup restoration route via alternate active substation
            backup_plan.append({
                "Isolated_Line": data["line_id"],
                "Action": "Reroute power load through secondary microgrid tie",
                "Alternate_Node": u,
                "Status": "Standby Ready"
            })
            
    return isolated_lines, pd.DataFrame(backup_plan)

# ---------------------------------------------------------
# 4. STREAMLIT DASHBOARD UI
# ---------------------------------------------------------
st.set_page_config(page_title="GridAI: Vegetation & Disaster Response", layout="wide")

st.title("⚡ GridAI: Transmission Vegetation Encroachment & Disaster Response Dashboard")
st.markdown("AI-driven vegetation monitoring, predictive risk analytics, and automated emergency network reconfiguration.")

df_lines, df_subs = load_utility_data()
model = train_risk_model(df_lines)

# Predict risk scores for all lines
X_infer = df_lines[["distance_to_conductor_m", "tree_height_m", "growth_rate_cm_yr", "soil_moisture_pct", "voltage_kv"]]
df_lines["risk_score"] = model.predict_proba(X_infer)[:, 1]
df_lines["risk_ranking"] = df_lines["risk_score"].rank(ascending=False).astype(int)

# Sidebar Controls
st.sidebar.header("Control Panel")
disaster_mode = st.sidebar.checkbox("🚨 Activate Emergency / Storm Mode", value=False)
risk_threshold = st.sidebar.slider("Encroachment Risk Alert Threshold", 0.0, 1.0, 0.65, 0.05)

# Top KPI Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Monitored Transmission Spans", len(df_lines))
high_risk_count = len(df_lines[df_lines["risk_score"] >= risk_threshold])
col2.metric("High-Risk Vegetation Zones", high_risk_count, delta=f"-{high_risk_count} critical", delta_color="inverse")
col3.metric("Active Substations", len(df_subs[df_subs["status"]=="Active"]))
col4.metric("Grid Status", "EMERGENCY PROTOCOL" if disaster_mode else "Normal Operations", 
            delta="Alert" if disaster_mode else "Secure", delta_color="off" if not disaster_mode else "inverse")

# Run Disaster Logic if enabled
isolated_lines, df_backup = run_disaster_isolation_and_backup(df_lines, df_subs, disaster_mode)

if disaster_mode:
    st.error(f"⚠️ STORM MODE ACTIVE: {len(isolated_lines)} high-risk power lines automatically isolated to prevent ground faults.")

# Tabs Layout
tab1, tab2, tab3 = st.tabs(["🗺️ Geospatial Risk Map", "📊 Vegetation Risk Ranking", "⚡ Emergency Restoration Plan"])

with tab1:
    st.subheader("Transmission Corridor Risk Map & Asset Layer")
    
    # Prepare data for PyDeck mapping
    map_lines = []
    for _, row in df_lines.iterrows():
        color = [255, 0, 0, 200] if row["risk_score"] >= risk_threshold else [0, 255, 100, 160]
        if disaster_mode and row["line_id"] in isolated_lines:
            color = [255, 140, 0, 200] # Orange for isolated
            
        map_lines.append({
            "path": [[row["start_lon"], row["start_lat"]], [row["end_lon"], row["end_lat"]]],
            "color": color,
            "name": row["line_id"],
            "risk": round(row["risk_score"], 2)
        })
        
    layer = pdk.Layer(
        "LineLayer",
        map_lines,
        get_path="path",
        get_color="color",
        get_width=5,
        pickable=True,
        auto_highlight=True,
    )
    
    sub_layer = pdk.Layer(
        "ScatterplotLayer",
        df_subs,
        get_position=["lon", "lat"],
        get_color=[0, 100, 255, 200],
        get_radius=1200,
        pickable=True,
    )
    
    view_state = pdk.ViewState(latitude=34.25, longitude=-118.2, zoom=10, pitch=30)
    r = pdk.Deck(layers=[layer, sub_layer], initial_view_state=view_state, tooltip={"text": "Line/Substation ID"})
    st.pydeck_chart(r)

with tab2:
    st.subheader("Ranked Transmission Corridors by Vegetation Encroachment Risk")
    sorted_df = df_lines.sort_values(by="risk_score", ascending=False)[
        ["risk_ranking", "line_id", "voltage_kv", "distance_to_conductor_m", "tree_height_m", "risk_score", "status"]
    ]
    st.dataframe(sorted_df.style.background_gradient(subset=["risk_score"], cmap="Reds"), use_container_width=True)

with tab3:
    st.subheader("Emergency Power Source Isolation & Backup Restoration Plan")
    if disaster_mode:
        if not df_backup.empty:
            st.warning("The following lines have been isolated. Backup restoration sequence initiated:")
            st.dataframe(df_backup, use_container_width=True)
            
            st.markdown("### 🔌 Alternate Feeder Routing Graph Optimization")
            st.info("Dynamic load balancing is active. Secondary substations (SUB-Alpha, SUB-Gamma) have successfully assumed load redistribution for isolated pathways.")
        else:
            st.success("No lines required isolation under the current threshold settings.")
    else:
        st.info("System is in normal operation mode. Enable **Emergency / Storm Mode** in the sidebar to simulate disaster response, line isolation, and backup switching topology.")
