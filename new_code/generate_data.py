import pandas as pd
import numpy as np

def generate_csvs():
    np.random.seed(42)
    n_lines = 100
    
    # 1. Transmission Lines
    df_lines = pd.DataFrame({
        "line_id": [f"LN-{1000+i}" for i in range(n_lines)],
        "start_lat": np.random.uniform(34.0, 34.5, n_lines),
        "start_lon": np.random.uniform(-118.5, -117.8, n_lines),
        "end_lat": np.random.uniform(34.0, 34.5, n_lines),
        "end_lon": np.random.uniform(-118.5, -117.8, n_lines),
        "voltage_kv": np.random.choice([132, 220, 400], n_lines),
        "substation_id": np.random.choice(["SUB-Alpha", "SUB-Beta", "SUB-Gamma", "SUB-Delta"], n_lines)
    })
    df_lines.to_csv("transmission_lines.csv", index=False)
    
    # 2. Vegetation & LiDAR Metrics
    df_veg = pd.DataFrame({
        "line_id": df_lines["line_id"],
        "distance_to_conductor_m": np.random.uniform(0.5, 15.0, n_lines),
        "tree_height_m": np.random.uniform(2.0, 25.0, n_lines),
        "growth_rate_cm_yr": np.random.uniform(10, 50, n_lines),
        "soil_moisture_pct": np.random.uniform(10, 95, n_lines)
    })
    df_veg.to_csv("vegetation_lidar_satellite.csv", index=False)
    
    # 3. Substations
    df_subs = pd.DataFrame({
        "substation_id": ["SUB-Alpha", "SUB-Beta", "SUB-Gamma", "SUB-Delta"],
        "status": ["Active", "Active", "Active", "Active"],
        "available_capacity_mw": [150, 120, 200, 180],
        "lat": [34.03, 34.15, 34.22, 34.30],
        "lon": [-118.28, -118.35, -118.02, -118.15]
    })
    df_subs.to_csv("substations.csv", index=False)
    print("Successfully generated all three CSV datasets!")

if __name__ == "__main__":
    generate_csvs()
