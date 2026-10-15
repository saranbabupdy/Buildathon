# Vegetation Encroachment Risk (span-level ML + dashboard)

    pip install -r requirements.txt
    python generate_data.py     # demo data; skip when you have real data (build_features.py)
    python train.py             # grouped CV, hold-out metrics, model -> models/
    python predict.py           # outputs/predictions.csv (probability, risk score, tier, rank)
    streamlit run dashboard.py

Target `encroach_12m`: 1 if the span had a clearance violation, trim ticket or vegetation outage in the
following 12 months. Risk score = probability x consequence (customers, wildfire zone, voltage).
Clearance limits in common.py are illustrative: use your regulatory standard (e.g. NERC FAC-003).
