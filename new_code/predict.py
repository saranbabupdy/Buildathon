"""Score spans -> probability, risk score (probability x consequence), tier, priority rank."""
import argparse, joblib, pandas as pd
from common import add_features, risk_score, to_tier


def main(a):
    bundle = joblib.load(a.model)
    d = add_features(pd.read_csv(a.input))
    d["prob_encroach_12m"] = bundle["model"].predict_proba(d[bundle["features"]])[:, 1]
    d["risk_score"] = risk_score(d["prob_encroach_12m"], d)
    d["tier"] = to_tier(d["risk_score"])
    d = d.sort_values("risk_score", ascending=False)
    d.insert(0, "priority_rank", range(1, len(d) + 1))
    d.round(4).to_csv(a.out, index=False)
    print(d["tier"].value_counts().to_string(), f"\nsaved {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/spans.csv")
    ap.add_argument("--model", default="models/veg_model.joblib")
    ap.add_argument("--out", default="outputs/predictions.csv")
    main(ap.parse_args())
