"""Train span-level 12-month encroachment classifier with line-grouped validation."""
import argparse, json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from common import add_features, FEATURES, TARGET


def make_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=500, max_leaf_nodes=31, min_samples_leaf=30,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, categorical_features=[FEATURES.index("species_code")],
        random_state=42)


def main(a):
    df = add_features(pd.read_csv(a.data))
    X, y, g = df[FEATURES], df[TARGET], df["line_id"]

    # Group by line so neighbouring spans never leak between train and test (spatial leakage)
    cv = cross_val_score(make_model(), X, y, groups=g, cv=GroupKFold(5), scoring="roc_auc")
    print(f"Grouped 5-fold ROC-AUC: {cv.mean():.3f} +/- {cv.std():.3f}")

    tr, te = next(GroupShuffleSplit(test_size=0.2, random_state=42).split(X, y, g))
    m = make_model().fit(X.iloc[tr], y.iloc[tr])
    p = m.predict_proba(X.iloc[te])[:, 1]
    top = np.argsort(-p)[: max(1, len(p) // 10)]
    metrics = {"cv_auc_mean": cv.mean(), "cv_auc_std": cv.std(),
               "test_auc": roc_auc_score(y.iloc[te], p),
               "test_pr_auc": average_precision_score(y.iloc[te], p),
               "test_brier": brier_score_loss(y.iloc[te], p),
               "precision_top10pct": float(y.iloc[te].values[top].mean()),
               "base_rate": float(y.mean())}
    print(json.dumps({k: round(float(v), 4) for k, v in metrics.items()}, indent=2))

    imp = permutation_importance(m, X.iloc[te], y.iloc[te], scoring="roc_auc", n_repeats=5, random_state=0)
    (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
       .sort_values("importance", ascending=False).to_csv("outputs/feature_importance.csv", index=False))

    final = make_model().fit(X, y)                       # refit on all data for deployment
    joblib.dump({"model": final, "features": FEATURES}, a.model)
    json.dump({k: float(v) for k, v in metrics.items()}, open("outputs/metrics.json", "w"), indent=2)
    print("saved", a.model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/spans.csv")
    ap.add_argument("--model", default="models/veg_model.joblib")
    main(ap.parse_args())
