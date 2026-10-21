"""
Train and compare drone-image models, then save the best one.

    python train_drone_model.py                  # classical models only
    python train_drone_model.py --cnn            # + CNN transfer learning
    python train_drone_model.py --data my_tiles  # your own labelled tiles

Data layout (one sub-folder per class, names must match CLASSES):
    <data>/clear/*.png   <data>/near/*.png   <data>/encroaching/*.png

Selection rule: best macro-F1 on a validation split.  The winner is then
scored once on a held-out test split, which is what gets reported.
"""
import argparse
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from drone_data_generator import generate_drone_dataset
from drone_model import (
    BUNDLE_PATH,
    CLASSES,
    MODEL_DIR,
    build_cnn,
    extract_features,
    to_tensor_array,
)

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SEED = 42


def load_dataset(root):
    paths, labels = [], []
    for idx, cls in enumerate(CLASSES):
        folder = Path(root) / cls
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                paths.append(p)
                labels.append(idx)
    if len(paths) < 30:
        raise ValueError("Need at least ~30 labelled images to train.")
    return paths, np.array(labels)


def split_indices(y):
    idx = np.arange(len(y))
    tr, rest = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=SEED
    )
    va, te = train_test_split(
        rest, test_size=0.50, stratify=y[rest], random_state=SEED
    )
    return tr, va, te


def classical_candidates():
    return {
        "LogisticRegression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)
        ),
        "SVM-RBF": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(SVC(random_state=SEED), ensemble=False),
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=300, random_state=SEED, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=SEED
        ),
    }


# ---------------------------------------------------------------------
# CNN training
# ---------------------------------------------------------------------
def train_cnn(arch, imgs, y, tr, va, epochs, pretrained, log):
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = np.stack([to_tensor_array(im) for im in imgs])
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).long()

    net = build_cnn(arch, pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=epochs * int(np.ceil(len(tr) / 32))
    )
    loss_fn = nn.CrossEntropyLoss()

    def predict(idx):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 128):
                b = Xt[idx[i:i + 128]].to(device)
                out.append(torch.softmax(net(b), 1).cpu().numpy())
        return np.concatenate(out)

    best_f1, best_state = -1.0, None
    rng = np.random.default_rng(SEED)
    for ep in range(epochs):
        net.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), 32):
            b = order[i:i + 32]
            xb, yb = Xt[b].clone(), yt[b]
            # Light augmentation: flips keep the corridor geometry intact.
            if rng.random() < 0.5:
                xb = xb.flip(3)
            if rng.random() < 0.5:
                xb = xb.flip(2)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()
            sched.step()
        f1 = f1_score(y[va], predict(va).argmax(1), average="macro")
        log(f"    epoch {ep + 1}/{epochs}  val macro-F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.cpu()
    device_cpu_predict = predict  # closure still points at net

    def proba(idx):
        return device_cpu_predict(idx)

    return net, proba


# ---------------------------------------------------------------------
# Main routine (importable so the Streamlit app can call it)
# ---------------------------------------------------------------------
def run_training(data="drone_images", use_cnn=False, cnn_archs=None,
                 epochs=8, pretrained=True, n_synthetic=300, log=print):
    data = Path(data)
    if not data.exists():
        log(f"{data} not found - generating synthetic tiles "
            "(stand-in data, not real drone imagery).")
        generate_drone_dataset(data, n_per_class=n_synthetic)

    paths, y = load_dataset(data)
    log(f"Loaded {len(paths)} images "
        f"({', '.join(f'{c}: {(y == i).sum()}' for i, c in enumerate(CLASSES))})")
    imgs = [Image.open(p).convert("RGB") for p in paths]
    tr, va, te = split_indices(y)
    log(f"Split  train {len(tr)} / val {len(va)} / test {len(te)}")

    log("Extracting features ...")
    F = np.stack([extract_features(im) for im in imgs])

    results, fitted = [], {}

    for name, model in classical_candidates().items():
        t0 = time.time()
        model.fit(F[tr], y[tr])
        pred = model.predict(F[va])
        f1 = f1_score(y[va], pred, average="macro")
        results.append({"model": name, "type": "sklearn",
                        "val_macro_f1": f1,
                        "val_accuracy": float((pred == y[va]).mean()),
                        "train_seconds": time.time() - t0})
        fitted[name] = ("sklearn", model, lambda idx, m=model: m.predict_proba(F[idx]))
        log(f"  {name:22s} val macro-F1 {f1:.3f}")

    if use_cnn:
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except ImportError:
            log("PyTorch/torchvision not installed - skipping CNN models.")
            use_cnn = False
    if use_cnn:
        for arch in (cnn_archs or ["mobilenet_v3_small", "resnet18",
                                   "efficientnet_b0"]):
            log(f"  Fine-tuning {arch} ...")
            t0 = time.time()
            net, proba = train_cnn(arch, imgs, y, tr, va, epochs,
                                   pretrained, log)
            pred = proba(va).argmax(1)
            f1 = f1_score(y[va], pred, average="macro")
            results.append({"model": arch, "type": "torch",
                            "val_macro_f1": f1,
                            "val_accuracy": float((pred == y[va]).mean()),
                            "train_seconds": time.time() - t0})
            fitted[arch] = ("torch", net, proba)
            log(f"  {arch:22s} val macro-F1 {f1:.3f}")

    table = pd.DataFrame(results).sort_values(
        ["val_macro_f1", "val_accuracy"], ascending=False
    ).reset_index(drop=True)

    best_name = table.loc[0, "model"]
    kind, best, proba = fitted[best_name]
    test_pred = proba(te).argmax(1)
    test_f1 = f1_score(y[te], test_pred, average="macro")
    test_acc = float((test_pred == y[te]).mean())
    cm = confusion_matrix(y[te], test_pred, labels=range(len(CLASSES)))
    table["is_best"] = table["model"] == best_name

    MODEL_DIR.mkdir(exist_ok=True)
    table.to_csv(MODEL_DIR / "model_comparison.csv", index=False)

    bundle = {
        "kind": kind,
        "name": best_name,
        "classes": CLASSES,
        "trained_on": str(data),
        "n_images": int(len(paths)),
        "val_macro_f1": float(table.loc[0, "val_macro_f1"]),
        "test_macro_f1": float(test_f1),
        "test_accuracy": test_acc,
        "test_confusion_matrix": cm.tolist(),
    }
    if kind == "sklearn":
        bundle["model"] = best
    else:
        import torch

        bundle["arch"] = best_name
        bundle["weights_file"] = "drone_best_weights.pt"
        torch.save(best.state_dict(), MODEL_DIR / bundle["weights_file"])
    joblib.dump(bundle, BUNDLE_PATH)

    log("\nModel comparison (validation split):")
    log(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log(f"\nBest model: {best_name}")
    log(f"Held-out test  macro-F1 {test_f1:.3f}   accuracy {test_acc:.3f}")
    log("Confusion matrix (rows = true, cols = predicted; "
        f"order {CLASSES}):\n{cm}")
    log(f"Saved -> {BUNDLE_PATH}")
    return bundle, table


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="drone_images")
    ap.add_argument("--cnn", action="store_true",
                    help="also fine-tune CNNs (needs torch + torchvision)")
    ap.add_argument("--archs", nargs="+", default=None)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--n-synthetic", type=int, default=300)
    a = ap.parse_args()
    run_training(a.data, a.cnn, a.archs, a.epochs,
                 not a.no_pretrained, a.n_synthetic)
