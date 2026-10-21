"""
Shared drone-image model code: feature extraction, CNN builder, saved-model
loading and prediction.  Used by both train_drone_model.py and app.py.

Input convention: corridor-aligned tiles, i.e. the conductor runs roughly
left-to-right through the middle of the image.
"""
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

CLASSES = ["clear", "near", "encroaching"]
# How much each class counts towards a 0-1 image risk score.
CLASS_SEVERITY = {"clear": 0.0, "near": 0.5, "encroaching": 1.0}

TILE = 128
MODEL_DIR = Path(__file__).resolve().parent / "models"
BUNDLE_PATH = MODEL_DIR / "drone_best.joblib"


# ---------------------------------------------------------------------
# Hand-crafted features (classical models)
# ---------------------------------------------------------------------
def _prep(img):
    return np.asarray(
        img.convert("RGB").resize((TILE, TILE), Image.BILINEAR), dtype=float
    ) / 255.0


def vegetation_mask(arr):
    """Excess-Green index, a standard RGB vegetation index."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    exg = 2 * g - r - b
    return exg, exg > 0.15


def find_conductor_row(arr):
    """Darkest horizontal band within the middle of the tile."""
    gray = arr.mean(axis=2)
    lo, hi = int(TILE * 0.3), int(TILE * 0.7)
    return lo + int(np.argmin(gray[lo:hi].mean(axis=1)))


def extract_features(img):
    arr = _prep(img)
    exg, mask = vegetation_mask(arr)
    row = find_conductor_row(arr)
    ys = np.arange(TILE)[:, None]
    dist = np.abs(ys - row) / TILE  # distance to conductor, fraction of tile

    feats = []

    # Closest vegetation pixel to the conductor, per column.
    gaps = np.where(mask, dist, 1.0).min(axis=0)
    feats += [
        gaps.min(),
        np.percentile(gaps, 5),
        np.percentile(gaps, 20),
        np.percentile(gaps, 50),
        (gaps < 0.03).mean(),
        (gaps < 0.08).mean(),
        (gaps < 0.15).mean(),
    ]

    # Vegetation cover in distance bands around the conductor.
    edges = [0, 0.03, 0.06, 0.10, 0.15, 0.22, 0.32, 0.55]
    for a, b in zip(edges[:-1], edges[1:]):
        band = (dist >= a) & (dist < b)
        feats.append(mask[np.broadcast_to(band, mask.shape)].mean()
                     if band.any() else 0.0)

    # Global colour / vegetation statistics.
    feats += [mask.mean(), exg.mean(), exg.std()]
    feats += list(arr.mean(axis=(0, 1))) + list(arr.std(axis=(0, 1)))
    return np.asarray(feats, dtype=np.float32)


# ---------------------------------------------------------------------
# CNN (optional, needs PyTorch + torchvision)
# ---------------------------------------------------------------------
def build_cnn(arch, num_classes=len(CLASSES), pretrained=True):
    import torch.nn as nn
    from torchvision import models

    def make(factory, weights_enum):
        if pretrained:
            try:
                return factory(weights=weights_enum.DEFAULT)
            except Exception as exc:  # offline, old torchvision, ...
                print(f"  [warn] pretrained weights unavailable for {arch}: "
                      f"{exc}\n         training from scratch instead.")
        return factory(weights=None)

    if arch == "mobilenet_v3_small":
        m = make(models.mobilenet_v3_small,
                 models.MobileNet_V3_Small_Weights)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "resnet18":
        m = make(models.resnet18, models.ResNet18_Weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = make(models.efficientnet_b0, models.EfficientNet_B0_Weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return m


_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor_array(img, size=TILE):
    """PIL -> normalised CHW float32 array."""
    a = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR),
                   dtype=np.float32) / 255.0
    a = (a - _MEAN) / _STD
    return a.transpose(2, 0, 1)


# ---------------------------------------------------------------------
# Saved bundle: load + predict
# ---------------------------------------------------------------------
def load_bundle(path=BUNDLE_PATH):
    path = Path(path)
    if not path.exists():
        return None
    bundle = joblib.load(path)
    if bundle["kind"] == "torch":
        import torch

        net = build_cnn(bundle["arch"], pretrained=False)
        weights = path.parent / bundle["weights_file"]
        net.load_state_dict(torch.load(weights, map_location="cpu"))
        net.eval()
        bundle["net"] = net
    return bundle


def predict_proba(bundle, img):
    """Return {class_name: probability} for one PIL image."""
    classes = bundle["classes"]
    if bundle["kind"] == "sklearn":
        p = bundle["model"].predict_proba(extract_features(img)[None, :])[0]
    else:
        import torch

        x = torch.from_numpy(to_tensor_array(img)[None, ...])
        with torch.no_grad():
            p = torch.softmax(bundle["net"](x), dim=1)[0].numpy()
    return {c: float(v) for c, v in zip(classes, p)}


def image_risk(probs):
    """Collapse class probabilities into one 0-1 risk number."""
    return float(sum(CLASS_SEVERITY[c] * p for c, p in probs.items()))


# ---------------------------------------------------------------------
# Dashboard helpers (pure functions, easy to test)
# ---------------------------------------------------------------------
import re

_LINE_RE = re.compile(r"(LN-\d+)", re.IGNORECASE)


def match_line_id(filename, valid_ids):
    """Find a transmission-line ID such as LN-1004 inside a file name."""
    m = _LINE_RE.search(filename)
    if not m:
        return None
    candidate = m.group(1).upper()
    return candidate if candidate in set(valid_ids) else None


def fuse_risk(tabular_risk, img_risk, image_weight, floor_at=0.8):
    """Blend tabular and drone-image risk.

    No image            -> tabular risk only.
    Image risk >= floor -> never lower than the image risk itself, so a
                           confident sighting of canopy on the conductor
                           cannot be averaged away by a low tabular score.
    """
    if img_risk is None or np.isnan(img_risk):
        return float(tabular_risk)
    blend = (1 - image_weight) * tabular_risk + image_weight * img_risk
    if img_risk >= floor_at and image_weight > 0:
        blend = max(blend, img_risk)
    return float(blend)
