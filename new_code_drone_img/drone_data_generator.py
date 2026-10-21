"""
Synthetic drone-image generator for the GridAI vegetation pipeline.

It creates top-down, corridor-aligned image tiles in an ImageFolder layout:

    drone_images/
        clear/         vegetation well away from the conductor
        near/          vegetation approaching the conductor
        encroaching/   canopy touching / overlapping the conductor

The class is decided by the *real geometry* of the tile (gap between the
nearest canopy edge and the conductor), so labels are exact.

IMPORTANT: these images are a stand-in so the pipeline runs end to end.
Accuracy measured on them says nothing about real-world performance.
To use real imagery, put your own labelled tiles into the same folder layout
and re-run train_drone_model.py.
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["clear", "near", "encroaching"]
IMG_SIZE = 128

# Gap between the nearest canopy edge and the conductor, as a fraction of
# the tile height.  Used to define the classes.
ENCROACH_MAX = 0.03   # gap < 3 %   -> encroaching (canopy touches the line)
NEAR_MAX = 0.15       # gap < 15 %  -> near


def _ground(rng, size):
    base = np.array(
        [[120, 100, 70], [140, 120, 85], [105, 95, 65], [150, 135, 100]][
            rng.integers(0, 4)
        ],
        dtype=float,
    )
    coarse = rng.integers(0, 255, (6, 6), dtype=np.uint8)
    coarse = np.asarray(
        Image.fromarray(coarse).resize((size, size), Image.BICUBIC), float
    )
    coarse = coarse / 255.0 - 0.5
    fine = rng.normal(0, 9, (size, size, 3))
    img = base + coarse[..., None] * 45 + fine
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _gap_range(label):
    """Allowed range (fraction of tile height) of the closest canopy gap."""
    if label == "clear":
        return NEAR_MAX + 0.04, 0.42
    if label == "near":
        return ENCROACH_MAX + 0.02, NEAR_MAX - 0.01
    return -0.06, ENCROACH_MAX - 0.01


def make_tile(label, rng, size=IMG_SIZE):
    img = _ground(rng, size)
    draw = ImageDraw.Draw(img, "RGBA")

    line_y = size / 2 + rng.integers(-4, 5)

    # Cleared right-of-way strip and the conductor itself.
    strip = size * 0.07
    draw.rectangle(
        [0, line_y - strip, size, line_y + strip],
        fill=(170, 150, 110, 70),
    )
    draw.line([(0, line_y), (size, line_y)], fill=(35, 35, 40, 255), width=2)
    # Pylons / cross-arms.
    for x in rng.choice(np.arange(10, size - 10), size=2, replace=False):
        draw.rectangle(
            [x - 3, line_y - 5, x + 3, line_y + 5], fill=(60, 60, 65, 255)
        )

    lo, hi = _gap_range(label)
    critical_gap = rng.uniform(lo, hi) * size

    canopies = []
    r = rng.uniform(0.07, 0.14) * size
    side = rng.choice([-1, 1])
    cy = line_y + side * (critical_gap + r)
    canopies.append((rng.uniform(0.1, 0.9) * size, cy, r))

    # Remaining trees must not be closer to the line than the critical one.
    for _ in range(rng.integers(5, 13)):
        r = rng.uniform(0.05, 0.13) * size
        gap = critical_gap + rng.uniform(0.02, 0.45) * size
        s = rng.choice([-1, 1])
        canopies.append((rng.uniform(0, size), line_y + s * (gap + r), r))

    greens = [(40, 110, 45), (55, 130, 55), (35, 95, 40), (80, 140, 50),
              (95, 130, 55)]  # last two: drier / yellower foliage
    for cx, cy, r in canopies:
        # soft shadow, then the canopy with a lighter top
        draw.ellipse(
            [cx - r + 3, cy - r + 3, cx + r + 3, cy + r + 3],
            fill=(20, 25, 20, 90),
        )
        g = greens[rng.integers(0, len(greens))]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=g + (255,))
        draw.ellipse(
            [cx - r * 0.6, cy - r * 0.7, cx + r * 0.3, cy + r * 0.2],
            fill=tuple(min(255, c + 25) for c in g) + (160,),
        )

    # Camera effects: blur, brightness, sensor noise.
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.1)))
    arr = np.asarray(img, float)
    arr = arr * rng.uniform(0.75, 1.2) + rng.normal(0, 4, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def generate_drone_dataset(root="drone_images", n_per_class=300, seed=7):
    root = Path(root)
    rng = np.random.default_rng(seed)
    for label in CLASSES:
        out = root / label
        out.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            make_tile(label, rng).save(out / f"{label}_{i:04d}.png")
    return root


def generate_inspection_samples(
    line_ids, root="drone_samples", seed=21, labels=None
):
    """Write a few demo 'drone uploads' named after transmission lines."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    labels = labels or [CLASSES[i % 3] for i in range(len(line_ids))]
    for line_id, label in zip(line_ids, labels):
        make_tile(label, rng).save(root / f"{line_id}_span.png")
    return root


if __name__ == "__main__":
    generate_drone_dataset()
    print("Synthetic drone tiles written to ./drone_images")
