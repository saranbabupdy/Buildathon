import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf

COLORS = np.array([
    [0, 0, 0],        # background
    [0, 255, 0],      # vegetation
    [110, 110, 110],  # visible power-line corridor
], dtype=np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/vepl_unet.keras")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    original = Image.open(args.image).convert("RGB")
    resized = original.resize((args.size, args.size), Image.Resampling.BILINEAR)
    x = np.asarray(resized, dtype=np.float32) / 255.0

    model = tf.keras.models.load_model(args.model)
    probs = model.predict(x[None, ...], verbose=0)[0]
    labels = np.argmax(probs, axis=-1).astype(np.uint8)
    mask_rgb = COLORS[labels]

    # Green overlay for vegetation; grey for corridor class.
    overlay = np.asarray(resized).copy()
    vegetation = labels == 1
    corridor = labels == 2
    overlay[vegetation] = (
        0.45 * overlay[vegetation] + 0.55 * np.array([0, 255, 0])
    ).astype(np.uint8)
    overlay[corridor] = (
        0.45 * overlay[corridor] + 0.55 * np.array([110, 110, 110])
    ).astype(np.uint8)

    stem = os.path.splitext(os.path.basename(args.image))[0]
    Image.fromarray(mask_rgb).save(os.path.join(args.output_dir, f"{stem}_mask.png"))
    Image.fromarray(overlay).save(os.path.join(args.output_dir, f"{stem}_overlay.png"))

    veg_pct = float(np.mean(labels == 1) * 100)
    corridor_pct = float(np.mean(labels == 2) * 100)
    print(f"Vegetation pixels: {veg_pct:.2f}%")
    print(f"Corridor-class pixels: {corridor_pct:.2f}%")
    print("Saved mask and overlay to:", args.output_dir)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Input image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Predicted overlay (green=vegetation, grey=corridor)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
