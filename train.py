import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split

# VEPL documented RGB mask colours
# 0 = background black, 1 = vegetation green, 2 = power-line corridor grey
CLASS_COLORS = np.array([
    [0, 0, 0],
    [0, 255, 0],
    [110, 110, 110],
], dtype=np.int16)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def stem_key(path):
    # Handles common naming patterns such as 0001.jpg and 0001.png.
    return Path(path).stem.lower()

def find_image_mask_dirs(root):
    dirs = [p for p in Path(root).rglob("*") if p.is_dir()]
    dirs = [Path(root)] + dirs
    image_candidates, mask_candidates = [], []
    for d in dirs:
        name = d.name.lower()
        files = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not files:
            continue
        if "mask" in name or "label" in name or "annotation" in name:
            mask_candidates.append((d, len(files)))
        elif "rgb" in name or "image" in name or "photo" in name:
            image_candidates.append((d, len(files)))

    # Choose folders with the largest number of image files.
    image_candidates.sort(key=lambda x: x[1], reverse=True)
    mask_candidates.sort(key=lambda x: x[1], reverse=True)
    if not image_candidates or not mask_candidates:
        raise RuntimeError(
            "Could not automatically identify image and mask folders.\n"
            "Use --image-dir and --mask-dir explicitly."
        )
    return image_candidates[0][0], mask_candidates[0][0]

def collect_files(folder):
    return sorted(
        p for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

def pair_images_masks(image_dir, mask_dir):
    images = collect_files(image_dir)
    masks = collect_files(mask_dir)
    mask_map = {}
    for p in masks:
        mask_map.setdefault(stem_key(p), []).append(p)

    pairs, missing = [], []
    for im in images:
        candidates = mask_map.get(stem_key(im), [])
        if len(candidates) == 1:
            pairs.append((im, candidates[0]))
        elif len(candidates) > 1:
            # Prefer mask filename with same relative parent when duplicates exist.
            same = [m for m in candidates if m.parent.name.lower() in ("mask", "masks")]
            if len(same) == 1:
                pairs.append((im, same[0]))
            else:
                missing.append(str(im))
        else:
            missing.append(str(im))

    print(f"Image folder: {image_dir}")
    print(f"Mask folder : {mask_dir}")
    print(f"Images found: {len(images)} | masks found: {len(masks)}")
    print(f"Pairs found : {len(pairs)} | unmatched images: {len(missing)}")
    if missing[:10]:
        print("First unmatched images:")
        for p in missing[:10]:
            print("  ", p)
    if len(pairs) < 10:
        raise RuntimeError(
            "Fewer than 10 image-mask pairs were found. Check folder paths and matching filenames."
        )
    return pairs

def mask_to_classes(mask_rgb):
    """Convert VEPL colour mask to integer labels 0,1,2."""
    pixels = mask_rgb.astype(np.int16)
    distances = ((pixels[..., None, :] - CLASS_COLORS[None, None, :, :]) ** 2).sum(axis=-1)
    labels = distances.argmin(axis=-1).astype(np.uint8)

    # Guard against masks with unexpected colors.
    min_dist = np.take_along_axis(distances, labels[..., None], axis=-1)[..., 0]
    bad_fraction = np.mean(min_dist > (45 ** 2 * 3))
    if bad_fraction > 0.05:
        unique = np.unique(mask_rgb.reshape(-1, 3), axis=0)
        raise ValueError(
            f"Mask has many colours that do not match documented VEPL colours "
            f"(approx. {bad_fraction:.1%} pixels). Sample colours: {unique[:12].tolist()}. "
            "Verify that you selected the labelled mask folder, not an RGB image folder."
        )
    return labels

def load_pair(image_path, mask_path, size):
    image = Image.open(image_path).convert("RGB").resize(
        (size, size), Image.Resampling.BILINEAR
    )
    mask = Image.open(mask_path).convert("RGB").resize(
        (size, size), Image.Resampling.NEAREST
    )
    x = np.asarray(image, dtype=np.float32) / 255.0
    y = mask_to_classes(np.asarray(mask, dtype=np.uint8))
    return x, y

def build_unet(input_shape=(256, 256, 3), n_classes=3):
    inputs = tf.keras.Input(shape=input_shape)

    def conv_block(x, filters):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return x

    c1 = conv_block(inputs, 16)
    p1 = tf.keras.layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, 32)
    p2 = tf.keras.layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, 64)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    b = conv_block(p3, 128)

    u3 = tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding="same")(b)
    u3 = tf.keras.layers.Concatenate()([u3, c3])
    c4 = conv_block(u3, 64)

    u2 = tf.keras.layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c4)
    u2 = tf.keras.layers.Concatenate()([u2, c2])
    c5 = conv_block(u2, 32)

    u1 = tf.keras.layers.Conv2DTranspose(16, 2, strides=2, padding="same")(c5)
    u1 = tf.keras.layers.Concatenate()([u1, c1])
    c6 = conv_block(u1, 16)

    outputs = tf.keras.layers.Conv2D(n_classes, 1, activation="softmax")(c6)
    return tf.keras.Model(inputs, outputs, name="VEPL_UNet")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="Root folder containing image and mask folders")
    parser.add_argument("--image-dir", default=None, help="Explicit image folder")
    parser.add_argument("--mask-dir", default=None, help="Explicit mask folder")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.image_dir and args.mask_dir:
        image_dir, mask_dir = args.image_dir, args.mask_dir
    elif args.data_dir:
        image_dir, mask_dir = find_image_mask_dirs(args.data_dir)
    else:
        parser.error("Provide --data-dir or both --image-dir and --mask-dir.")

    pairs = pair_images_masks(image_dir, mask_dir)
    # Prototype split. For research-quality evaluation, split by source orthomosaic/area.
    train_pairs, temp_pairs = train_test_split(
        pairs, test_size=0.30, random_state=args.seed
    )
    val_pairs, test_pairs = train_test_split(
        temp_pairs, test_size=0.50, random_state=args.seed
    )
    print(f"Train: {len(train_pairs)} | validation: {len(val_pairs)} | test: {len(test_pairs)}")

    def arrays_for(items):
        xs, ys = [], []
        for idx, (ip, mp) in enumerate(items, start=1):
            try:
                x, y = load_pair(ip, mp, args.size)
                xs.append(x)
                ys.append(y)
            except Exception as exc:
                raise RuntimeError(f"Failed to load pair:\n image={ip}\n mask={mp}\n error={exc}") from exc
            if idx % 100 == 0:
                print(f"Loaded {idx}/{len(items)} pairs")
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.uint8)

    print("Loading training data...")
    x_train, y_train = arrays_for(train_pairs)
    print("Loading validation data...")
    x_val, y_val = arrays_for(val_pairs)
    print("Loading test data...")
    x_test, y_test = arrays_for(test_pairs)

    model = build_unet((args.size, args.size, 3), 3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, "best_vepl_unet.keras"),
            monitor="val_loss", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate model on held-out test images.
    test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=1)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test pixel accuracy: {test_acc:.4f}")

    # Per-class IoU from confusion matrix.
    probs = model.predict(x_test, batch_size=args.batch_size, verbose=1)
    pred = np.argmax(probs, axis=-1)
    cm = np.zeros((3, 3), dtype=np.int64)
    for true_class in range(3):
        for pred_class in range(3):
            cm[true_class, pred_class] = np.sum(
                (y_test == true_class) & (pred == pred_class)
            )
    print("Confusion matrix (rows=true, columns=predicted):")
    print(cm)
    ious = []
    for c in range(3):
        tp = cm[c, c]
        denom = cm[c, :].sum() + cm[:, c].sum() - tp
        iou = float(tp / denom) if denom else float("nan")
        ious.append(iou)
    print("IoU background, vegetation, corridor:", [round(v, 4) for v in ious])

    model.save(os.path.join(args.output_dir, "vepl_unet.keras"))
    np.save(os.path.join(args.output_dir, "class_iou.npy"), np.asarray(ious))
    print(f"Finished. Model saved in {args.output_dir}/vepl_unet.keras")

if __name__ == "__main__":
    main()
