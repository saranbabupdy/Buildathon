import os
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

CLASS_COLORS = np.array([
    [0, 0, 0],
    [0, 255, 0],
    [110, 110, 110],
], dtype=np.uint8)

st.set_page_config(page_title="VEPL Vegetation Screening", layout="wide")
st.title("AI Vegetation Encroachment Screening")
st.caption("VEPL semantic-segmentation demo — predictions are for screening and require human verification.")

model_path = st.text_input("Trained model path", "outputs/vepl_unet.keras")
uploaded = st.file_uploader("Upload a drone image", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.subheader("Input image")
    st.image(image, use_container_width=True)

    if st.button("Run vegetation detection"):
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}. Train the model first.")
        else:
            with st.spinner("Running model..."):
                model = tf.keras.models.load_model(model_path)
                size = int(model.input_shape[1])
                resized = image.resize((size, size), Image.Resampling.BILINEAR)
                x = np.asarray(resized, dtype=np.float32) / 255.0
                probabilities = model.predict(x[None, ...], verbose=0)[0]
                labels = np.argmax(probabilities, axis=-1).astype(np.uint8)
                colors = CLASS_COLORS[labels]

                overlay = np.asarray(resized).copy()
                veg = labels == 1
                line = labels == 2
                overlay[veg] = (0.45 * overlay[veg] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
                overlay[line] = (0.45 * overlay[line] + 0.55 * np.array([110, 110, 110])).astype(np.uint8)

                veg_pct = float(veg.mean() * 100)
                line_pct = float(line.mean() * 100)

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Predicted class mask")
                st.image(colors, caption="Black=background, green=vegetation, grey=visible corridor")
            with c2:
                st.subheader("Overlay")
                st.image(overlay, caption="Predictions over the resized input image")

            m1, m2 = st.columns(2)
            m1.metric("Image pixels predicted as vegetation", f"{veg_pct:.2f}%")
            m2.metric("Image pixels predicted as corridor", f"{line_pct:.2f}%")
            st.warning(
                "These percentages describe the uploaded image, not actual conductor clearance. "
                "The model does not establish a safety violation. Verify detections with suitable "
                "measurements and an authorized reviewer."
            )
