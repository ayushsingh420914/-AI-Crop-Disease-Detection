"""Streamlit web interface for AI Yield: Crop Disease Detection and Management."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from predictor import DEFAULT_MODEL_DIR, load_trained_model, predict_pil_image
from recommendations import DISCLAIMER, get_recommendations

st.set_page_config(
    page_title="AI Yield | Crop Disease Detection",
    page_icon="🌱",
    layout="centered",
)

st.title("🌱 AI Yield")
st.subheader("Crop Disease Detection and Management")
st.write(
    "Upload a clear photo of one crop leaf. The model will return its most likely "
    "PlantVillage class and general crop-care steps."
)
st.info(
    "This is an educational prototype. It screens leaf images; it does not guarantee "
    "a diagnosis or calculate actual crop yield."
)

MODEL_PATH = Path(DEFAULT_MODEL_DIR) / "crop_disease_model.pth"


@st.cache_resource(show_spinner="Loading trained AI model...")
def get_model_bundle():
    return load_trained_model(DEFAULT_MODEL_DIR)


if not MODEL_PATH.exists():
    st.warning("The trained model has not been created yet.")
    st.markdown(
        "**Run these commands from the project folder after downloading the dataset:**\n\n"
        "```bash\n"
        "python train.py --data_dir data/raw --epochs 12\n"
        "streamlit run app.py\n"
        "```"
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Use a focused, well-lit image showing one leaf clearly.",
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read this image: {exc}")
        st.stop()

    st.image(image, caption="Uploaded leaf image", use_container_width=True)

    try:
        model, class_names, image_transform, device = get_model_bundle()
        with st.spinner("Analysing image..."):
            predictions = predict_pil_image(
                image, model, class_names, image_transform, device, top_k=3
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not run the model: {exc}")
        st.stop()

    best = predictions[0]
    advice = get_recommendations(str(best["label"]))

    st.markdown("### Prediction")
    st.success(f"{advice['crop']} — {advice['condition']}")
    st.metric("Model confidence score", f"{float(best['confidence']):.1%}")
    st.caption(
        "Confidence is the model's score among its known classes; it is not a "
        "verified probability that the diagnosis is correct."
    )

    if float(best["confidence"]) < 0.55:
        st.warning(
            "The model is not very confident. Retake the photo in natural light "
            "and ask a local agriculture expert to confirm the symptoms."
        )

    st.markdown("### Management suggestions")
    for action in advice["actions"]:
        st.markdown(f"- {action}")

    with st.expander("View the model's top 3 predictions"):
        for item in predictions:
            st.write(f"**{item['label'].replace('___', ' — ').replace('_', ' ')}**")
            st.progress(float(item["confidence"]))
            st.caption(f"Score: {float(item['confidence']):.2%}")

    st.warning(DISCLAIMER)
else:
    st.caption("No image is uploaded yet. Supported formats: JPG, PNG, WebP, BMP.")

st.divider()
st.caption("AI Yield • Image classification prototype • Model: MobileNetV3-Small")
