"""Streamlit demo app for Global SAR Ship-Sea Classification.

Run from the project root:

streamlit run app/streamlit_app.py
"""

from pathlib import Path
import csv
from io import BytesIO, StringIO
import sys

import streamlit as st
import torch
from PIL import Image, ImageFilter, ImageEnhance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "simple_sar_classifier.pth"

sys.path.append(str(SRC_DIR))

from model import SimpleSARClassifier  # noqa: E402


CLASS_NAMES = {
    0: "sea",
    1: "ship",
}


DATASET_INFO = {
    "Total images": "2000",
    "Ship images": "1000",
    "Sea images": "1000",
    "Train split": "1400 images",
    "Validation split": "300 images",
    "Test split": "300 images",
    "Model": "Simple CNN",
    "Input size": "1 × 256 × 256 grayscale SAR patch",
}


def get_device() -> torch.device:
    """Use CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model() -> tuple[SimpleSARClassifier, torch.device]:
    """Load trained PyTorch model."""
    device = get_device()

    model = SimpleSARClassifier(num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, device


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert uploaded image to model input tensor."""
    image = image.convert("L")
    image = image.resize((256, 256))

    image_bytes = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    image_tensor = image_bytes.float().view(image.height, image.width) / 255.0

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def predict(image: Image.Image) -> tuple[str, float, list[float]]:
    """Predict class and confidence."""
    model, device = load_model()

    input_tensor = preprocess_image(image)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    predicted_label = int(torch.argmax(probabilities).item())
    predicted_class = CLASS_NAMES[predicted_label]
    confidence = float(probabilities[predicted_label].item())

    probability_list = [
        float(probabilities[0].item()),
        float(probabilities[1].item()),
    ]

    return predicted_class, confidence, probability_list


def results_to_csv(results: list[dict]) -> str:
    """Convert prediction results to CSV text."""
    if not results:
        return ""

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

    return output.getvalue()


def image_to_png_bytes(image: Image.Image) -> bytes:
    """Convert PIL image to downloadable PNG bytes."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def apply_processing(image: Image.Image, method: str) -> Image.Image:
    """Apply basic SAR image processing filters."""
    image = image.convert("L")

    if method == "Median filter":
        return image.filter(ImageFilter.MedianFilter(size=3))

    if method == "Sharpen":
        sharpened = image.filter(ImageFilter.SHARPEN)
        enhancer = ImageEnhance.Contrast(sharpened)
        return enhancer.enhance(1.2)

    if method == "Smooth":
        return image.filter(ImageFilter.SMOOTH_MORE)

    return image


def render_sidebar() -> None:
    """Render project information sidebar."""
    st.sidebar.title("📡 Project Info")
    st.sidebar.write("Global SAR Ship-Sea Classification")

    st.sidebar.subheader("Dataset")
    for key, value in DATASET_INFO.items():
        st.sidebar.write(f"**{key}:** {value}")

    st.sidebar.subheader("Important Note")
    st.sidebar.info(
        "This is a prototype system. Ship images and sea patches come from "
        "different SAR data sources, so results should be interpreted carefully."
    )


def render_classification_tab() -> None:
    """Render SAR ship-sea classification interface."""
    st.header("🚢 SAR Ship-Sea Classification")

    st.write(
        "Upload one or more SAR image patches. The trained CNN model will classify "
        "each image as **sea** or **ship**."
    )

    confidence_threshold = st.slider(
        "Confidence threshold for reliable prediction",
        min_value=0.50,
        max_value=0.99,
        value=0.70,
        step=0.01,
    )

    uploaded_files = st.file_uploader(
        "Upload SAR images",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="classification_upload",
    )

    if not uploaded_files:
        st.warning("Please upload one or more SAR image patches to start prediction.")
        return

    results = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)

        predicted_class, confidence, probabilities = predict(image)

        if confidence < confidence_threshold:
            decision = "UNCERTAIN"
        else:
            decision = predicted_class.upper()

        results.append(
            {
                "Image": uploaded_file.name,
                "Prediction": predicted_class.upper(),
                "Decision": decision,
                "Confidence (%)": round(confidence * 100, 2),
                "Sea Probability (%)": round(probabilities[0] * 100, 2),
                "Ship Probability (%)": round(probabilities[1] * 100, 2),
            }
        )

    st.subheader("Batch Prediction Results")
    st.dataframe(results, use_container_width=True)

    csv_text = results_to_csv(results)

    st.download_button(
        label="⬇️ Download prediction results as CSV",
        data=csv_text,
        file_name="sar_ship_sea_predictions.csv",
        mime="text/csv",
    )

    st.subheader("Image Preview and Individual Results")

    for uploaded_file, result in zip(uploaded_files, results):
        image = Image.open(uploaded_file)

        with st.expander(f"{uploaded_file.name} → {result['Decision']}"):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Input SAR image", use_container_width=True)

            with col2:
                decision = result["Decision"]
                confidence = result["Confidence (%)"]

                if decision == "SHIP":
                    st.success("Prediction: **SHIP** 🚢")
                elif decision == "SEA":
                    st.success("Prediction: **SEA** 🌊")
                else:
                    st.warning("Prediction: **UNCERTAIN**")

                st.metric("Confidence", f"{confidence:.2f}%")

                st.write(f"Sea probability: **{result['Sea Probability (%)']:.2f}%**")
                st.progress(result["Sea Probability (%)"] / 100)

                st.write(f"Ship probability: **{result['Ship Probability (%)']:.2f}%**")
                st.progress(result["Ship Probability (%)"] / 100)


def render_processing_tab() -> None:
    """Render basic speckle and blur processing interface."""
    st.header("✨ SAR Speckle / Blur Processing")

    st.write(
        "Upload a SAR image and apply simple image-processing filters. "
        "This tab is prepared as a prototype extension for SAR restoration tasks."
    )

    st.info(
        "Current filters are classical image-processing methods. "
        "DnCNN + U-Net based restoration can be added later when a trained checkpoint is available."
    )

    method = st.selectbox(
        "Select processing method",
        [
            "Median filter",
            "Sharpen",
            "Smooth",
            "DnCNN + U-Net restoration (coming soon)",
        ],
    )

    uploaded_file = st.file_uploader(
        "Upload SAR image for processing",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="processing_upload",
    )

    if uploaded_file is None:
        st.warning("Please upload a SAR image to start processing.")
        return

    image = Image.open(uploaded_file)

    if method == "DnCNN + U-Net restoration (coming soon)":
        st.warning(
            "DnCNN + U-Net restoration is planned as a future model-based module. "
            "A trained restoration checkpoint is required before this option can run."
        )
        st.image(image, caption="Input SAR image", use_container_width=True)
        return

    processed_image = apply_processing(image, method)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Original Image")
        st.image(image, caption="Original SAR image", use_container_width=True)

    with col2:
        st.subheader("Processed Image")
        st.image(processed_image, caption=f"Processed with {method}", use_container_width=True)

    st.download_button(
        label="⬇️ Download processed image",
        data=image_to_png_bytes(processed_image),
        file_name=f"processed_{uploaded_file.name.rsplit('.', 1)[0]}.png",
        mime="image/png",
    )


def render_about_tab() -> None:
    """Render project explanation and limitations."""
    st.header("ℹ️ About This Demo")

    st.write(
        "This web application demonstrates a prototype SAR image processing and "
        "classification pipeline. The current trained model classifies SAR patches "
        "as **ship** or **sea**."
    )

    st.subheader("Current Dataset")
    st.table(DATASET_INFO)

    st.subheader("Current Capabilities")
    st.write(
        "- Batch SAR ship-sea classification\n"
        "- Confidence score visualization\n"
        "- CSV export of prediction results\n"
        "- Basic speckle/blur processing filters\n"
        "- Downloadable processed images"
    )

    st.subheader("Limitations")
    st.write(
        "- The model is trained on a controlled prototype dataset.\n"
        "- Ship images and sea patches come from different SAR data sources.\n"
        "- Large Sentinel-1 tiles should be converted into 256×256 patches before classification.\n"
        "- Model-based despeckling/deblurring requires a separate trained restoration model."
    )


def main() -> None:
    """Run Streamlit application."""
    st.set_page_config(
        page_title="SAR Ship-Sea Classification",
        page_icon="📡",
        layout="wide",
    )

    render_sidebar()

    st.title("📡 SAR Image Processing Platform")
    st.caption("Ship-Sea Classification | Speckle / Blur Processing | Prototype Demo")

    if not MODEL_PATH.exists():
        st.error(
            "Trained model file was not found. Please run `python src/train.py` first."
        )
        st.stop()

    classification_tab, processing_tab, about_tab = st.tabs(
        [
            "🚢 Classification",
            "✨ Speckle / Blur Processing",
            "ℹ️ About",
        ]
    )

    with classification_tab:
        render_classification_tab()

    with processing_tab:
        render_processing_tab()

    with about_tab:
        render_about_tab()


if __name__ == "__main__":
    main()