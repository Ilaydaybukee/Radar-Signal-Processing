"""Advanced Streamlit app for SAR Ship-Sea Classification and SAR image processing.

Run from the project root:

streamlit run app/streamlit_app.py
"""

from datetime import datetime
from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

CLASSIFICATION_MODEL_PATH = PROJECT_ROOT / "models" / "simple_sar_classifier.pth"
RESTORATION_MODEL_PATH = PROJECT_ROOT / "models" / "dncnn_unet_restoration.pth"

sys.path.append(str(SRC_DIR))

from model import SimpleSARClassifier  # noqa: E402


CLASS_NAMES = {
    0: "sea",
    1: "ship",
}


def initialize_history() -> None:
    """Initialize Streamlit session history."""
    if "history" not in st.session_state:
        st.session_state["history"] = []


def add_history_record(record: dict) -> None:
    """Add one record to session history."""
    st.session_state["history"].append(record)


def get_device() -> torch.device:
    """Use CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_classification_model() -> tuple[SimpleSARClassifier, torch.device]:
    """Load trained ship-sea classification model."""
    device = get_device()

    model = SimpleSARClassifier(num_classes=2)
    model.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    return model, device


def pil_to_model_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized model tensor [1, 1, 256, 256]."""
    image = image.convert("L")
    image = image.resize((256, 256))

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0)

    return image_tensor


def classify_image(image: Image.Image) -> tuple[str, float, list[float]]:
    """Classify image as sea or ship."""
    model, device = load_classification_model()

    input_tensor = pil_to_model_tensor(image).to(device)

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


def apply_speckle_median_filter(image: Image.Image, filter_size: int = 3) -> Image.Image:
    """Apply a simple median filter as a classical speckle reduction demo."""
    image = image.convert("L")
    return image.filter(ImageFilter.MedianFilter(size=filter_size))


def apply_blur_sharpen_filter(image: Image.Image) -> Image.Image:
    """Apply a simple sharpening filter as a classical blur restoration demo."""
    image = image.convert("L")
    return image.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)


def image_to_download_bytes(image: Image.Image, image_format: str = "PNG") -> bytes:
    """Convert PIL image to downloadable bytes."""
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def classification_tab() -> None:
    """Classification interface."""
    st.header("🚢 Ship-Sea Classification")

    st.write(
        "Upload one or more SAR image patches. "
        "The trained CNN model will classify each image as **sea** or **ship**."
    )

    if not CLASSIFICATION_MODEL_PATH.exists():
        st.error("Classification model was not found. Please train the model first.")
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload SAR images for classification",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="classification_uploader",
    )

    if not uploaded_files:
        st.warning("Please upload one or more SAR images.")
        return

    results = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        predicted_class, confidence, probabilities = classify_image(image)

        result = {
            "Image": uploaded_file.name,
            "Prediction": predicted_class.upper(),
            "Confidence (%)": round(confidence * 100, 2),
            "Sea Probability (%)": round(probabilities[0] * 100, 2),
            "Ship Probability (%)": round(probabilities[1] * 100, 2),
        }
        results.append(result)

        add_history_record(
            {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Task": "Classification",
                "Image": uploaded_file.name,
                "Result": predicted_class.upper(),
                "Confidence (%)": round(confidence * 100, 2),
            }
        )

    st.subheader("Batch Prediction Results")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.subheader("Image Preview and Individual Results")

    for uploaded_file, result in zip(uploaded_files, results):
        image = Image.open(uploaded_file)

        with st.expander(f"{uploaded_file.name} → {result['Prediction']}"):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Input SAR image", use_container_width=True)

            with col2:
                prediction = result["Prediction"]
                confidence = result["Confidence (%)"]

                if prediction == "SHIP":
                    st.success("Prediction: **SHIP** 🚢")
                else:
                    st.success("Prediction: **SEA** 🌊")

                st.metric("Confidence", f"{confidence:.2f}%")

                st.write(f"Sea probability: **{result['Sea Probability (%)']:.2f}%**")
                st.progress(result["Sea Probability (%)"] / 100)

                st.write(f"Ship probability: **{result['Ship Probability (%)']:.2f}%**")
                st.progress(result["Ship Probability (%)"] / 100)


def processing_tab() -> None:
    """SAR image processing interface."""
    st.header("🛠️ SAR Speckle / Blur Processing")

    st.write(
        "This section provides prototype SAR image processing tools. "
        "Classical filters are available now. DnCNN + U-Net restoration will be connected "
        "after the restoration model is trained."
    )

    method = st.selectbox(
        "Select processing method",
        [
            "Speckle Reduction - Median Filter Demo",
            "Blur Restoration - Sharpen Filter Demo",
            "DnCNN + U-Net Restoration Model",
        ],
    )

    uploaded_files = st.file_uploader(
        "Upload SAR images for processing",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="processing_uploader",
    )

    if not uploaded_files:
        st.warning("Please upload one or more SAR images.")
        return

    if method == "DnCNN + U-Net Restoration Model" and not RESTORATION_MODEL_PATH.exists():
        st.error(
            "DnCNN + U-Net restoration checkpoint was not found yet. "
            "We need to train and save `models/dncnn_unet_restoration.pth` first."
        )
        st.info(
            "For now, use the classical speckle or blur demo filters. "
            "The DnCNN + U-Net architecture will be added as the next development step."
        )
        return

    st.subheader("Processing Results")

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("L")

        if method == "Speckle Reduction - Median Filter Demo":
            processed_image = apply_speckle_median_filter(image, filter_size=3)
            result_name = "speckle_median"
        elif method == "Blur Restoration - Sharpen Filter Demo":
            processed_image = apply_blur_sharpen_filter(image)
            result_name = "blur_sharpen"
        else:
            st.warning("DnCNN + U-Net inference is not connected yet.")
            continue

        add_history_record(
            {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Task": method,
                "Image": uploaded_file.name,
                "Result": "Processed",
                "Confidence (%)": "-",
            }
        )

        with st.expander(f"{uploaded_file.name} → {method}"):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.image(image, caption="Original SAR image", use_container_width=True)

            with col2:
                st.image(processed_image, caption="Processed SAR image", use_container_width=True)

                download_name = f"{Path(uploaded_file.name).stem}_{result_name}.png"

                st.download_button(
                    label="Download processed image",
                    data=image_to_download_bytes(processed_image),
                    file_name=download_name,
                    mime="image/png",
                )


def history_tab() -> None:
    """Show session history."""
    st.header("📜 Session History")

    history = st.session_state.get("history", [])

    if not history:
        st.info("No history yet. Run classification or processing first.")
        return

    history_df = pd.DataFrame(history)
    st.dataframe(history_df, use_container_width=True)

    csv_bytes = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download history as CSV",
        data=csv_bytes,
        file_name="sar_processing_history.csv",
        mime="text/csv",
    )

    if st.button("Clear session history"):
        st.session_state["history"] = []
        st.rerun()


def about_tab() -> None:
    """About section."""
    st.header("ℹ️ About This Demo")

    st.write(
        """
        This application is a prototype interface for SAR image processing.

        Current modules:

        - **Ship-Sea Classification:** CNN-based binary classification.
        - **Speckle Reduction Demo:** Classical median filtering.
        - **Blur Restoration Demo:** Classical sharpening filter.
        - **History:** Session-level record of uploaded images and results.

        Planned modules:

        - DnCNN-based speckle noise estimation.
        - U-Net-based restoration/refinement.
        - DnCNN + U-Net hybrid SAR restoration model.
        - Region/source-based dataset splitting for stronger generalization testing.
        """
    )

    st.warning(
        "This is a research prototype. Results should not be interpreted as final real-world performance."
    )


def main() -> None:
    """Run Streamlit application."""
    st.set_page_config(
        page_title="SAR Image Processing Platform",
        page_icon="📡",
        layout="wide",
    )

    initialize_history()

    st.title("📡 SAR Image Processing Platform")
    st.write(
        "Classification, SAR speckle/blur processing demos, and session history "
        "in one Streamlit interface."
    )

    st.info(
        "Current classifier is trained on a 2000-image balanced SAR dataset. "
        "Restoration modules are prototype tools and will be improved with DnCNN + U-Net."
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🚢 Classification",
            "🛠️ Speckle / Blur Processing",
            "📜 History",
            "ℹ️ About",
        ]
    )

    with tab1:
        classification_tab()

    with tab2:
        processing_tab()

    with tab3:
        history_tab()

    with tab4:
        about_tab()


if __name__ == "__main__":
    main()