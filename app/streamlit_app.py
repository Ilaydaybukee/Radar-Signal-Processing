"""Streamlit demo app for Global SAR Ship-Sea Classification.

Run from the project root:

streamlit run app/streamlit_app.py
"""

from pathlib import Path
import sys

import streamlit as st
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MODEL_PATH = PROJECT_ROOT / "models" / "simple_sar_classifier.pth"

sys.path.append(str(SRC_DIR))

from model import SimpleSARClassifier  # noqa: E402


CLASS_NAMES = {
    0: "sea",
    1: "ship",
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


def main() -> None:
    """Run Streamlit application."""
    st.set_page_config(
        page_title="SAR Ship-Sea Classification",
        page_icon="📡",
        layout="wide",
    )

    st.title("📡 SAR Ship-Sea Classification")
    st.write(
        "Upload one or more SAR image patches and the trained CNN model will classify "
        "each image as **sea** or **ship**."
    )

    st.info(
        "This is a prototype demo trained on a small balanced SAR dataset. "
        "Results should be interpreted as prototype-level outputs."
    )

    if not MODEL_PATH.exists():
        st.error(
            "Trained model file was not found. Please run `python src/train.py` first."
        )
        st.stop()

    uploaded_files = st.file_uploader(
        "Upload SAR images",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.warning("Please upload one or more SAR images to start prediction.")
        return

    st.subheader("Batch Prediction Results")

    results = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)

        predicted_class, confidence, probabilities = predict(image)

        results.append(
            {
                "Image": uploaded_file.name,
                "Prediction": predicted_class.upper(),
                "Confidence (%)": round(confidence * 100, 2),
                "Sea Probability (%)": round(probabilities[0] * 100, 2),
                "Ship Probability (%)": round(probabilities[1] * 100, 2),
            }
        )

    st.dataframe(results, use_container_width=True)

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


if __name__ == "__main__":
    main()