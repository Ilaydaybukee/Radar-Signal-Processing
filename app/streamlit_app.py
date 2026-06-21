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
        layout="centered",
    )

    st.title("📡 SAR Ship-Sea Classification")
    st.write(
        "Upload a SAR image patch and the trained CNN model will classify it "
        "as **sea** or **ship**."
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

    uploaded_file = st.file_uploader(
        "Upload SAR image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )

    if uploaded_file is None:
        st.warning("Please upload a SAR image to start prediction.")
        return

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, caption="Input SAR image", use_container_width=True)

    predicted_class, confidence, probabilities = predict(image)

    st.subheader("Prediction Result")

    if predicted_class == "ship":
        st.success(f"Prediction: **SHIP** 🚢")
    else:
        st.success(f"Prediction: **SEA** 🌊")

    st.metric("Confidence", f"{confidence * 100:.2f}%")

    st.subheader("Class Probabilities")

    st.write(f"Sea probability: **{probabilities[0] * 100:.2f}%**")
    st.progress(probabilities[0])

    st.write(f"Ship probability: **{probabilities[1] * 100:.2f}%**")
    st.progress(probabilities[1])


if __name__ == "__main__":
    main()