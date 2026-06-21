"""Evaluate the trained CNN for Global SAR Ship-Sea Classification.

This script loads:

models/simple_sar_classifier.pth

and evaluates it on:

data/processed/test/ship
data/processed/test/sea

Run from the project root:

python src/evaluate.py
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SARShipSeaDataset, CLASS_TO_LABEL
from model import SimpleSARClassifier


TEST_DIR = Path("data/processed/test")
MODEL_PATH = Path("models/simple_sar_classifier.pth")

BATCH_SIZE = 8


def get_device() -> torch.device:
    """Use CUDA GPU if available, otherwise use CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main() -> None:
    """Evaluate the trained model on the test dataset."""
    if not MODEL_PATH.exists():
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Train the model first:")
        print("python src/train.py")
        return

    if not TEST_DIR.exists():
        print(f"ERROR: Test folder not found: {TEST_DIR}")
        print("Prepare the dataset first:")
        print("python scripts/prepare_dataset.py")
        return

    test_dataset = SARShipSeaDataset(TEST_DIR)

    if len(test_dataset) == 0:
        print("ERROR: No test images found.")
        print(f"Expected images under: {TEST_DIR / 'ship'} and {TEST_DIR / 'sea'}")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = get_device()
    model = SimpleSARClassifier(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    label_to_class = {label: class_name for class_name, label in CLASS_TO_LABEL.items()}

    total_correct = 0
    total_samples = 0

    per_class_correct = {class_name: 0 for class_name in CLASS_TO_LABEL}
    per_class_total = {class_name: 0 for class_name in CLASS_TO_LABEL}

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            for prediction, label in zip(predictions, labels):
                class_name = label_to_class[label.item()]
                per_class_total[class_name] += 1

                if prediction.item() == label.item():
                    per_class_correct[class_name] += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    print("SAR Ship-Sea Test Evaluation")
    print("============================")
    print(f"Using device: {device}")
    print(f"Test samples: {total_samples}")
    print(f"Test accuracy: {accuracy:.4f}")
    print()

    print("Per-class results:")
    for class_name in sorted(per_class_total):
        correct = per_class_correct[class_name]
        total = per_class_total[class_name]
        class_accuracy = correct / total if total > 0 else 0.0

        print(f"  {class_name}: {correct}/{total} correct | accuracy: {class_accuracy:.4f}")


if __name__ == "__main__":
    main()
