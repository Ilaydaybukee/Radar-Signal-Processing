"""Train a simple CNN for Global SAR Ship-Sea Classification.

This script trains a small CNN using the processed dataset:

data/processed/train/ship
data/processed/train/sea
data/processed/val/ship
data/processed/val/sea

Run from the project root:

python src/train.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SARShipSeaDataset
from model import SimpleSARClassifier


TRAIN_DIR = Path("data/processed/train")
VAL_DIR = Path("data/processed/val")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "simple_sar_classifier.pth"

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001


def get_device() -> torch.device:
    """Use CUDA GPU if available, otherwise use CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def calculate_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    """Calculate classification accuracy for one batch."""
    predictions = outputs.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)

    return correct / total if total > 0 else 0.0


def run_one_training_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Train the model for one epoch."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    average_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return average_loss, accuracy


def run_validation(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on the validation split."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    return average_loss, accuracy


def main() -> None:
    """Main training function."""
    if not TRAIN_DIR.exists():
        print(f"ERROR: Training folder not found: {TRAIN_DIR}")
        print("Run this first:")
        print("python scripts/prepare_dataset.py")
        return

    train_dataset = SARShipSeaDataset(TRAIN_DIR)
    val_dataset = SARShipSeaDataset(VAL_DIR)

    if len(train_dataset) == 0:
        print("ERROR: No training images found.")
        print(f"Expected images under: {TRAIN_DIR / 'ship'} and {TRAIN_DIR / 'sea'}")
        return

    if len(val_dataset) == 0:
        print("WARNING: No validation images found. Training will continue without validation.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = get_device()
    print(f"Using device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print()

    model = SimpleSARClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_one_training_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        if len(val_dataset) > 0:
            val_loss, val_accuracy = run_validation(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
            )

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )
        else:
            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f}"
            )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    print()
    print(f"Training complete. Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
