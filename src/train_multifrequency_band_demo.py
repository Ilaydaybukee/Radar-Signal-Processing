"""Train a small CNN to classify SAR patches by frequency band."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_ROOT = Path(r"C:\SAR_Datasets\multifrequency_band_demo")
DEFAULT_OUTPUT_DIR = Path("outputs/multifrequency_band_demo")
CLASSES = ["C_band", "S_band", "L_band"]
CLASS_TO_IDX = {class_name: idx for idx, class_name in enumerate(CLASSES)}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class BandPatchDataset(Dataset):
    def __init__(self, split_dir: Path, image_size: int = 256) -> None:
        self.split_dir = Path(split_dir)
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []
        for class_name in CLASSES:
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, CLASS_TO_IDX[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("L").resize((self.image_size, self.image_size))
            array = np.array(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)


class TinyBandCNN(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, kernel_size=3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(48 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    average_loss = total_loss / max(1, len(all_labels))
    accuracy = accuracy_score(all_labels, all_predictions) if all_labels else 0.0
    return average_loss, accuracy


def predict_all(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    labels_out: list[int] = []
    predictions_out: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu().tolist()
            labels_out.extend(labels.tolist())
            predictions_out.extend(predictions)
    return labels_out, predictions_out


def save_training_curves(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], marker="o", label="Train")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], marker="o", label="Train")
    plt.plot(epochs, history["val_acc"], marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_confusion_matrix(labels: list[int], predictions: list[int], output_path: Path) -> None:
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(CLASSES))))
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title("Band Classification Confusion Matrix")
    plt.xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    plt.yticks(range(len(CLASSES)), CLASSES)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            plt.text(x, y, str(matrix[y, x]), ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_sample_grid(dataset: BandPatchDataset, output_path: Path, samples_per_class: int = 4) -> None:
    font = ImageFont.load_default()
    cell_size = 140
    label_height = 20
    grid = Image.new("RGB", (samples_per_class * cell_size, len(CLASSES) * (cell_size + label_height)), "white")
    draw = ImageDraw.Draw(grid)
    by_class: dict[int, list[Path]] = {idx: [] for idx in range(len(CLASSES))}
    for path, label in dataset.samples:
        if len(by_class[label]) < samples_per_class:
            by_class[label].append(path)

    for label_idx, class_name in enumerate(CLASSES):
        y = label_idx * (cell_size + label_height)
        for col, path in enumerate(by_class[label_idx]):
            x = col * cell_size
            with Image.open(path) as image:
                image = image.convert("L").resize((cell_size, cell_size))
                grid.paste(image.convert("RGB"), (x, y))
            draw.text((x + 4, y + cell_size + 3), class_name, fill="black", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = BandPatchDataset(args.data_root / "train", image_size=args.image_size)
    val_dataset = BandPatchDataset(args.data_root / "val", image_size=args.image_size)
    test_dataset = BandPatchDataset(args.data_root / "test", image_size=args.image_size)

    if len(train_dataset) == 0:
        raise SystemExit(f"No training images found under {args.data_root / 'train'}")
    if len(val_dataset) == 0:
        raise SystemExit(f"No validation images found under {args.data_root / 'val'}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset if len(test_dataset) else val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyBandCNN(num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    checkpoint_path = args.output_dir / "multifrequency_band_demo_cnn.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "image_size": args.image_size,
                    "best_val_acc": best_val_acc,
                },
                checkpoint_path,
            )

    labels, predictions = predict_all(model, test_loader, device)
    metrics = {
        "classes": CLASSES,
        "device": str(device),
        "epochs": args.epochs,
        "best_val_accuracy": best_val_acc,
        "test_or_val_accuracy": accuracy_score(labels, predictions) if labels else 0.0,
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            zero_division=0,
            output_dict=True,
        ),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_training_curves(history, args.output_dir / "training_curves.png")
    save_confusion_matrix(labels, predictions, args.output_dir / "confusion_matrix.png")
    save_sample_grid(train_dataset, args.output_dir / "sample_grid.png")
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
