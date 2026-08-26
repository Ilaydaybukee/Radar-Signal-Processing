"""Dataset loader for Global SAR Ship-Sea Classification.

This file defines a PyTorch Dataset class that reads processed SAR images
from folders such as:

data/processed/train/ship
data/processed/train/sea

Labels:
- sea  -> 0
- ship -> 1
"""

from pathlib import Path
from typing import Callable

from PIL import Image
import torch
from torch.utils.data import Dataset


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

CLASS_TO_LABEL = {
    "sea": 0,
    "ship": 1,
}


class SARShipSeaDataset(Dataset):
    """PyTorch Dataset for ship/sea SAR image classification."""

    def __init__(
        self,
        split_dir: str | Path,
        image_size: int = 256,
        transform: Callable | None = None,
    ) -> None:
        self.split_dir = Path(split_dir)
        self.image_size = image_size
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        self._collect_samples()

    def _collect_samples(self) -> None:
        """Collect image paths and labels from sea/ship folders."""
        for class_name, label in CLASS_TO_LABEL.items():
            class_dir = self.split_dir / class_name

            if not class_dir.exists():
                continue

            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    self.samples.append((image_path, label))

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Load one image and return image tensor with label."""
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("L")
            image = image.resize((self.image_size, self.image_size))

            if self.transform is not None:
                image_tensor = self.transform(image)
            else:
                image_tensor = self._pil_to_tensor(image)

        return image_tensor, label

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        """Convert grayscale PIL image to normalized tensor [1, H, W]."""
        image_bytes = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        image_tensor = image_bytes.float().view(image.height, image.width) / 255.0

        return image_tensor.unsqueeze(0)


def main() -> None:
    """Small manual check."""
    dataset = SARShipSeaDataset("data/processed/train")
    print(f"Train samples found: {len(dataset)}")

    if len(dataset) > 0:
        image, label = dataset[0]
        print(f"First image tensor shape: {image.shape}")
        print(f"First label: {label}")


if __name__ == "__main__":
    main()