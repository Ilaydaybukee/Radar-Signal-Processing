"""Simple CNN model for Global SAR Ship-Sea Classification.

Input:
- 1-channel grayscale SAR image
- default size: 256x256

Output:
- 2 classes:
  0 -> sea
  1 -> ship
"""

import torch
import torch.nn as nn


class SimpleSARClassifier(nn.Module):
    """A lightweight CNN for binary SAR image classification."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Input: [batch, 1, 256, 256]
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> [batch, 16, 128, 128]

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> [batch, 32, 64, 64]

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # -> [batch, 64, 32, 32]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = self.classifier(x)
        return x


def main() -> None:
    """Quick model shape check."""
    model = SimpleSARClassifier()

    dummy_input = torch.randn(1, 1, 256, 256)
    output = model(dummy_input)

    print(model)
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
