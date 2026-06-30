"""DnCNN + U-Net hybrid restoration model for SAR image processing.

This module defines a prototype restoration architecture for SAR images.

Pipeline:
1. DnCNN estimates noise/degradation from the input SAR image.
2. Initial restored image is obtained by subtracting estimated noise.
3. U-Net refines the restored image and reconstructs spatial details.

Input:
- 1-channel grayscale SAR image
- shape: [batch, 1, 256, 256]

Output:
- 1-channel restored SAR image
- shape: [batch, 1, 256, 256]
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    """DnCNN block for residual noise/degradation estimation."""

    def __init__(
        self,
        in_channels: int = 1,
        num_features: int = 64,
        depth: int = 17,
    ) -> None:
        super().__init__()

        if depth < 3:
            raise ValueError("DnCNN depth should be at least 3.")

        layers: list[nn.Module] = []

        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_features,
                kernel_size=3,
                padding=1,
                bias=True,
            )
        )
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=num_features,
                    out_channels=num_features,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        layers.append(
            nn.Conv2d(
                in_channels=num_features,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            )
        )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return initial restored image and estimated noise."""
        estimated_noise = self.network(x)
        initial_restored = x - estimated_noise

        return initial_restored, estimated_noise


class DoubleConv(nn.Module):
    """Two consecutive convolution layers used in U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run double convolution block."""
        return self.block(x)


class UNet(nn.Module):
    """Lightweight U-Net for SAR restoration refinement."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 32,
    ) -> None:
        super().__init__()

        self.encoder1 = DoubleConv(in_channels, base_features)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.encoder2 = DoubleConv(base_features, base_features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.encoder3 = DoubleConv(base_features * 2, base_features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = DoubleConv(base_features * 4, base_features * 8)

        self.up3 = nn.ConvTranspose2d(
            base_features * 8,
            base_features * 4,
            kernel_size=2,
            stride=2,
        )
        self.decoder3 = DoubleConv(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose2d(
            base_features * 4,
            base_features * 2,
            kernel_size=2,
            stride=2,
        )
        self.decoder2 = DoubleConv(base_features * 4, base_features * 2)

        self.up1 = nn.ConvTranspose2d(
            base_features * 2,
            base_features,
            kernel_size=2,
            stride=2,
        )
        self.decoder1 = DoubleConv(base_features * 2, base_features)

        self.output_layer = nn.Conv2d(
            in_channels=base_features,
            out_channels=out_channels,
            kernel_size=1,
        )

        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run U-Net forward pass."""
        enc1 = self.encoder1(x)

        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))

        bottleneck = self.bottleneck(self.pool3(enc3))

        dec3 = self.up3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.up2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.up1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        output = self.output_layer(dec1)
        output = self.output_activation(output)

        return output


class DnCNNUNetRestoration(nn.Module):
    """Hybrid DnCNN + U-Net restoration model."""

    def __init__(self) -> None:
        super().__init__()

        self.dncnn = DnCNN(in_channels=1)
        self.unet = UNet(in_channels=1, out_channels=1)

    def forward(
        self,
        x: torch.Tensor,
        return_intermediate: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Run DnCNN followed by U-Net refinement."""
        initial_restored, estimated_noise = self.dncnn(x)
        initial_restored = torch.clamp(initial_restored, 0.0, 1.0)

        refined_restored = self.unet(initial_restored)

        if return_intermediate:
            return {
                "input": x,
                "estimated_noise": estimated_noise,
                "initial_restored": initial_restored,
                "refined_restored": refined_restored,
            }

        return refined_restored


def main() -> None:
    """Quick architecture test."""
    model = DnCNNUNetRestoration()

    dummy_input = torch.randn(1, 1, 256, 256)
    output = model(dummy_input)

    intermediate_outputs = model(dummy_input, return_intermediate=True)

    print(model.__class__.__name__)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Intermediate outputs:")

    for name, tensor in intermediate_outputs.items():
        print(f"  {name}: {tensor.shape}")


if __name__ == "__main__":
    main()