import torch
import torch.nn as nn

# 1. Aşama: Gürültüyü tahmin eden DnCNN Modeli
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(channels, 64, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(64, channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # Gürültü haritasını (residual) çıkarır
        return self.dncnn(x)

# U-Net için İkili Evrişim Bloğu
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

# 2. Aşama: Detayları rafine eden U-Net Modeli
class UNet(nn.Module):
    def __init__(self, channels=1):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(128, 64)
        
        self.out_conv = nn.Conv2d(64, channels, kernel_size=1)

    def forward(self, x):
        # Kodlama (Encoder)
        x1 = self.down1(x)
        x2 = self.pool1(x1)
        x3 = self.down2(x2)
        
        # Çözme ve Birleştirme (Decoder & Skip Connection)
        x_up = self.up1(x3)
        x_up = torch.cat([x_up, x1], dim=1)
        x_up = self.conv_up1(x_up)
        
        return self.out_conv(x_up)

# Ana Hibrit Mimari
class HybridDnCNNUnet(nn.Module):
    def __init__(self, channels=1):
        super(HybridDnCNNUnet, self).__init__()
        self.dncnn = DnCNN(channels)
        self.unet = UNet(channels)

    def forward(self, x):
        # Adım 1: Bozulma (gürültü) tahmini ve girişten çıkarma
        estimated_noise = self.dncnn(x)
        denoised_initial = x - estimated_noise
        
        # Adım 2: U-Net refinement (ince onarım)
        final_output = self.unet(denoised_initial)
        
        # Piksellerin 0-1 aralığında kalmasını garantiliyoruz
        return torch.clamp(final_output, 0.0, 1.0)