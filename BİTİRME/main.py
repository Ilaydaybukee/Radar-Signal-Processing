import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from pytorch_msssim import ssim
from pathlib import Path
from PIL import Image

# =========================================================
# 1) DnCNN BLOĞU
# =========================================================
# Görevi:
# Bozuk SAR görüntüsündeki gürültüyü / bozulmayı tahmin eder.
# Sonra:
# temizlenmiş_görüntü = bozuk_görüntü - tahmin_edilen_gürültü
# mantığıyla ilk temizlemeyi yapar.
# =========================================================

class DnCNN(nn.Module):
    def __init__(self, in_channels=1, depth=17, num_features=64):
        super(DnCNN, self).__init__()

        layers = []

        # İlk katman: Conv + ReLU
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_features,
                kernel_size=3,
                padding=1,
                bias=True
            )
        )
        layers.append(nn.ReLU(inplace=True))

        # Orta katmanlar: Conv + BatchNorm + ReLU
        for _ in range(depth - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=num_features,
                    out_channels=num_features,
                    kernel_size=3,
                    padding=1,
                    bias=False
                )
            )
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # Son katman: tahmin edilen gürültü görüntüsü
        layers.append(
            nn.Conv2d(
                in_channels=num_features,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
                bias=True
            )
        )

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        predicted_noise = self.dncnn(x)
        denoised = x - predicted_noise

        return denoised, predicted_noise

# =========================================================
# 0) VERİ KLASÖRÜ KONTROLÜ
# =========================================================
# Görevi:
# data/clean/europe klasörü altındaki SAR görüntülerini bulur.
# Şimdilik eğitim yapmaz, sadece veri yolu doğru mu diye kontrol eder.
# =========================================================

def check_clean_dataset(data_root="data/clean/europe"):
    data_root = Path(data_root)

    if not data_root.exists():
        print("UYARI: Veri klasörü bulunamadı:", data_root)
        return []

    image_extensions = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]

    image_paths = []
    for ext in image_extensions:
        image_paths.extend(data_root.rglob(f"*{ext}"))

    image_paths = sorted(image_paths)

    print("Veri klasörü:", data_root)
    print("Bulunan görüntü sayısı:", len(image_paths))

    print("\nİlk 10 görüntü yolu:")
    for path in image_paths[:10]:
        print(path)

    return image_paths

# =========================================================
# 2) U-NET BLOĞU
# =========================================================
# Hazır açık kaynak U-Net kullanıyoruz.
# segmentation_models_pytorch kütüphanesinden geliyor.
# =========================================================

def build_unet():
    unet = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=1,
        classes=1,
        activation=None
    )

    return unet


# =========================================================
# 3) HİBRİT DnCNN + U-NET MODELİ
# =========================================================
# Akış:
#
# Bozuk SAR Görüntüsü
#       ↓
# DnCNN
#       ↓
# Ön Temizlenmiş Görüntü
#       ↓
# U-Net
#       ↓
# Düzeltme Haritası
#       ↓
# Final Restore Görüntü
# =========================================================

class HybridDnCNNUNetResidual(nn.Module):
    def __init__(self):
        super(HybridDnCNNUNetResidual, self).__init__()

        self.dncnn = DnCNN(
            in_channels=1,
            depth=17,
            num_features=64
        )

        self.unet = build_unet()

    def forward(self, x):
        # 1. DnCNN ile ilk gürültü temizleme
        denoised, predicted_noise = self.dncnn(x)

        # 2. U-Net kalan düzeltmeyi öğrenir
        correction = self.unet(denoised)

        # 3. Ön temiz görüntü + düzeltme haritası
        restored = denoised + correction

        # 4. Değerleri 0-1 aralığına sıkıştır
        restored = torch.sigmoid(restored)

        return restored, denoised, predicted_noise, correction


# =========================================================
# 4) LOSS FONKSİYONU
# =========================================================
# Eğitim aşamasında kullanılacak.
# Şu an modeli test ederken kullanmayacağız.
# Ama ana omurgada hazır dursun.
# =========================================================

class SARHybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super(SARHybridLoss, self).__init__()

        self.alpha = alpha  # L1 ağırlığı
        self.beta = beta    # MSE ağırlığı
        self.gamma = gamma  # SSIM ağırlığı

        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        mse_loss = self.mse(pred, target)

        ssim_value = ssim(
            pred,
            target,
            data_range=1.0,
            size_average=True
        )

        ssim_loss = 1 - ssim_value

        total_loss = (
            self.alpha * l1_loss +
            self.beta * mse_loss +
            self.gamma * ssim_loss
        )

        loss_dict = {
            "l1": l1_loss.item(),
            "mse": mse_loss.item(),
            "ssim_loss": ssim_loss.item(),
            "ssim": ssim_value.item()
        }

        return total_loss, loss_dict


# =========================================================
# 5) MODEL TEST KISMI
# =========================================================
# Bu bölüm sadece kod çalışıyor mu diye bakmak için.
# Gerçek SAR görüntüsü kullanmıyoruz.
# Rastgele sahte görüntü veriyoruz.
# =========================================================

if __name__ == "__main__":

    print("SAR temiz veri klasörü kontrol ediliyor...")

    clean_image_paths = check_clean_dataset("data/clean/europe")

    print("--------------------------------------------------")
    print("SAR Restoration Hybrid DnCNN + U-Net modeli test ediliyor...")

    # Bilgisayarda NVIDIA ekran kartı ve CUDA varsa GPU kullanır.
    # Yoksa CPU kullanır.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Kullanılan cihaz:", device)

    # Modeli oluştur
    model = HybridDnCNNUNetResidual().to(device)

    print("Model başarıyla oluşturuldu.")

    # Sahte SAR görüntüsü oluştur
    # 2   = aynı anda 2 görüntü
    # 1   = tek kanal, yani grayscale SAR
    # 256 = yükseklik
    # 256 = genişlik
    x = torch.randn(2, 1, 256, 256).to(device)

    print("Sahte giriş görüntüsü oluşturuldu.")

    # Modeli çalıştır
    restored, denoised, predicted_noise, correction = model(x)

    # Boyutları yazdır
    print("--------------------------------------------------")
    print("Girdi boyutu:              ", x.shape)
    print("DnCNN sonrası görüntü:     ", denoised.shape)
    print("Tahmin edilen gürültü:     ", predicted_noise.shape)
    print("U-Net düzeltme haritası:   ", correction.shape)
    print("Final restore görüntü:     ", restored.shape)
    print("--------------------------------------------------")

    print("Test başarılı.")
    print("Hibrit DnCNN + U-Net modeli sorunsuz çalışıyor.")
