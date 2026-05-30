import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from pytorch_msssim import ssim
from pathlib import Path
from PIL import Image
from PIL import ImageFilter
import numpy as np
import matplotlib.pyplot as plt

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
# 1.5) SAR DATASET SINIFI
# =========================================================
# Görevi:
# Temiz SAR görüntülerini klasörden okur.
# Görüntüyü grayscale yapar.
# 256x256 boyutuna getirir.
# PyTorch tensor formatına çevirir.
# =========================================================

class SARDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, image_size=256):
        self.image_paths = image_paths
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # Görüntüyü grayscale olarak aç
        image = Image.open(image_path).convert("L")

        # Boyutu 256x256 yap
        image = image.resize((self.image_size, self.image_size))

        # 0-255 aralığından 0-1 aralığına normalize et
        image_np = np.array(image, dtype=np.float32) / 255.0

        # Tensor formatı: [kanal, yükseklik, genişlik]
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)

        return image_tensor, str(image_path)

# =========================================================
# 1.6) BLUR BOZULMASI EKLEME
# =========================================================
# Görevi:
# Temiz SAR görüntüsüne Gaussian blur uygular.
# Böylece eğitim için bozuk giriş görüntüsü oluşturulur.
# =========================================================

def add_blur(image_tensor, blur_radius=2.0):
    """
    image_tensor: [1, H, W] formatında 0-1 aralığında tensor
    blur_radius: Blur şiddeti
    """

    # Tensor -> NumPy -> PIL
    image_np = image_tensor.squeeze(0).numpy()
    image_uint8 = (image_np * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_uint8)

    # Gaussian blur uygula
    blurred_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # PIL -> NumPy -> Tensor
    blurred_np = np.array(blurred_pil, dtype=np.float32) / 255.0
    blurred_tensor = torch.from_numpy(blurred_np).unsqueeze(0)

    return blurred_tensor

# =========================================================
# 1.7) SPECKLE NOISE EKLEME
# =========================================================
# Görevi:
# Temiz SAR görüntüsüne multiplicative speckle noise ekler.
# SAR görüntülerinde speckle çarpımsal gürültü olarak modellenir.
# =========================================================

def add_speckle_noise(image_tensor, noise_level=0.2):
    """
    image_tensor: [1, H, W] formatında 0-1 aralığında tensor
    noise_level: Speckle gürültü şiddeti
    """

    # Aynı boyutta rastgele gürültü üret
    noise = torch.randn_like(image_tensor) * noise_level

    # Multiplicative speckle modeli:
    # bozuk_görüntü = temiz_görüntü + temiz_görüntü * gürültü
    noisy_tensor = image_tensor + image_tensor * noise

    # Değerleri 0-1 aralığında tut
    noisy_tensor = torch.clamp(noisy_tensor, 0.0, 1.0)

    return noisy_tensor

# =========================================================
# 1.8) BOZULMA ÖNİZLEME GÖRSELİ KAYDETME
# =========================================================
# Görevi:
# Temiz, blur uygulanmış ve speckle eklenmiş SAR görüntülerini
# yan yana kaydeder.
# =========================================================

def save_corruption_preview(clean_tensor, blurred_tensor, speckle_tensor, save_path="sample_corruption_preview.png"):
    clean_np = clean_tensor.squeeze(0).numpy()
    blurred_np = blurred_tensor.squeeze(0).numpy()
    speckle_np = speckle_tensor.squeeze(0).numpy()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(clean_np, cmap="gray")
    plt.title("Clean SAR")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(blurred_np, cmap="gray")
    plt.title("Blurred SAR")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(speckle_np, cmap="gray")
    plt.title("Speckle SAR")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print("Bozulma önizleme görseli kaydedildi:", save_path)

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
    
    clean_dataset = SARDataset(clean_image_paths, image_size=256)

    print("Dataset örnek sayısı:", len(clean_dataset))

    if len(clean_dataset) > 0:
        sample_image, sample_path = clean_dataset[0]
        print("İlk örnek tensor boyutu:", sample_image.shape)
        print("İlk örnek dosya yolu:", sample_path)
        blurred_sample = add_blur(sample_image, blur_radius=2.0)
        print("Blur uygulanmış örnek tensor boyutu:", blurred_sample.shape)

        speckle_sample = add_speckle_noise(sample_image, noise_level=0.2)
        print("Speckle eklenmiş örnek tensor boyutu:", speckle_sample.shape)

            save_corruption_preview(
            sample_image,
            blurred_sample,
            speckle_sample,
            save_path="sample_corruption_preview.png"
        )

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
