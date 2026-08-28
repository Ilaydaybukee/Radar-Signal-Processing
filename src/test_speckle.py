import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from scipy.ndimage import laplace

from dataset_speckle import SARDespeckleDataset
from model_speckle import HybridDnCNNUnet

# ESKİ (Hata veren) Yollar:
# TEST_NOISY_DIR = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration\test_noisy"
# TEST_CLEAN_DIR = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration\test_clean"

# YENİ (Düzeltilmiş) Yollar:
TEST_NOISY_DIR = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration\noisy"
TEST_CLEAN_DIR = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration\clean"
MODEL_PATH = "hybrid_speckle_model.pth"
OUTPUT_DIR = "sample_comparisons" # Şekil Y için görseller buraya çıkacak

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MATEMATİKSEL METRİKLER ---
def calculate_snr(clean, restored):
    """Sinyal-Gürültü Oranını hesaplar (Mansourpour uyumlu)"""
    mse = np.mean((clean - restored) ** 2)
    if mse == 0: return float('inf')
    return 10 * np.log10(np.mean(clean ** 2) / mse)

def calculate_beta(restored, clean):
    """Kenar koruma endeksi (Beta) hesaplar (Değer 1'e ne kadar yakınsa o kadar iyi)"""
    lap_restored = laplace(restored)
    lap_clean = laplace(clean)
    num = np.sum(lap_restored * lap_clean)
    den = np.sqrt(np.sum(lap_restored**2) * np.sum(lap_clean**2))
    return num / den if den != 0 else 0

# --- SİSTEMİ AYAĞA KALDIR ---
dataset = SARDespeckleDataset(noisy_dir=TEST_NOISY_DIR, clean_dir=TEST_CLEAN_DIR)
# Batch size 1 olmalı ki her fotoğrafı teker teker inceleyip metrik çıkarabilelim
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

model = HybridDnCNNUnet(channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

band_metrics = []
classic_metrics = []

print("Test başlıyor, metrikler hesaplanıyor...\n")

with torch.no_grad():
    for i, (noisy, clean, filename) in enumerate(test_loader):
        filename = filename[0] # Tuple'dan stringi çıkar
        noisy, clean = noisy.to(device), clean.to(device)
        
        # Model tahmini (Restoration)
        restored = model(noisy)
        
        # Tensörleri numpy dizilerine (0-1 aralığında) çevir
        noisy_np = noisy.cpu().squeeze().numpy()
        clean_np = clean.cpu().squeeze().numpy()
        restored_np = restored.cpu().squeeze().numpy()
        
        # --- ÇİZELGE X İÇİN: PSNR & SSIM ---
        # Data range 1.0 olarak belirlenir çünkü resimler 0-1 aralığına normalize edildi
        noisy_psnr = compare_psnr(clean_np, noisy_np, data_range=1.0)
        restored_psnr = compare_psnr(clean_np, restored_np, data_range=1.0)
        noisy_ssim = compare_ssim(clean_np, noisy_np, data_range=1.0)
        restored_ssim = compare_ssim(clean_np, restored_np, data_range=1.0)
        
        # --- ÇİZELGE Y İÇİN: MSE, SNR, BETA ---
        mse = np.mean((clean_np - restored_np) ** 2)
        snr = calculate_snr(clean_np, restored_np)
        beta = calculate_beta(restored_np, clean_np)
        
        # Dosya adından bandı bul (Örn: "C-band_resim1.tif" -> "C-band")
        band_name = filename.split('_')[0] if '_' in filename else "Unknown"
        
        band_metrics.append({
            "band": band_name, "image_id": filename,
            "noisy_psnr": noisy_psnr, "restored_psnr": restored_psnr,
            "noisy_ssim": noisy_ssim, "restored_ssim": restored_ssim
        })
        
        classic_metrics.append({
            "image_id": filename, "mse": mse, "snr": snr, "beta": beta
        })
        
        # --- ŞEKİL Y İÇİN 3x3 GÖRSEL ÜRETİMİ ---
        # Her banttan sadece ilk birkaç örneği kaydetmek yeterlidir
        if i < 15:  
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(noisy_np, cmap='gray')
            plt.title('Gürültülü Giriş')
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(restored_np, cmap='gray')
            plt.title('DnCNN-U-Net Çıktısı')
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(clean_np, cmap='gray')
            plt.title('Referans')
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"comparison_{filename}.png"))
            plt.close()

# --- CSV ÇIKTILARINI KAYDET ---
df_band = pd.DataFrame(band_metrics)
df_band.to_csv("test_metrics_by_band.csv", index=False)

df_classic = pd.DataFrame(classic_metrics)
df_classic.to_csv("test_metrics_classic.csv", index=False)

print("Tüm görevler tamamlandı!")
print("- Çizelge X için 'test_metrics_by_band.csv' oluşturuldu.")
print("- Çizelge Y için 'test_metrics_classic.csv' oluşturuldu.")
print(f"- Şekil Y için görseller '{OUTPUT_DIR}' klasörüne kaydedildi.")