import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model_speckle import HybridDnCNNUnet

# --- YAPILANDIRMA VE YOLLAR ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "hybrid_speckle_model.pth"
noisy_dir = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration\noisy"
clean_dir = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration\clean"

# SİLİNECEK ESKİ KISIM:
# samples = {
#     "C-band": "C-band_0000.jpg",
#     "S-band": "S-band_0000.jpg",
#     "L-band": "L-band_0000.jpg"
# }

# YENİ VE DOĞRU KISIM:
samples = {
    "C-band": "C-band_0000.jpg",
    "S-band": "S-band_0000.tif",
    "L-band": "L-band_0000.tif"
}

# Eğitilmiş modeli ayağa kaldır
model = HybridDnCNNUnet(channels=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 3x3 Makale Şablonunu Oluştur
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
col_labels = ["Gürültülü Giriş", "DnCNN-U-Net Çıktısı", "Referans"]

for idx, (band, filename) in enumerate(samples.items()):
    noisy_path = os.path.join(noisy_dir, filename)
    clean_path = os.path.join(clean_dir, filename)
    
    # Görüntüleri Türkçe karakter sorunu olmadan imdecode ile oku
    noisy_img = cv2.imdecode(np.fromfile(noisy_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    clean_img = cv2.imdecode(np.fromfile(clean_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    
    # Gürültülü resmi tensöre çevirip yapay zekadan geçir
    input_tensor = torch.from_numpy(noisy_img).float().unsqueeze(0).unsqueeze(0) / 255.0
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    output_img = (output_tensor.squeeze().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    
    # 1. Sütun: Gürültülü Giriş
    axes[idx, 0].imshow(noisy_img, cmap='gray')
    axes[idx, 0].axis('off')
    if idx == 0: axes[idx, 0].set_title(col_labels[0], fontsize=16, fontweight='bold', pad=20)
    
    # 2. Sütun: Yapay Zeka Çıktısı
    axes[idx, 1].imshow(output_img, cmap='gray')
    axes[idx, 1].axis('off')
    if idx == 0: axes[idx, 1].set_title(col_labels[1], fontsize=16, fontweight='bold', pad=20)
    
    # 3. Sütun: Referans (Temiz)
    axes[idx, 2].imshow(clean_img, cmap='gray')
    axes[idx, 2].axis('off')
    if idx == 0: axes[idx, 2].set_title(col_labels[2], fontsize=16, fontweight='bold', pad=20)
    
    # Satır Etiketleri (Sol tarafa C, S, L band yazıları)
    axes[idx, 0].text(-0.15, 0.5, band, va='center', ha='right', 
                      fontsize=16, fontweight='bold', transform=axes[idx, 0].transAxes)

plt.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.1) 
plt.savefig("sekil_y_makale_grid.png", dpi=300, bbox_inches='tight')
print("Rehbere tam uygun Şekil Y başarıyla 'sekil_y_makale_grid.png' olarak oluşturuldu!")