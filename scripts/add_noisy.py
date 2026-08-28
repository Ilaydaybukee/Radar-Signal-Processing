import os
import cv2
import numpy as np

CLEAN_KLASOR = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration\clean"
NOISY_KLASOR = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration\noisy"
os.makedirs(NOISY_KLASOR, exist_ok=True)

for dosya in os.listdir(CLEAN_KLASOR):
    if dosya.endswith((".tif", ".png", ".jpg")):
        img = cv2.imread(os.path.join(CLEAN_KLASOR, dosya), cv2.IMREAD_GRAYSCALE)
        
        # Speckle (Benek) Gürültüsü formülü: İmage + İmage * Noise
        noise = np.random.normal(0, 0.2, img.shape) # 0.2 gürültü şiddeti
        noisy_img = img + img * noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
        cv2.imwrite(os.path.join(NOISY_KLASOR, dosya), noisy_img)
        print(f"{dosya} gürültülendirildi ve kaydedildi.")