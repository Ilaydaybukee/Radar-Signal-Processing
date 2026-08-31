import cv2
import numpy as np
import os

def add_gamma_speckle(image, looks=4):
    row, col = image.shape
    img_float = image.astype(np.float32) / 255.0
    noise = np.random.gamma(shape=looks, scale=1.0/looks, size=(row, col))
    noisy_img = img_float * noise
    noisy_img = np.clip(noisy_img * 255.0, 0, 255).astype(np.uint8)
    return noisy_img

clean_dir = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration\clean"
out_dir = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration\sentetik_noisy"
os.makedirs(out_dir, exist_ok=True)

valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
clean_files = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.lower().endswith(valid_extensions)]

success_count = 0
for path in clean_files:
    # Türkçe karakter (Masaüstü) destekli okuma
    img_array = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        synthetic_noisy = add_gamma_speckle(img, looks=4)
        filename = os.path.basename(path)
        save_path = os.path.join(out_dir, filename)
        
        # Türkçe karakter (Masaüstü) destekli yazma
        is_success, im_buf_arr = cv2.imencode(".jpg", synthetic_noisy)
        if is_success:
            im_buf_arr.tofile(save_path)
            success_count += 1

print(f"{success_count} adet kusursuz sentetik SAR verisi GERÇEKTEN üretildi ve klasöre kaydedildi!")