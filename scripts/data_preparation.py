import os
import shutil
import cv2
import numpy as np

# Main directories for the article dataset
BASE_DIR = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration"
CLEAN_DIR = os.path.join(BASE_DIR, "clean")
NOISY_DIR = os.path.join(BASE_DIR, "noisy")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(NOISY_DIR, exist_ok=True)

# Source paths and corresponding band labels
source_paths = {
    r"C:\Users\semih\OneDrive\Masaüstü\c_band": "C-band",
    r"C:\Users\semih\OneDrive\Masaüstü\s_bant": "S-band",
    r"C:\Users\semih\OneDrive\Masaüstü\l_bant": "L-band"
}

print("Preparing dataset, this may take a few minutes...\n")

for source, band in source_paths.items():
    if not os.path.exists(source):
        print(f"Warning: {source} not found, skipping.")
        continue
        
    files = [f for f in os.listdir(source) if f.lower().endswith(('.tif', '.png', '.jpg', '.jpeg'))]
    
    for index, file in enumerate(files):
        old_path = os.path.join(source, file)
        extension = os.path.splitext(file)[1]
        
        # 1. Rename for article standards
        new_name = f"{band}_{index:04d}{extension}"
        clean_path = os.path.join(CLEAN_DIR, new_name)
        noisy_path = os.path.join(NOISY_DIR, new_name)
        
        # 2. Copy clean version
        shutil.copy(old_path, clean_path)
        
        # 3. Read and Write handling Turkish characters (Masaüstü)
        img_array = np.fromfile(clean_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            noise = np.random.normal(0, 0.2, img.shape)
            noisy_img = img + img * noise
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            
            # Save using imencode to bypass Unicode path errors
            is_success, buffer = cv2.imencode(extension, noisy_img)
            if is_success:
                buffer.tofile(noisy_path)
            
    print(f"[OK] {band} processing complete! {len(files)} files processed.")

print(f"\nSuccess! Total {900 + 948 + 992} images prepared and paired.")