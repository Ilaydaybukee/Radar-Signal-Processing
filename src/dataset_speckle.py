import os
from pathlib import Path
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class SARDespeckleDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, img_size=256):
        """
        noisy_dir: Benek gürültülü (speckle) SAR görüntülerinin olduğu klasör
        clean_dir: Referans (hedef) temiz görüntülerin olduğu klasör
        """
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        
        # Klasördeki tüm geçerli dosyaları bul ve isme göre sırala
        valid_ext = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
        self.image_filenames = sorted([f for f in os.listdir(noisy_dir) if f.lower().endswith(valid_ext)])
        
        # Tüm görüntüleri makale standartlarına uygun sabit bir boyuta zorla
        self.resize = transforms.Resize((img_size, img_size))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        
        noisy_path = self.noisy_dir / filename
        clean_path = self.clean_dir / filename # İsimlerin birebir aynı olması şart!
        
        # Görüntüleri uzantısına göre TIF veya standart formatta oku
        if noisy_path.suffix.lower() in ['.tif', '.tiff']:
            noisy_img = tifffile.imread(str(noisy_path))
            clean_img = tifffile.imread(str(clean_path))
        else:
            noisy_img = np.array(Image.open(noisy_path).convert('L'))
            clean_img = np.array(Image.open(clean_path).convert('L'))
            
        # Çok kanallı görüntüler varsa tek kanala (Grayscale) indirge
        if len(noisy_img.shape) == 3:
            noisy_img = noisy_img[:, :, 0]
        if len(clean_img.shape) == 3:
            clean_img = clean_img[:, :, 0]
            
        # PyTorch Tensörüne Çevir, [1, H, W] kanal formatına sok ve normalize et
        noisy_tensor = torch.from_numpy(np.expand_dims(noisy_img, axis=0)).float() / 255.0
        clean_tensor = torch.from_numpy(np.expand_dims(clean_img, axis=0)).float() / 255.0
        
        # Yeniden boyutlandırma
        noisy_tensor = self.resize(noisy_tensor)
        clean_tensor = self.resize(clean_tensor)
        
        # Sadece resimleri değil, dosya adını da döndürüyoruz
        # Bu sayede test aşamasında C, S ve L bantlarını isimlerinden ayırabileceğiz
        return noisy_tensor, clean_tensor, filename