import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import random

class SARDespeckleDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        
        # Tüm uzantıları sorunsuz yakalar (.JPG, .png, .tif)
        valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        self.images = [f for f in os.listdir(noisy_dir) if f.lower().endswith(valid_exts)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        noisy_path = os.path.join(self.noisy_dir, img_name)
        clean_path = os.path.join(self.clean_dir, img_name)

        # Türkçe karakter (Masaüstü) içeren dosya yolları için güvenli okuma
        noisy_img = cv2.imdecode(np.fromfile(noisy_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        clean_img = cv2.imdecode(np.fromfile(clean_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        noisy_tensor = torch.from_numpy(noisy_img).float().unsqueeze(0) / 255.0
        clean_tensor = torch.from_numpy(clean_img).float().unsqueeze(0) / 255.0

        # STRATEJİ DEĞİŞİKLİĞİ: Sıkıştırmak yerine orijinal boyuttan yama (patch) kırpıyoruz
        # 4 GB VRAM sınırını korumak için 256x256'lık pencere koordinatları belirleniyor
        i, j, h, w = T.RandomCrop.get_params(noisy_tensor, output_size=(256, 256))
        
        # Matematiksel hizalamanın bozulmaması için iki resim de tam olarak aynı noktadan kesiliyor
        noisy_tensor = TF.crop(noisy_tensor, i, j, h, w)
        clean_tensor = TF.crop(clean_tensor, i, j, h, w)

        # Dinamik Veri Çoğaltma (Augmentation)
        if random.random() > 0.5:
            noisy_tensor = TF.hflip(noisy_tensor)
            clean_tensor = TF.hflip(clean_tensor)
        
        if random.random() > 0.5:
            noisy_tensor = TF.vflip(noisy_tensor)
            clean_tensor = TF.vflip(clean_tensor)
        
        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            noisy_tensor = torch.rot90(noisy_tensor, k, [1, 2])
            clean_tensor = torch.rot90(clean_tensor, k, [1, 2])

        return noisy_tensor, clean_tensor, img_name