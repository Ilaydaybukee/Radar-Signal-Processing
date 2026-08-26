from pathlib import Path
import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms  # Görüntü boyutlandırmak için eklendi

class SARDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)
        
        # Hem senin TIF'leri hem de Kaggle'ın JPG/PNG'lerini bulacak
        extensions = ["*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.png"]
        image_paths = []
        for ext in extensions:
            image_paths.extend(self.folder_path.rglob(ext))
        
        self.class_to_idx = {
            "sea": 0,
            "ship": 1
        }
        
        self.samples = []
        
        for img_path in image_paths:
            folder_name = img_path.parent.name.lower()
            if folder_name in self.class_to_idx:
                self.samples.append((img_path, self.class_to_idx[folder_name]))
                
        # --- SİHİRLİ DOKUNUŞ: Bütün resimleri 256x256 piksele sabitle ---
        self.resize = transforms.Resize((256, 256))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx = self.samples[idx]
        
        if img_path.suffix.lower() in ['.tif', '.tiff']:
            image = tifffile.imread(str(img_path))
        else:
            image = np.array(Image.open(img_path).convert('L'))
        
        if len(image.shape) == 3:
            image = image[:, :, 0]
            
        image = np.expand_dims(image, axis=0) 
        image_tensor = torch.from_numpy(image).float() / 255.0 
        
        # --- Okunan her resim burada 256x256 boyutuna zorlanır ---
        image_tensor = self.resize(image_tensor)
        
        label_tensor = torch.tensor(label_idx, dtype=torch.long)
        
        return image_tensor, label_tensor