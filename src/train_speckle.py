import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
from tqdm import tqdm

from dataset_speckle import SARDespeckleDataset
from model_speckle import HybridDnCNNUnet

# KESİN ÇÖZÜM: Yollar doğrudan orijinal sar_restoration klasörüne bakıyor
NOISY_DIR = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration\sentetik_noisy"
CLEAN_DIR = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration\clean"
BATCH_SIZE = 4  
ACCUMULATION_STEPS = 8 # 32 Efektif Batch Size
LEARNING_RATE = 0.0002
EPOCHS = 200 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eğitim Cihazı: {device} | Hedef: Sentetik Veri, 200 Epoch & 32 Efektif Batch")

dataset = SARDespeckleDataset(noisy_dir=NOISY_DIR, clean_dir=CLEAN_DIR)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = HybridDnCNNUnet(channels=1).to(device)

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad() 
    
    for i, (noisy, clean, _) in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train")):
        noisy, clean = noisy.to(device), clean.to(device)
        
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        
        loss = loss / ACCUMULATION_STEPS
        loss.backward()
        
        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
        train_loss += loss.item() * ACCUMULATION_STEPS
        
    avg_train_loss = train_loss / len(train_loader)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy, clean, _ in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Val"):
            noisy, clean = noisy.to(device), clean.to(device)
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr}\n")
    history.append({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

pd.DataFrame(history).to_csv("training_history.csv", index=False)
torch.save(model.state_dict(), "hybrid_speckle_model.pth")
print("Sentetik verilerle eğitilmiş mükemmel SAR modeli kaydedildi!")