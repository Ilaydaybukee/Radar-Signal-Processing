import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from dataset_speckle import SARDespeckleDataset
from model_speckle import HybridDnCNNUnet

# --- YAPILANDIRMA ---
NOISY_DIR = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration\noisy"
CLEAN_DIR = r"C:\Users\semih\OneDrive\Masaüstü\sar_restoration\clean"
BATCH_SIZE = 4  # CUDA belleğini şişirmemek için 4 olarak ayarlandı
LEARNING_RATE = 0.0001
EPOCHS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eğitim Cihazı: {device}")

# --- VERİ HAZIRLIĞI ---
dataset = SARDespeckleDataset(noisy_dir=NOISY_DIR, clean_dir=CLEAN_DIR)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- MODEL VE MATEMATİK ---
model = HybridDnCNNUnet(channels=1).to(device)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

history = []

# --- EĞİTİM DÖNGÜSÜ ---
for epoch in range(EPOCHS):
    # 1. Eğitim Aşaması
    model.train()
    train_loss = 0.0
    
    for noisy, clean, _ in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train"):
        # CPU'daki resimleri GPU'ya taşıyan kritik satır
        noisy, clean = noisy.to(device), clean.to(device)
        
        optimizer.zero_grad()
        outputs = model(noisy)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)
    
    # 2. Doğrulama (Validation) Aşaması
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for noisy, clean, _ in tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Val"):
            # CPU'daki resimleri GPU'ya taşıyan kritik satır
            noisy, clean = noisy.to(device), clean.to(device)
            
            outputs = model(noisy)
            loss = criterion(outputs, clean)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] Sonucu | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n")
    
    # Makaledeki Şekil X'in üretilmesi için değerlerin kaydedilmesi[cite: 1]
    history.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss
    })

# --- ÇIKTILARI KAYDETME ---
# Eğitim ve doğrulama kayıp değerlerini içeren CSV dosyasının oluşturulması[cite: 1]
history_df = pd.DataFrame(history)
history_df.to_csv("training_history.csv", index=False)
print("Makale Şekil X için 'training_history.csv' başarıyla oluşturuldu.")

torch.save(model.state_dict(), "hybrid_speckle_model.pth")
print("Model ağırlıkları 'hybrid_speckle_model.pth' olarak kaydedildi.")