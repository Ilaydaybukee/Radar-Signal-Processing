import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SARDataset 
from model import SARClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Eğitim için kullanılacak cihaz: {device}\n")

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 50  # Üzerine ekleme yaptığımız için 50 tur yeterli

# 1. YENİ VERİ SETİNİN YOLU (İkinci eğitim)
TRAIN_DIR = r"C:\Users\semih\OneDrive\Masaüstü\eğitim iki"
train_dataset = SARDataset(folder_path=TRAIN_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"İkinci Aşamaya Giren Toplam Görüntü: {len(train_dataset)}\n")

# 2. MODELİ ÇAĞIR VE GECEKİ HAFIZAYI YÜKLE
model = SARClassifier(num_classes=2).to(device)

# Ağırlıkları weights klasöründen çekiyoruz
ESKI_MODEL_YOLU = r"C:\Users\semih\OneDrive\Masaüstü\analizzz\weights\sar_ikili_gece_modeli.pth"
model.load_state_dict(torch.load(ESKI_MODEL_YOLU))
print("Önceki gece eğitimi başarıyla yüklendi. Üstüne inşa başlıyor!\n")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 3. YENİ EĞİTİM DÖNGÜSÜ
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
    train_accuracy = 100 * correct_train / total_train
    avg_train_loss = running_loss / len(train_loader)
    
    print(f"Part 2 Turu [{epoch+1}/{EPOCHS}] | Loss: {avg_train_loss:.4f} | Başarı: %{train_accuracy:.2f}")

# 4. YENİ MODELİ WEIGHTS KLASÖRÜNE KAYDET
YENI_MODEL_YOLU = r"C:\Users\semih\OneDrive\Masaüstü\analizzz\weights\sar_ikili_ikinci_model.pth"
torch.save(model.state_dict(), YENI_MODEL_YOLU)
print(f"\nİkinci eğitim bitti! Yeni zeka kaydedildi: {YENI_MODEL_YOLU}")