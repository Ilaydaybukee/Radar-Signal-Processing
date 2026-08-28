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
# Gece boyu açık kalacağı için tur sayısını 100 yaptık. 
EPOCHS = 100  

# YENİ KLASÖR YOLUN (İçinde 'ship' ve 'sea' klasörleri olduğundan emin ol)
TRAIN_DIR = r"C:\Users\semih\OneDrive\Masaüstü\ilk eğitim"

train_dataset = SARDataset(folder_path=TRAIN_DIR)
print(f"Gece Mesaisine Giren Toplam Görüntü: {len(train_dataset)}\n")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# İkili (Binary) sınıflandırma (Ship ve Sea)
model = SARClassifier(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Gece Boyu Sürecek Dev Eğitim Başlıyor...\n")
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
    
    print(f"Gece Turu [{epoch+1}/{EPOCHS}] | Loss: {avg_train_loss:.4f} | Başarı: %{train_accuracy:.2f}")

# Sabah uyandığında bu dosyayı hazır bulacaksın
torch.save(model.state_dict(), "sar_ikili_gece_modeli.pth")
print("\nSabah Oldu, Eğitim Bitti! Zeka 'sar_ikili_gece_modeli.pth' olarak kaydedildi.")