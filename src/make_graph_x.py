import pandas as pd
import matplotlib.pyplot as plt

# Eğitim loglarını oku
df = pd.read_csv("training_history.csv")

# Çizim ayarları
plt.figure(figsize=(8, 5))
plt.plot(df['epoch'], df['train_loss'], label='Eğitim Kaybı (Train Loss)', color='blue', linewidth=2)
plt.plot(df['epoch'], df['val_loss'], label='Doğrulama Kaybı (Val Loss)', color='orange', linewidth=2, linestyle='--')

# Etiketler ve başlık
plt.title('Şekil X: Eğitim ve Doğrulama Kayıp Eğrileri (Loss Curve)')
plt.xlabel('Epoch (Tur)')
plt.ylabel('Kayıp Değeri (MSE)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# PNG olarak proje dizinine kaydet
plt.savefig('sekil_x_grafik.png', dpi=300, bbox_inches='tight')
print("Şekil X başarıyla 'sekil_x_grafik.png' olarak kaydedildi!")