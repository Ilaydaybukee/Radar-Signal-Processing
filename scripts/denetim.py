import re
from pathlib import Path

import pandas as pd

# Bir önceki adımda oluşturduğumuz CSV dosyasının yolu
CSV_PATH = Path(r"C:\Users\semih\OneDrive\Masaüstü\sar_aybüke\raw\results\image_metadata.csv")

# Sadece geçerli (okunabilen) görüntüleri alıyoruz
df = pd.read_csv(CSV_PATH)
df = df[df["is_valid"] == True].copy()

# Bildiğimiz sınıflar
KNOWN_CLASSES = ["Fishing", "Sailing", "Tanker", "Passenger", "Pleasure", "Cargo"]

def extract_metadata(filename):
    """Dosya adından sınıfı, ID'yi ve versiyonu (raw/clean vb.) çıkarır."""
    # 1. Sınıfı Bul
    label = "Unknown"
    for cls in KNOWN_CLASSES:
        if cls in filename:
            label = cls
            break
            
    # 2. Görüntü ID'sini Bul (Örn: _00004_ veya _00192_ gibi 4-5 haneli sayılar)
    id_match = re.search(r'_(\d{4,5})_', filename)
    image_id = id_match.group(1) if id_match else "Bilinmiyor"
    
    # 3. Versiyonu Bul (clean, raw vb.)
    # ID ve Sınıf dışındaki son kısımlar
    version = "Belirsiz"
    if label != "Unknown" and image_id != "Bilinmiyor":
        try:
            # novasar_sband_00004_Fishing_clean.tif -> clean
            parts = filename.replace(".tiff", "").replace(".tif", "").split("_")
            cls_index = parts.index(label)
            if len(parts) > cls_index + 1:
                version = "_".join(parts[cls_index + 1:])
        except ValueError:
            pass
            
    return pd.Series([image_id, label, version])

# Dosya adından verileri çıkarıp yeni sütunlar olarak ekliyoruz
df[["image_id", "label", "version"]] = df["filename"].apply(extract_metadata)

# Hangi alt klasörde olduklarını buluyoruz (raw, processed vb.)
df["folder_name"] = df["relative_path"].apply(lambda x: str(Path(x).parent))

print("\n" + "="*50)
print("1. KLASÖR DAĞILIMI")
print("="*50)
print(df["folder_name"].value_counts())

print("\n" + "="*50)
print("2 & 3. SINIF DAĞILIMI (Görüntü Sayısı)")
print("="*50)
print(df["label"].value_counts())

print("\n" + "="*50)
print("4 & 5. TEKRARLAYAN GÖRÜNTÜ KONTROLÜ (Aynı ID'ye sahip dosyalar)")
print("="*50)
duplicate_ids = df[df.duplicated(subset=["image_id"], keep=False) & (df["image_id"] != "Bilinmiyor")]

if not duplicate_ids.empty:
    print(f"UYARI: Aynı görüntü ID'sine sahip {len(duplicate_ids)} dosya bulundu! (Farklı versiyonlar veya kopyalar)")
    print("Örnek 10 çakışan dosya:")
    print(duplicate_ids[["filename", "folder_name", "label", "version"]].sort_values("image_id").head(10).to_string(index=False))
else:
    print("Harika! Tekrar eden veya aynı ID'ye sahip farklı versiyon (raw/clean) bulunmadı.")

print("\n" + "="*50)
print("6. 'UNKNOWN' SINIFLI DOSYALAR")
print("="*50)
unknowns = df[df["label"] == "Unknown"]
if not unknowns.empty:
    print(f"DİKKAT: Sınıfı belirlenemeyen {len(unknowns)} dosya var. Örnekler:")
    print(unknowns["filename"].head(10).tolist())
else:
    print("Bilinmeyen sınıfa ait dosya yok, isimler kurallı.")

# Analiz sonucunu yeni bir CSV olarak kaydedelim (Eğitimde bunu kullanacağız)
output_path = CSV_PATH.parent / "audited_dataset.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\nDenetim tablosu şuraya kaydedildi: {output_path}")