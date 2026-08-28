from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm


# Eğitim verilerinin bulunduğu hedef klasör yolu
PROJECT_DIR = Path(r"C:\Users\semih\OneDrive\Masaüstü\sar_aybüke\raw")

# Analiz sonuçlarının kaydedileceği klasör (raw klasörünün içine results adında açılır).
RESULTS_DIR = PROJECT_DIR / "results"

# results klasörü yoksa oluşturur.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Proje klasörü ve alt klasörlerdeki bütün TIFF dosyalarını bulur.
image_paths = list(PROJECT_DIR.rglob("*.tif"))
image_paths += list(PROJECT_DIR.rglob("*.tiff"))

# .venv içindeki dosyaları tarama listesinden çıkarır.
image_paths = [
    path
    for path in image_paths
    if ".venv" not in path.parts
]

# Görüntü bilgilerinin tutulacağı boş liste.
records = []

# Bütün TIFF görüntülerini sırayla inceler.
for image_path in tqdm(
    image_paths,
    desc="Görüntüler kontrol ediliyor"
):
    try:
        # TIFF görüntüsünü NumPy dizisi olarak okur.
        image = tifffile.imread(image_path)

        # Görüntü iki boyutluysa tek kanallıdır.
        if image.ndim == 2:
            height, width = image.shape
            channels = 1

        # Görüntü üç boyutluysa kanal bilgisi vardır.
        elif image.ndim == 3:
            height, width = image.shape[:2]
            channels = image.shape[2]

        # Beklenmeyen bir boyut varsa değerleri boş bırakır.
        else:
            height = None
            width = None
            channels = None

        # Görüntü bilgilerini listeye ekler.
        records.append(
            {
                "filename": image_path.name,
                "relative_path": str(
                    image_path.relative_to(PROJECT_DIR)
                ),
                "height": height,
                "width": width,
                "channels": channels,
                "dimensions": image.ndim,
                "dtype": str(image.dtype),
                "minimum_pixel": float(np.min(image)),
                "maximum_pixel": float(np.max(image)),
                "mean_pixel": float(np.mean(image)),
                "is_valid": True,
                "error": ""
            }
        )

    except Exception as error:
        # Görüntü okunamazsa program çökmez.
        # Hatalı dosyanın bilgisi kaydedilir.
        records.append(
            {
                "filename": image_path.name,
                "relative_path": str(
                    image_path.relative_to(PROJECT_DIR)
                ),
                "height": None,
                "width": None,
                "channels": None,
                "dimensions": None,
                "dtype": None,
                "minimum_pixel": None,
                "maximum_pixel": None,
                "mean_pixel": None,
                "is_valid": False,
                "error": str(error)
            }
        )


# Toplanan bilgileri tabloya dönüştürür.
metadata_df = pd.DataFrame(records)

# Tabloyu CSV dosyası olarak kaydeder.
metadata_path = RESULTS_DIR / "image_metadata.csv"

metadata_df.to_csv(
    metadata_path,
    index=False,
    encoding="utf-8-sig"
)


# Terminale özet sonuçları yazdırır.
print("\nKontrol tamamlandı.")
print(f"Bulunan TIFF sayısı: {len(metadata_df)}")

if not metadata_df.empty:

    valid_count = int(metadata_df["is_valid"].sum())
    invalid_count = len(metadata_df) - valid_count

    print(f"Sağlam görüntü sayısı: {valid_count}")
    print(f"Bozuk görüntü sayısı: {invalid_count}")

    print("\nEn sık görülen görüntü boyutları:")

    print(
        metadata_df[
            ["height", "width", "channels"]
        ].value_counts().head(10)
    )

    print("\nVeri tipleri:")

    print(
        metadata_df["dtype"].value_counts()
    )

print(f"\nSonuç dosyası: {metadata_path}")