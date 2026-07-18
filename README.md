# SAR AYBUKE — Güvenli SAR görüntü işleme ve CNN eğitimi

Bu proje, ham TIFF dosyalarını değiştirmeden denetler; gerçek dosya adı etiketlerine göre görev seçer; tek kanallı küçük bir CNN'i CUDA ile eğitir ve değerlendirir. **Sea yoksa Ship/Sea etiketi uydurulmaz.** Varsayılan Windows veri yolu `C:\SAR_AYBUKE` olsa da bütün kod yollarını depo kökünden güvenli biçimde çözer.

## Klasör yapısı
- `src/`: 01–11 sıralı iş akışı ve ortak modüller
- `configs/config.yaml`: eğitim ayarları
- `data/processed/`: hamdan bağımsız filtre çıktıları (Git dışında)
- `results/{tables,figures}`: raporlar
- `models/`: en iyi ağırlık (Git dışında)
- `tests/`: veri, normalizasyon, model ve sızıntı testleri

## Kurulum ve CUDA
Python 3.12 PowerShell ortamında `.venv` kullanıcı tarafından yönetilir:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
NVIDIA sürücüsüne uygun CUDA destekli PyTorch paketi kurulmalıdır. `require_cuda: true` iken CPU eğitimi yapılmaz.

## İş akışı
```powershell
.\run_pipeline.ps1
.\run_pipeline.ps1 -ComparePreprocessing
```
İlk komut denetim → ön izleme → split → CUDA eğitimi → test → dayanıklılık sırasını izler. Uzun filtre karşılaştırması yalnız ikinci komutla eklenir. Elle çalıştırma için `python src/01_data_audit.py`, `python src/04_prepare_splits.py`, `python src/07_train.py`, `python src/08_evaluate.py` kullanılabilir.

## Denetim, eğitim ve değerlendirme
Denetim boyut, dtype, dinamik aralık, hash, bozuk/sabit görüntü, bant/polarizasyon/kaynak ipuçlarını CSV'ye yazar. Split hash tekrarlarını tek grupta tutar ve fiziksel dosya taşımaz. Eğitim AdamW, class weight, AMP, macro-F1 erken durdurma kullanır. Değerlendirme accuracy/precision/recall/F1, confusion matrix; ikili görevde ROC-AUC ve PR eğrisi üretir.

## Tek görüntü tahmini
```powershell
python src/11_predict.py --image "C:\SAR_AYBUKE\ornek.tif"
```
Sınıf, güven, bütün olasılıklar, model ve ön işleme adı yazdırılır.

## Görüntü işleme ve sonuçlar
Percentile, log, median, Gaussian, CLAHE, Lee, unsharp ve bilateral yöntemleri **birbirinden bağımsız** uygulanır. İstatistikler `results/tables`, grafikler `results/figures`, karar/raporlar `results/*.md` altındadır.

## Bilinen sınırlamalar
Etiket çıkarımı dosya adına dayanır ve uzman doğrulaması ister. En az iki sınıf ve sınıf başına en az üç benzersiz örnek yoksa eğitim yapılmaz. Küçük/veri kaynağına bağımlı setlerde skor genellenebilirliği sınırlıdır. CUDA zorunluluğu test/denetimi engellemez, yalnız eğitimi durdurur.
