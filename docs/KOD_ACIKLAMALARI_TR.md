# Kod Açıklamaları ve Temel Kavramlar

## Dosyaların akışı
`common.py` kökü, seed'i, TIFF okuma, hash, güvenli percentile normalizasyonu ve görev kararını merkezileştirir. `01_data_audit.py` dosyaları salt okunur açıp metadata/tekrar/geçersizlik tablolarını yazar. `02_dataset_preview.py` yalnız gösterim kopyasını normalize eder. `03_image_processing.py` sekiz yöntemi her defasında ham diziden başlatır. `04_prepare_splits.py` aynı hash'i bir kez tutar ve stratified 70/15/15 böler.

`05_dataset.py` TIFF'i `1×256×256` tensöre letterbox ile dönüştürür; rastgele flip, küçük rotasyon, blur ve speckle yalnız train'de çalışır. `06_model.py` üç Conv–BatchNorm–ReLU bloğu, iki pooling, adaptive pooling, dropout ve Linear içerir. `trainlib.py` ortak loader/eğitim/inference işlemlerini sağlar. `07_train.py` CUDA AMP, CrossEntropyLoss, AdamW, scheduler ve early stopping uygular. `08_evaluate.py` test metriklerini ve hatalı dosyaları kaydeder. `09_compare_preprocessing.py` her yöntem için modeli seed 42 ile sıfırlar. `10_robustness_test.py` kontrollü bozulmaları ölçer. `11_predict.py` tek görüntüyü aynı giriş yoluyla tahmin eder.

## Kavram sözlüğü
- **NumPy dizisi:** Piksel değerlerini boyutlar ve veri tipiyle tutan sayısal yapı.
- **TIFF veri tipi:** `uint8`, `uint16`, `float32` gibi her pikselin hassasiyet/aralığını belirler; doğrudan 8 bite kesmek bilgi kaybettirir.
- **Normalization:** Farklı aralıkları öğrenilebilir ortak aralığa dönüştürür.
- **Percentile clipping:** Çok az sayıdaki aşırı piksel yerine örneğin %1–%99 sınırlarını kullanır.
- **Speckle noise:** Koherent SAR ölçümünde görülen çarpımsal benek gürültüsüdür.
- **Lee filter:** Yerel ortalama/varyansla benek azaltırken kenarları korumaya çalışır.
- **CLAHE:** Yerel histogram eşitlemesini küçük bölgelerde ve kontrast sınırıyla uygular.
- **Convolution:** Küçük öğrenilebilir filtreyi görüntü üzerinde gezdirip örüntü çıkarır.
- **Batch normalization:** Ara aktivasyonların ölçeğini dengeler. **ReLU** negatifleri sıfırlar. **Pooling** uzamsal boyutu küçültür.
- **Loss:** Tahmin hatasının türevlenebilir sayısıdır. **Backpropagation** bu hatanın gradyanını katmanlara taşır. **Optimizer** gradyanla ağırlıkları günceller.
- **Epoch:** Train setinin tam geçişi; **batch** tek güncellemede kullanılan alt kümedir.
- **Early stopping:** Validation F1 iyileşmezse eğitimi keser. **Overfitting:** Train'i ezberleyip yeni veride kötüleşmedir.
- **Confusion matrix:** Gerçek/tahmin sınıf çapraz sayılarıdır. **Precision:** seçilenlerin doğruluğu; **recall:** gerçeklerin yakalanma oranı; **F1:** ikisinin harmonik ortalamasıdır.
- **Data leakage:** Aynı/ilişkili örneğin farklı splitlerde bulunup skoru yapay artırmasıdır; hash kontrolü bunu engeller.
