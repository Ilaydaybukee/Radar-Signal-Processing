# Radar-Signal-Processing-Bitirme

## Veri Klasör Yapısı

Proje kapsamında temiz SAR görüntüleri `data/clean` klasörü altında toplanacaktır.

Avrupa bölgesi için veri klasörleri:

```text
data/clean/europe/
├── agriculture
├── coast_port
├── mountain_forest
├── river_bridge
├── ship_sea
└── urban
```

## Kullanılan Veri Kaynağı

Avrupa kıyı bölgesi SAR örnekleri için Kaggle üzerinde bulunan CoastLine-DualPol veri seti kullanılmıştır. Bu veri seti, Avrupa kıyı bölgelerinden elde edilmiş Sentinel-1 SAR genlik görüntü patch'lerini ve bunlara ait kıyı maskelerini içermektedir.

Bu projede ilk aşamada yalnızca `Training/Img2Pol` klasöründeki SAR görüntü patch'leri kullanılmıştır. `Mask` klasöründeki etiket/mask dosyaları temiz görüntü veri klasörüne dahil edilmemiştir.

Seçilen `.mat` dosyalarındaki `ImgPol` değişkeninin birinci polarizasyon kanalı PNG formatına dönüştürülerek `data/clean/europe/coast_port` klasörüne eklenmiştir.

### SARscope Maritime Images

`ship_sea` klasöründeki ilk SAR gemi/deniz örnekleri için Kaggle üzerinde bulunan SARscope: Synthetic Aperture Radar Maritime Images veri seti kullanılmıştır.

Bu veri seti HRSID ve OPEN-SSDD kaynaklarından işlenmiş SAR gemi görüntülerini içermektedir. Projede ilk aşamada yalnızca `.jpg` görüntü dosyaları kullanılmış, `_annotations.coco.json` etiket dosyası temiz görüntü veri klasörüne dahil edilmemiştir.

Seçilen görüntüler `data/clean/europe/ship_sea` klasörüne eklenmiştir.
