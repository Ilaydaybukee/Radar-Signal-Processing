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

### Sentinel-1&2 Image Pairs Urban Samples

`urban` klasöründeki SAR şehir/bina dokusu örnekleri için Kaggle üzerinde bulunan Sentinel-1&2 Image Pairs (SAR & Optical) veri seti kullanılmıştır.

Bu veri seti Sentinel-1 SAR ve Sentinel-2 optik görüntü çiftlerinden oluşmaktadır. Projede ilk aşamada yalnızca `urban` sınıfındaki Sentinel-1/SAR görüntüleri kullanılmış, optik/RGB görüntüler temiz SAR veri klasörüne dahil edilmemiştir.

Seçilen SAR görüntüleri `data/clean/europe/urban` klasörüne eklenmiştir.

### Sentinel-1&2 Image Pairs Agriculture Samples

`agriculture` klasöründeki SAR tarım alanı örnekleri için Kaggle üzerinde bulunan Sentinel-1&2 Image Pairs (SAR & Optical) veri seti kullanılmıştır.

Bu veri seti Sentinel-1 SAR ve Sentinel-2 optik görüntü çiftlerinden oluşmaktadır. Projede ilk aşamada yalnızca `agri` sınıfındaki Sentinel-1/SAR görüntüleri kullanılmış, optik/RGB görüntüler temiz SAR veri klasörüne dahil edilmemiştir.

Seçilen SAR görüntüleri `data/clean/europe/agriculture` klasörüne eklenmiştir.


### Sentinel-1&2 Image Pairs Grassland Samples

`mountain_forest` klasöründeki SAR doğal arazi/kırsal doku örnekleri için Kaggle üzerinde bulunan Sentinel-1&2 Image Pairs (SAR & Optical) veri seti kullanılmıştır.

Bu veri seti Sentinel-1 SAR ve Sentinel-2 optik görüntü çiftlerinden oluşmaktadır. Projede yalnızca Sentinel-1/SAR görüntüleri kullanılmış, optik/RGB görüntüler temiz SAR veri klasörüne dahil edilmemiştir.

`mountain_forest` klasörü için veri setindeki `grassland` sınıfı kullanılmıştır. Bu görüntüler doğrudan orman/dağ sınıfı değil, doğal açık arazi/kırsal doku örnekleri olarak değerlendirilmiştir.

`agriculture` klasörü için veri setindeki `agri` sınıfı kullanılmıştır.

`mountain_forest` klasörü için veri setindeki `grassland` sınıfı kullanılmıştır. Bu görüntüler doğrudan orman/dağ sınıfı değil, doğal açık arazi/kırsal doku örnekleri olarak değerlendirilmiştir.
