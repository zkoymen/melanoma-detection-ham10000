---
name: methodology-flaws
description: Projedeki tüm metodolojik hatalar — öncelik sırasıyla, sebepleriyle
metadata:
  type: project
---

# Metodolojik Hatalar — Öncelik Sırası

## CRITICAL (En büyük impact)

### 1. Tek Dataset, Sıfır Dış Validasyon
- Sadece HAM10000 (10K görüntü) kullanıldı.
- ISIC 2019 (25K), ISIC 2020 (33K), PH2 hiç kullanılmadı.
- Literatür hep daha büyük veri setlerinde test ediyor.
- Sonuç: Model HAM10000'e özgü özelliklere overfitting yapıyor olabilir.
- **Fix:** ISIC 2019 üzerinde en azından zero-shot test yapılmalı.

### 2. Çok Az Pozitif Örnek + Yetersiz Augmentation
- Training'de yaklaşık 779 melanoma örneği var (7,008 × 1113/10015).
- Bu kadar az pozitif için augmentation ÇOK muhafazakardı: ±15° rotasyon, colorjitter ≤0.10.
- Mixup/CutMix/RandAugment tamamen kapalı.
- **Gerçek sebep:** "ABCD sinyali bozulur" denildi AMA bu sadece klasik ML (method 2) için geçerli; CNN'ler ABCD kullanmıyor, piksel öğreniyor.
- **Fix:** CNN eğitiminde daha agresif augmentation (rotation ±30°, ColorJitter 0.3, Mixup α=0.2).
  Melanoma sınıfına ayrıca class-specific augmentation uygulanabilir.

### 3. Imbalanced Data — Yetersiz Çözüm
- 1:8 imbalance var. WeightedRandomSampler + Focal Loss uygulandı AMA:
  - Ablasyon: TTA ON (0.6115) vs TTA OFF (0.6144) → TTA biraz zarar veriyor!
  - Threshold=0.5 (0.5735) vs tuned (0.6115) → büyük fark, yani model calibrated değil.
  - En iyi ensemble recall = 0.6946, yani melanomalar %30 gözden kaçıyor.
- **Klinik açıdan**: Yüksek recall kritik. Threshold daha agresif düşürülmeli (precision-recall trade-off kabul edilmeli).
- **Fix:** ISIC 2018/2019 oversample + class-weighted SMOTE sadece training'de.

### 4. Binary Sınıflandırma Kararı Savunulamıyor Detaylıca
- 11 değil 7 sınıflı HAM10000'i binary yaptık.
- Bu klinik açıdan mantıklı (melanoma vs. rest) ama şöyle bir sorun var:
  BCC, AK gibi premalignant lezyonlar non-melanoma'ya girdi, oysa bunlar da tehlikeli.
- Daha önemli: 7-sınıflı yapıp sonra binary'e dönüştürmek yerine direkt binary eğitmek daha az bilgi içeriyor.
- **Fix:** Makale bu kararı daha iyi savunmalı; alternatif olarak 7-sınıf denenebilir.

### 5. Hybrid Fusion'daki XGBoost Eşiği = 0.09 (Şüpheli)
- XGBoost decision threshold 0.09 → neredeyse her şeyi pozitif predict ediyor olmalı.
- Ama recall 0.6587 ile makul — bu tutarsızlık; threshold sweep'te hata olmuş olabilir.
- SMOTE-Tomek 13,568 boyutlu uzayda uygulandı → curse of dimensionality sorunu.
- **Fix:** PCA(300) sonrası boyut hala fazla; daha iyi feature selection gerekiyor.

### 6. VGG16-BN'de Precision = Recall = F1 = 0.6228 (Anomali)
- P=R=F1 eşitliği istatistiksel olarak çok nadir; hesaplamada hata işareti.
- FP=FN olduğu durum mümkün ama bu denli tam eşitlik şüpheli.
- **Fix:** VGG notebookunu tekrar çalıştırıp kontrol et.

## HIGH (Önemli ama daha az kritik)

### 7. Preprocessing Sorunları
- DullRazor kernel 17×17 çok büyük → ince yapıları (damarcıklar, pigment deseni) silebilir.
- Otsu segmentasyonu LAB-L kanalında çalışıyor ama vignetting artifact'ları var.
- 448×448 store edilip bazı modeller 224'e küçültüyor → bilgi kaybı.
- Hair removal sonrası inpainting, piksel dağılımını bozuyor.

### 8. ImageNet Normalizasyon Dermoskopi için Suboptimal
- Dermoskopi görüntüleri ImageNet'ten çok farklı renk dağılımına sahip.
- Stain normalization (Macenko/Vahadane) veya dataset-specific mean/std daha iyi olurdu.

### 9. Cross-Validation Yapılmadı
- Tek bir 70/15/15 split var.
- Test set 1,504 görüntü → sadece 167 melanoma. Bu sayıyla güvenilir istatistik zor.
- k-fold (veya en azından 3-fold) güven aralıkları verecekti.

### 10. Dermoskopi-Özel Augmentation Eksik
- Saç simülasyonu (sentetik saç overlay)
- Ruler/calibration mark gizleme
- Elastic deformations (lezyonlar esnek şekilli)
- Daha agresif ColorJitter (dermoskopi cihazları arası renk farkları büyük)

## MEDIUM (Paper kalitesi için düzelt)

### 11. Literatür Karşılaştırması Adaletsiz
- Al-Waisy 2025 F1=0.9954 vs bizim 0.6967 — neden bu kadar fark var açıklanmadı.
- Fark: dataset büyüklüğü, dataset farklılığı, 7-sınıf vs binary, özel mimari.
- Karşılaştırmayı sadece HAM10000 üzerinde çalışanlarla yapmak gerekiyor.

### 12. Ablasyon Tablosu Yetersiz
- TTA sadece ON/OFF test edildi.
- Augmentation güçlü vs muhafazakar ablasyonu yok.
- Focal loss vs BCE ablasyonu yok.
- SMOTE olmadan hybrid fusion denenmiyor.

### 13. Error Analysis Yüzeysel
- Hangi melanoma alt tipi gözden kaçıyor? (nodular, superficial spreading, etc.)
- Hangi non-melanoma tipi en çok melanoma ile karıştırılıyor? (BKL? BCC?)
- Görüntü kalitesi (bulanık/kıllı/artefaktlı) ile hata arasındaki ilişki analiz edilmedi.

## Özet Öncelik

1. Daha büyük/farklı dataset veya dış validasyon → **en büyük F1 artışı**
2. Agresif augmentation melanoma sınıfı için → **F1'i 0.72+ yapabilir**
3. Recall-odaklı threshold seçimi → **klinik değer**
4. Hybrid XGBoost threshold anomalisi düzelt
5. VGG anomalisi kontrol et
6. Paper'ı revize et: literatür farkını açıkla, ablasyon genişlet
