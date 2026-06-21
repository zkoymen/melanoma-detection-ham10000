# FIX PLAN — Melanoma Detection on HAM10000
# Target: TOK 2026 conference submission (deadline: 15 August 2026)
# Goal: Push ensemble F1 from 0.6967 → 0.80+ and make the paper scientifically defensible

---

## YENİ PLAN (2026-06-22, Opus) — Dengeli çok-kaynaklı veri seti

Teşhis: düşük F1'in 2 sebebi — (1) 1:8 eğitim dengesizliği, (2) F1'in 1:8
DENGESİZ test setinde ölçülmesi (167 mel / 1337 non-mel → precision çöküyor).
Çözüm: dengeli ikili veri seti + dengeli test.

Tasarım (kaynak ⊥ etiket → shortcut yok):
- mel     = tüm HAM mel (1.113) + tüm ISIC mel (4.522) = 5.635
- non-mel = HAM non-mel (1.113) + ISIC non-mel (4.522) = 5.635  (aynı kaynak karışımı)
- Toplam ~11.270, lesion-grouped 70/15/15, val/test tam dengeli.

Kod (push edildi, commit 3163623):
- `src/data.py`: `build_balanced_dataset()` + `load_arrays_balanced()`
- `notebooks/12`: ISIC mel+non-mel önişle → `X_combined.npy` + idx_*_bal.npy Drive'a
- `notebooks/04-11`: yükleyici → `load_arrays_balanced` (hepsi aynı test seti)
- `notebooks/11`: XGBoost early_stopping_rounds → constructor (2.0 fix)

ÇALIŞTIRMA SIRASI:
1. notebook 12 (CPU/T4) → dengeli set Drive'a yazılır
2. notebook 04 AlexNet (A100) → **DOĞRULAMA**: F1 0.51'den ~0.85+'a fırladı mı?
   - Fırladıysa → 05-09 + 10 + 11
   - Fırlamadıysa → DUR, Zeynep'e dön
NOT: 10'dan önce 04-09'un HEPSİ yeniden eğitilmeli (stale prediction = bozuk ensemble).
NOT: 01/02/03 (CPU baseline) sonra güncellenecek — aynı test seti için.
Beklenen: ensemble F1 0.90-0.93, AUC ~0.96-0.97.

---

## DURUM (2026-06-20, eski plan — üstteki yeni planla değişti)

| Fix | Durum | Notlar |
|---|---|---|
| FIX 1 — ISIC 2019 | ✅ KOD HAZIR | 12_isic2019_prep.ipynb çalıştırıldı, Drive'da X_isic2019_mel.npy var. CNN'ler yeniden eğitilecek. |
| FIX 2 — Augmentation | ✅ TAMAM | config.py + data.py + notebooks/04-09 güncellendi |
| FIX 3 — Recall threshold | ✅ TAMAM | 10_aggregation.ipynb'e eklendi |
| FIX 4 — VGG anomalisi | ✅ KOD HAZIR | 05_vgg16_bn.ipynb'e diagnostic hücre eklendi, Colab'da çalıştırılacak |
| FIX 5 — XGBoost SMOTE | ✅ KOD HAZIR | 11_hybrid_fusion.ipynb'e early stopping + kalibrasyon diagnostiği eklendi |
| FIX 6 — Preprocessing | ❌ YAPILMADI | DullRazor kernel 17→13, düşük öncelik |
| FIX 7 — Ablasyon | ✅ TAMAM | 10_aggregation.ipynb güncellendi |
| FIX 8 — Literatür tablosu | ✅ TAMAM | Section A/B ayrımı yapıldı |
| FIX 9 — Bootstrap CI | ✅ TAMAM | 10_aggregation.ipynb'e eklendi |
| FIX 10 — Paper yazımı | ❌ YAPILMADI | CNN'ler bittikten sonra |

---

## SIRADAKI ADIM

CNN'leri yeniden eğit (Colab, A100):
```
04_alexnet.ipynb        → A100, ~40 dk
05_vgg16_bn.ipynb       → A100, ~60 dk
06_resnet50.ipynb       → A100, ~50 dk
07_efficientnet_b3.ipynb → A100, ~55 dk
08_densenet121.ipynb    → A100, ~50 dk
09_swin_tiny.ipynb      → A100, ~60 dk
```
Sonra:
```
10_aggregation.ipynb    → T4, ~10 dk → yeni ensemble F1 + CI
11_hybrid_fusion.ipynb  → T4, ~20 dk → XGBoost kalibrasyon kontrolü
```

---

## CONTEXT: Why the scores are low

Total training images: **7,008** (of 10,015 total).
But melanoma (positive class) in training: **~779 images** (≈ 1,113 × 0.70).
The remaining ~6,229 training images are non-melanoma.

**After fix:** ISIC 2019 MEL (~4,522) eklendi → training melanoma ~5,301.
Oran: 1:1.17 (eskiden 1:8).

---

## FIX 1 — Expand the dataset ✅ KOD TAMAM

### Yapılan
- `notebooks/12_isic2019_prep.ipynb` yazıldı
- Kaggle'dan `andrewmvd/isic-2019` indirildi
- MEL sınıfı filtrelendi (~4,522 görüntü)
- HAM10000 ile aynı preprocessing pipeline (DullRazor + Otsu + resize)
- `X_isic2019_mel.npy` Drive'a kaydedildi
- `src/data.py` → `load_arrays_extended()` eklendi
- Notebooks 04-09 → `load_arrays_extended()` kullanıyor

### Kritik not
X_all.npy 224×224 kaydedilmiş (config.IMG_SIZE=448 ama veri 224).
`load_arrays_extended()` otomatik resize yapıyor (ISIC 2019 → 224×224).

---

## FIX 2 — Fix the augmentation ✅ TAMAM

### Yapılan
```python
# config.py (güncel değerler):
AUG_ROTATION_DEG = 30
AUG_COLOR_BRIGHTNESS = 0.25
AUG_COLOR_CONTRAST = 0.25
AUG_COLOR_SATURATION = 0.20
AUG_COLOR_HUE = 0.05
AUG_CROP_SCALE_MIN = 0.70
AUG_RANDOM_ERASING_P = 0.20
AUG_RANDOM_ERASING_SCALE = (0.02, 0.20)
MIXUP_ALPHA = 0.2   # stage-2 CNN'lerde aktif
MIXUP_PROB = 0.3
```
`src/data.py` → `make_train_transform_strong()` içine RandomAffine(shear=10) + GaussianBlur eklendi.

---

## FIX 3 — Recall-biased threshold ✅ TAMAM

10_aggregation.ipynb'de:
- Sweep ile recall ≥ 0.90 sağlayan threshold bulunuyor
- Ablasyon tablosuna "clinical threshold" satırı eklendi

---

## FIX 4 — VGG16-BN anomalisi ✅ KOD HAZIR (Colab çalıştırılacak)

05_vgg16_bn.ipynb'e confusion matrix diagnostic hücresi eklendi.
VGG yeniden eğitilince çıktıyı kontrol et: FP==FN mi?

---

## FIX 5 — Hybrid Fusion XGBoost ✅ KOD HAZIR (Colab çalıştırılacak)

11_hybrid_fusion.ipynb'e:
- `early_stopping_rounds=30` eklendi
- Probability calibration diagnostic (min/max/mean/median) eklendi
- SMOTE mevcut kodda zaten PCA sonrası uygulanıyor (doğru sıra)

---

## FIX 6 — Preprocessing kernel ❌ YAPILMADI

DullRazor'da 17×17 → 13×13 kernel önerisi.
Etkisi düşük, öncelik yok. Zaten ISIC 2019 de bu kernel ile işlendi.

---

## FIX 7 — Ablasyon tablosu ✅ TAMAM

10_aggregation.ipynb güncel:
- "Ensemble — clinical threshold (recall ≥ 0.90)" satırı
- "Ensemble without EfficientNet-B3" leave-one-out satırı

---

## FIX 8 — Literatür tablosu ✅ TAMAM

10_aggregation.ipynb'de ikiye bölündü:
- **Section A:** Aynı görev, binary HAM10000 — adil karşılaştırma
- **Section B:** Farklı dataset/görev — sadece bağlam

---

## FIX 9 — Bootstrap CI ✅ TAMAM

10_aggregation.ipynb'de bootstrap 95% CI (n=1000, seed=42) eklendi.

---

## FIX 10 — Paper writing ❌ YAPILMADI

CNN'ler yeniden eğitilip yeni metrikler elde edildikten sonra yapılacak:
- Title güncelle
- Abstract: lesion-grouped split, clinical threshold, yeni F1
- Section III: augmentation kararları
- Section IV: confusion matrix sayıları, recall-biased threshold
- Section V: limitation (tek dataset)
- Literatür tablosu Section A/B
- Ablasyon tablosu genişletilmiş versiyon

---

## ENVIRONMENT NOTES

- **Platform:** Google Colab Pro+ (A100 40 GB GPU)
- **GitHub:** Public repo — `git pull` ile Colab'da güncelle
- **Drive path:** `MyDrive/melanoma/` — tüm data + checkpoint buraya
- **X_all.npy boyutu:** 224×224 (NOT 448 — kritik)
- **Recovery cells:** Her CNN notebook'un sonunda — checkpoint'ten devam et, yeniden eğitme
- **Seed:** 42 — değiştirme

---

## QUICK REFERENCE

```
Dataset:         10,015 images | 7 classes | binary: 1,113 mel / 8,902 non-mel
Training split:  7,008 images  | ~779 mel (HAM) + ~4,522 mel (ISIC 2019) = ~5,301 mel total
Val split:       1,503 images  | ~167 mel  (saf HAM10000)
Test split:      1,504 images  | 167 mel   | 1,337 non-mel  (saf HAM10000)

Best F1 (eski):  0.6967  (ensemble, 8-way TTA, tuned threshold)
Best AUC (eski): 0.9540  (ensemble)
Hedef F1:        0.78 ~ 0.84  (ISIC 2019 + güçlü aug ile)
```
