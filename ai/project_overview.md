---
name: project-overview
description: HAM10000 melanoma projesi — ne yapıldı, ne çalıştı, temel sayılar, güncel durum
metadata:
  type: project
---

# Proje Özeti

**Veri seti:** HAM10000 — 10,015 dermoskopi görüntüsü, 7 tanı kodu.
**Görev:** Binary sınıflandırma → mel (1,113 görüntü) vs. non-mel (8,902 görüntü). Oran ~1:8.
**Split:** Lesion-grouped stratified 70/15/15 → train 7,008 / val 1,503 / test 1,504.
**Platform:** Google Colab Pro+ (A100 40 GB GPU).
**Ortam:** Python 3.10, PyTorch 2.x, timm 0.9, scikit-learn 1.4.

## Benchmark Yöntemler (eski sonuçlar — yeniden eğitim devam ediyor)

| # | Yöntem | F1 | AUC |
|---|---|---|---|
| 1 | Logistic Reg. (raw pixels 64×64) | 0.3132 | 0.7337 |
| 2 | Classical ML (HOG+HSV+GLCM → PCA → RBF-SVM) | 0.4356 | 0.8350 |
| 3 | AlexNet | 0.5135 | 0.8925 |
| 4 | VGG16-BN ⚠️ P=R=F1 anomalisi | 0.6228 | 0.9190 |
| 5 | ResNet50 | 0.6836 | 0.9470 |
| 6 | EfficientNet-B3 | 0.5769 | 0.9189 |
| 7 | DenseNet121 | 0.6149 | 0.9369 |
| 8 | Swin-Tiny | 0.6575 | 0.9400 |
| **9** | **Soft-vote ensemble (3-8)** | **0.6967** | **0.9540** |
| 10 | Hybrid fusion (HOG+ABCD+deep → SMOTE → XGBoost/MLP) | 0.6647 | 0.9218 |

**Hedef:** F1 ≥ 0.78 (ISIC 2019 + güçlü augmentation ile)

## Güncel Durum (2026-06-20)

### Yapılan kod değişiklikleri (hepsi push edildi, GitHub'da):

**config.py:**
- Augmentation güçlendirildi: rotation 15→30°, colorjitter 2-4×, crop 0.85→0.70, erasing 0.10→0.20
- MIXUP_ALPHA=0.2, MIXUP_PROB=0.3 yeniden aktif (ABCD sadece method 10'da, CNN'lerde yok)

**src/data.py:**
- `load_arrays_extended()` eklendi — ISIC 2019 MEL'i training'e otomatik ekler
- Val/test saf HAM10000 kalır (adil benchmark)

**notebooks/04-09 (tüm CNN'ler):**
- cell-5: `load_arrays_extended()` kullanıyor (ISIC 2019 otomatik eklenir)
- cell-9 (stage-2): mixup parametreleri config'den okuyor

**notebooks/05_vgg16_bn:** P=R=F1 anomalisi için confusion matrix diagnostic hücresi eklendi

**notebooks/10_aggregation:**
- Recall-biased threshold sweep (≥0.90)
- Bootstrap 95% CI (n=1000, seed=42)
- Ablasyon tablosu genişletildi
- Literatür tablosu Section A (adil) + Section B (context) olarak bölündü

**notebooks/11_hybrid_fusion:** XGBoost early_stopping_rounds=30 + kalibrasyon diagnostiği

**notebooks/12_isic2019_prep.ipynb:** YENİ — ISIC 2019 indir + önişle + Drive'a kaydet

### Colab'da yapılacaklar (sırayla):

1. ✅ **12_isic2019_prep.ipynb** — T4 GPU — ~45 dk — ISIC 2019 MEL'i önişle
2. ⏳ **04_alexnet.ipynb** — A100 — ~40 dk — yeniden eğit
3. ⏳ **05_vgg16_bn.ipynb** — A100 — ~60 dk — yeniden eğit + anomali kontrol
4. ⏳ **06_resnet50.ipynb** — A100 — ~50 dk — yeniden eğit
5. ⏳ **07_efficientnet_b3.ipynb** — A100 — ~55 dk — yeniden eğit
6. ⏳ **08_densenet121.ipynb** — A100 — ~50 dk — yeniden eğit
7. ⏳ **09_swin_tiny.ipynb** — A100 — ~60 dk — yeniden eğit
8. ⏳ **10_aggregation.ipynb** — T4 — ~10 dk — yeni ensemble + CI + threshold
9. ⏳ **11_hybrid_fusion.ipynb** — T4 — ~20 dk — XGBoost fix

## Beklenen İyileşme

| Etken | Etki |
|---|---|
| ISIC 2019 (779→5301 melanoma) | F1 +0.06 ~ +0.10 tahmini |
| Güçlü augmentation | F1 +0.02 ~ +0.04 tahmini |
| **Birlikte** | **F1 ~0.78 ~ 0.84 hedef** |

## Preprocessing

DullRazor hair removal (17×17 cross kernel) → Otsu segmentation (LAB-L channel)
→ largest connected component → 15% margin bbox crop → resize → RGB uint8.
**X_all.npy boyutu: 224×224** (config.IMG_SIZE=448 ama veri 224 ile kaydedilmiş — kritik not)

## CNN Ortak Recipe (güncel, methods 3–8)

Focal loss (γ=2, β=0.999) + WeightedRandomSampler + AdamW + WarmupCosine LR
+ EMA (0.999) + 8-way TTA + val-tuned decision threshold.
Augmentation: GÜÇLENDİRİLDİ (rotation ±30°, ColorJitter 0.25, shear, GaussianBlur, RandomErasing p=0.20).
Mixup α=0.2, p=0.3 — stage-2'de aktif.

## Dosya Yapısı

```
src/        → data.py, preprocessing.py, segmentation.py, abcd.py,
              features.py, models.py, training.py, evaluation.py, gradcam.py
notebooks/  → 00_data_setup ... 12_isic2019_prep
results_csv/→ per-method JSON + CSV
config.py   → tüm hyperparameter tek yer
ai/         → bu memory sistemi
```

**Why:** Revizyon için [[methodology-flaws]] listesini öncelik sırasıyla takip et.
