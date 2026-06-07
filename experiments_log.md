# Experiments Log

This document records the hardware, software stack, random seeds, and
hyperparameters used to produce every result in the paper. It is the
"Experiment Logs" deliverable from the final-project brief (hardware,
seed, hyperparameter list).

The live numerical results are in `MyDrive/melanoma/results/`:
`comparison_table.csv`, `ablation_table.csv`, `literature_comparison.csv`,
plus per-method `*_metrics.json` and `*_predictions.csv`. The aggregation
notebook (`notebooks/10_aggregation.ipynb`) reads those into the
publication tables.

---

## 1. Hardware

| Item | Value |
|---|---|
| Platform | Google Colab **Pro+** (paid tier) |
| GPU (training) | NVIDIA **A100 40 GB** (CUDA 12.x) |
| GPU (fallback) | NVIDIA L4 for retries when A100 unavailable |
| CPU runtime | for notebooks 00, 01, 02, 10, 11 (Standard runtime) |
| RAM | 83.5 GB (A100 high-RAM Colab Pro+ tier) |
| Disk | `/content` SSD ~225 GB; Drive FUSE for persistence |
| Compute units spent | ~50+ paid units across all final runs |

The course brief stated that the project could be completed on free
Colab. We deliberately used the paid Pro+ tier with A100 access so the
six CNNs could be **fully fine-tuned** (head + entire backbone) rather
than last-layer transfer-learned, and so 8-way test-time augmentation
could be applied to every test image without runtime pressure.

## 2. Software stack

| Package | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.x (Colab default at run time) |
| torchvision | 0.17 |
| timm (PyTorch Image Models) | 0.9 |
| scikit-learn | 1.4 |
| scikit-image | 0.21 |
| OpenCV (`cv2`) | 4.8 |
| imbalanced-learn (SMOTE-Tomek) | 0.12 |
| XGBoost | 2.0 |
| matplotlib | 3.7 |
| pandas | 2.x |
| numpy | 1.26 |

All packages are listed in `requirements.txt`. Colab provides PyTorch
and torchvision pre-installed; the notebook preambles only `pip install
--quiet timm` to add the timm dependency.

## 3. Reproducibility

| Item | Value |
|---|---|
| Global seed | **42** |
| numpy, torch, random, torch.cuda — all seeded | yes |
| Stratified split | lesion-grouped 70 / 15 / 15 (Tschandl / Cassidy 2022) |
| Split indices | saved as `idx_train.npy`, `idx_val.npy`, `idx_test.npy` to Drive |
| All methods evaluate on the **same** test set | yes |
| Train-time augmentation seeding | per-epoch via DataLoader workers; deterministic with seed |

## 4. Dataset

| Item | Value |
|---|---|
| Source | Kaggle `kmader/skin-cancer-mnist-ham10000` |
| Original size | 10,015 dermoscopic images, 7 dx codes |
| Binarisation | `y = 1 if dx == "mel" else 0` |
| Class counts (full) | melanoma 1,113 / non-melanoma 8,902 (~1:8) |
| Stored array shape | `(10015, 448, 448, 3)` uint8 RGB |
| Preprocessing | DullRazor hair removal → Otsu segmentation in LAB-L → largest CC → 15% margin bbox crop → resize 448×448 |
| Segmentation fallback rate | ~3.5% of images (Otsu rejected → centred square crop) |
| Split sizes | train 7,008  /  val 1,503  /  test 1,504 |
| Test class counts | non-mel ≈ 1,337  /  mel ≈ 167 |

## 5. Hyperparameters (shared CNN recipe — methods 3–8)

Single source of truth: `config.py`. Values used in every CNN notebook:

| Parameter | Value |
|---|---|
| Stage-1 LR (head-only) | 1e-3 |
| Stage-1 epochs | 3 |
| Stage-2 head LR | 3e-4 |
| Stage-2 backbone LR (discriminative) | 3e-5 |
| Stage-2 max epochs | 25 |
| Linear warm-up epochs | 3 |
| Cosine annealing minimum LR | 1e-6 |
| Early-stopping metric | validation F1 |
| Early-stopping patience | 7 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Gradient clipping | 1.0 (norm) |
| EMA decay | 0.999 |
| Focal loss γ | 2.0 |
| Focal loss β (class-balanced α) | 0.999 |
| WeightedRandomSampler | enabled (replacement=True) |
| Mixup α | 0.0 (intentionally disabled — see paper §III) |
| CutMix α | 0.0 (intentionally disabled) |
| RandAugment | disabled (color ops would distort ABCD signal) |
| RandomResizedCrop scale | (0.85, 1.00) |
| Rotation range | ±15° |
| ColorJitter — brightness | 0.10 |
| ColorJitter — contrast | 0.10 |
| ColorJitter — saturation | 0.05 |
| ColorJitter — hue | 0.02 |
| RandomErasing probability | 0.10 |
| RandomErasing scale | (0.02, 0.10) |
| ImageNet normalisation | mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) |
| TTA transforms | identity, hflip, vflip, hvflip, rot90, rot180, rot270, hflip+rot90 (8-way) |
| Decision threshold | F1-maximising on validation, sweep 181 points in [0.05, 0.95] |

### Per-architecture overrides

| Arch | Input size | Batch size | Pretrained source |
|---|---|---|---|
| AlexNet | 224 | 64 | torchvision IMAGENET1K_V1 |
| VGG16-BN | 224 | 16 | torchvision IMAGENET1K_V1 |
| ResNet50 | 320 | 32 | torchvision IMAGENET1K_V2 (improved recipe weights) |
| EfficientNet-B3 | 320 | 24 | timm `efficientnet_b3` pretrained |
| DenseNet121 | 320 | 32 | torchvision IMAGENET1K_V1 |
| Swin-Tiny | 224 | 32 | timm `swin_tiny_patch4_window7_224` pretrained |

### Classical / baseline hyperparameters

| Method | Key values |
|---|---|
| Method 1 — Logistic Regression baseline | input 64×64 raw pixels; `class_weight='balanced'`; `max_iter=1000`; default L2 regularisation |
| Method 2 — Classical ML SVM | input 128×128; HOG (8×8 pixels/cell, 2×2 cells/block); HSV color histogram (8 bins/channel); GLCM at distances=(1,) angles=(0°,45°,90°,135°); PCA(200); RBF-SVM with `C=1`, `gamma='scale'`, `class_weight='balanced'`, `probability=True` |
| Method 10 — Hybrid fusion | handcrafted (2292-D) + ABCD (13-D) + 6-CNN penultimate features concatenated (13,568-D = 2048 ResNet50 + 1024 DenseNet121 + 1536 EfficientNet-B3 + 768 Swin-Tiny + 4096 VGG16-BN + 4096 AlexNet) → StandardScaler → PCA(300) → SMOTE-Tomek (train only) → MLP [256-128-2] and XGBoost (n=500, depth=6, lr=0.05) — winner chosen by validation F1 |

## 6. Per-method results (test set, lesion-grouped split)

Recovered from `MyDrive/melanoma/results/*_metrics.json` on 2026-05-25.
All deep-learning rows use 8-way TTA at inference and a validation-tuned
decision threshold.

| Method | Acc | Prec | Rec | F1 | ROC-AUC | Threshold | Inf (ms/img) | Approx training time (A100) |
|---|---|---|---|---|---|---|---|---|
| Logistic regression baseline | 0.7957 | 0.2500 | 0.4192 | 0.3132 | 0.7337 | 0.5 | 0.03 | 217 s (CPU) |
| Classical ML SVM | 0.8397 | 0.3577 | 0.5569 | 0.4356 | 0.8350 | 0.5 | 1.13 | 31 s (CPU) |
| AlexNet | 0.8802 | 0.4680 | 0.5689 | 0.5135 | 0.8925 | 0.620 (TTA) | 0.78 | ~20 min (~1200 s) |
| VGG16-BN | 0.9162 | 0.6228 | 0.6228 | 0.6228 | 0.9190 | 0.585 (TTA) | 5.51 | ~50 min (~3000 s) |
| ResNet50 | 0.9255 | 0.6471 | 0.7246 | 0.6836 | 0.9470 | 0.440 (TTA) | 5.72 | ~35 min (~2100 s) |
| EfficientNet-B3 | 0.8975 | 0.5330 | 0.6287 | 0.5769 | 0.9189 | 0.600 (TTA) | 6.73 | ~40 min (~2400 s) |
| DenseNet121 | 0.9208 | 0.6690 | 0.5689 | 0.6149 | 0.9369 | 0.630 (TTA) | 7.67 | ~30 min (~1800 s) |
| Swin-Tiny | 0.9168 | 0.6061 | 0.7186 | 0.6575 | 0.9400 | 0.495 (TTA) | 7.60 | ~35 min (~2100 s) |
| **Ensemble (6 CNNs, soft-vote)** | **0.9328** | **0.6988** | **0.6946** | **0.6967** | **0.9540** | 0.555 | 5.67 | 0 (post-hoc aggregation) |
| Hybrid fusion (Method 10) | 0.9261 | 0.6707 | 0.6587 | 0.6647 | 0.9218 | 0.09 (XGBoost) | 0.0001 | 5.7 s (shallow only; 6-CNN features pre-extracted) |

Total CNN training time (all six) ≈ **3.5 hours of A100 compute**.

## 7. Ablation results (no extra training)

From `MyDrive/melanoma/results/ablation_table.csv`:

| Configuration | F1 |
|---|---|
| Threshold = 0.5 (no tuning), mean across 6 CNNs | 0.5735 |
| Threshold tuned on validation, mean across 6 CNNs | 0.6115 |
| TTA OFF (single forward), mean across 6 CNNs | 0.6144 |
| TTA ON (8-way average), mean across 6 CNNs | 0.6115 |
| Best single CNN (ResNet50, test) | 0.6836 |
| Soft-vote ensemble (6 CNNs) | **0.6967** |
| Loss/sampling: softened class-weight (legacy EfficientNet-B0) | 0.5680 |
| Loss/sampling: focal + WeightedRandomSampler (EfficientNet-B3) | 0.5769 |

## 8. Methodology evolution

The earlier course report committed to EfficientNet-B0 transfer learning
as the modern deep-learning baseline. The final project deepened this to
EfficientNet-B3 with full fine-tuning and added five further CNN
architectures (AlexNet, VGG16-BN, ResNet50, DenseNet121, Swin-Tiny) to
satisfy the AlexNet + VGG + ResNet comparative-analysis requirement.
The classical-ML pipeline (HOG + HSV histogram + GLCM + PCA + RBF-SVM)
was retained unchanged from the earlier commitment.

## 9. Where the artefacts live

```
MyDrive/melanoma/
├── kaggle.json
├── data/
│   ├── X_all.npy                 (1.6 GB uint8, (10015, 448, 448, 3))
│   ├── y_all.npy
│   ├── ids_all.npy               (object dtype; allow_pickle=True)
│   ├── lesion_ids_all.npy        (for lesion-grouped split)
│   ├── seg_fallback_all.npy      (1 if Otsu rejected, 0 otherwise)
│   ├── idx_train.npy             (7008 indices)
│   ├── idx_val.npy               (1503 indices)
│   └── idx_test.npy              (1504 indices)
├── results/                       CSVs, JSONs, PNGs — one set per method
├── checkpoints/                   resnet50_best.pt, etc.
└── paper/                         tables and figures
```
