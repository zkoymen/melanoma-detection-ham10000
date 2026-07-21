# Experiments Log

This document records the hardware, software stack, random seeds, and
hyperparameters used to produce every result in the paper. It is the
"Experiment Logs" deliverable from the final-project brief (hardware,
seed, hyperparameter list).

The live numerical results are in `results_csv/` in this repository
(`comparison_table.csv`, `ablation_table.csv`, `literature_comparison.csv`,
plus per-method `*_metrics.json`) and, at run time, in
`MyDrive/melanoma/results/`. The aggregation notebook
(`notebooks/10_aggregation.ipynb`) and the hybrid-fusion notebook
(`notebooks/11_hybrid_fusion.ipynb`) read those into the publication tables.

All numbers below are the balanced, cross-dataset, source-decorrelated
evaluation (HAM10000 + ISIC 2019; test set 1,698 images, 849 melanoma /
849 non-melanoma).

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

The course brief stated that the project could be completed on free
Colab. We deliberately used the paid Pro+ tier with A100 access so the
six CNNs could be **fully fine-tuned** (head + entire backbone) rather
than last-layer transfer-learned, and so 8-way test-time augmentation
could be applied to every test image without runtime pressure. Total CNN
training time (all six) is approximately **3.5 hours of A100 compute**.

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
| SHAP (interpretability, notebook 11) | ≥0.44 |
| matplotlib | 3.7 |
| pandas | 2.x |
| numpy | 1.26 |

Colab ships PyTorch and torchvision pre-installed; the notebook preambles
`pip install` only the extra dependencies (`timm`, `imbalanced-learn`,
`xgboost`, `shap`).

## 3. Reproducibility

| Item | Value |
|---|---|
| Global seed | **42** |
| numpy, torch, random, torch.cuda — all seeded | yes |
| Split | lesion-grouped 70/15/15 (Tschandl / Cassidy 2022) |
| Split constraint | grouped by `lesion_id`; validation and test exactly class-balanced |
| Split indices | saved as `idx_*_bal.npy` to Drive; every method evaluates on the same test index set |
| Source decorrelation | both classes drawn from HAM10000 + ISIC 2019 with the same per-class source mixture (source ⟂ label) |
| Train-time augmentation seeding | per-epoch via DataLoader workers; deterministic with the seed |

## 4. Dataset

| Item | Value |
|---|---|
| Sources | HAM10000 (`kmader/skin-cancer-mnist-ham10000`) + ISIC 2019 (`andrewmvd/isic-2019`) |
| Binarisation | `y = 1 if dx == "mel" else 0` |
| Corpus | balanced ~11,270 images (1:1) after size-matching the two classes |
| Stored array shape | `448×448×3` uint8 RGB (per image) |
| Preprocessing | DullRazor hair removal → Otsu segmentation in LAB-L → largest CC → 15% margin bbox crop → resize 448×448 |
| Segmentation fallback rate | ~3.5% of images (Otsu rejected → centred square crop) |
| Train split | 7,856 images (before SMOTE: non-mel 3,937 / mel 3,919) |
| Test split | **1,698 images (849 mel / 849 non-mel)** — exactly balanced |
| Validation split | balanced, ~15% of the corpus |

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
| Mixup α | 0.2 (30% of stage-2 batches) |
| CutMix α | 0.0 (intentionally disabled — see paper §III) |
| RandAugment | disabled (color ops would distort the ABCD signal) |
| RandomResizedCrop scale | (0.85, 1.00) |
| Rotation range | ±30° |
| ColorJitter — brightness / contrast | 0.25 |
| ColorJitter — saturation | 0.20 |
| ColorJitter — hue | 0.05 |
| RandomErasing probability | 0.20 |
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

### Classical / baseline / hybrid hyperparameters

| Method | Key values |
|---|---|
| Method 1 — Logistic Regression baseline | input 64×64 raw pixels; `class_weight='balanced'`; `max_iter=1000`; L-BFGS |
| Method 2 — Classical ML SVM | input 128×128; HOG (16×16 cells, 2×2 blocks, 9 bins); HSV histogram (8 bins/ch); GLCM d=(1,), angles (0,45,90,135°); PCA(200); RBF-SVM `class_weight='balanced'` |
| Method 10 — MelFidAI (hybrid fusion) | handcrafted (2,292-D) + ABCD (13-D) + 6-CNN penultimate (13,568-D) = 15,873-D → StandardScaler(no-center) → TruncatedSVD(300, expl. var. ≈0.634) → SMOTE-Tomek (train only, {0:3937,1:3919}→{0:3844,1:3844}) → MLP [256-128] and XGBoost (500 trees, depth 6, lr 0.05); winner by validation F1 (XGBoost, threshold 0.365) |

## 6. Per-method results (balanced cross-dataset test set, 1,698 images)

From `results_csv/comparison_table.csv`. All deep-learning rows use 8-way TTA
at inference and a validation-tuned decision threshold.

| Method | Acc | Prec | Rec | F1 | ROC-AUC | Inf (ms/img) |
|---|---|---|---|---|---|---|
| Logistic regression (raw 64×64) | 0.6084 | 0.6274 | 0.5336 | 0.5767 | 0.6550 | 0.05 |
| Classical ML SVM | 0.6938 | 0.6915 | 0.6996 | 0.6956 | 0.7792 | 2.89 |
| AlexNet | 0.7538 | 0.7149 | 0.8445 | 0.7743 | 0.8544 | 0.76 |
| VGG16-BN | 0.7850 | 0.7415 | 0.8751 | 0.8028 | 0.8867 | 5.53 |
| ResNet50 | 0.7803 | 0.7200 | **0.9176** | 0.8068 | 0.8915 | 5.78 |
| EfficientNet-B3 | 0.7574 | 0.7010 | 0.8975 | 0.7872 | 0.8720 | 22.65 |
| DenseNet121 | 0.8009 | 0.7589 | 0.8822 | 0.8159 | 0.9037 | 7.57 |
| Swin-Tiny | 0.8351 | 0.8244 | 0.8516 | 0.8378 | **0.9198** | 7.64 |
| Soft-vote ensemble (6 CNNs) | 0.8074 | 0.7589 | 0.9011 | 0.8239 | 0.9160 | ~49.9† |
| **MelFidAI (hybrid fusion, M10)** | 0.8404 | 0.8262 | 0.8622 | **0.8438** | 0.9172 | ~49.9† |

† Both the soft-vote ensemble and MelFidAI require a forward pass through all
six CNNs, so their end-to-end latency is dominated by the six upstream feature
extractors (Σ ≈ 49.9 ms/img); the soft-vote aggregation and the shallow fusion
classifier each add <0.01 ms on precomputed features.

Ensemble confusion matrix (balanced test): TP=765, FP=243, FN=84, TN=606
(recall 0.9011).

## 7. Ablation results (no extra training)

From `results_csv/ablation_table.csv`:

| Configuration | F1 |
|---|---|
| Decision threshold = 0.5 (mean across 6 CNNs) | 0.8025 |
| Decision threshold tuned on val (mean across 6 CNNs) | 0.8041 |
| TTA OFF (single forward, mean across 6 CNNs) | 0.8024 |
| TTA ON (8-way average, mean across 6 CNNs) | 0.8041 |
| Best single CNN (Swin-Tiny) | 0.8378 |
| Soft-vote ensemble (6 CNNs) | 0.8239 |
| MelFidAI (handcrafted + ABCD + deep) | **0.8438** |

On a balanced test set the threshold-tuning and TTA gains are small (the
default operating point is already near-optimal), unlike the large gains seen
under a heavily imbalanced test.

Additional analyses produced by `notebooks/11_hybrid_fusion.ipynb` (no CNN
retraining): a feature-block ablation of MelFidAI (deep vs. handcrafted vs.
ABCD), a PCA explained-variance sensitivity sweep, a McNemar test of
MelFidAI vs. Swin-Tiny with bootstrap 95% confidence intervals, and a
Tree-SHAP interpretation of the fusion booster. See `paper/HOW_TO_FINALIZE.md`.

## 8. Where the artefacts live

```
results_csv/                       committed CSV + per-method JSON (this repo)
MyDrive/melanoma/
├── data/                          X_combined.npy, idx_*_bal.npy, lesion ids
├── results/                       CSVs, JSONs, PNGs — one set per method
│   └── paper_generated/           .tex / .png produced by notebook 11 for the paper
├── checkpoints/                   <arch>_best.pt
└── ...
```
