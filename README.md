# Melanoma Detection on HAM10000 — Binary Classification

Final project for the **Digital Image Processing (COM4504 / BLM4504)**
course, Ankara University, Faculty of Engineering, Department of
Computer Engineering, Spring 2025–2026.

**Authors**

- Adem Efe Devrez (21290598)
- Zeynep Köymen (22290286)
- Ezra Bolat (22290037)

**Instructor:** Asst. Prof. Dr. Feyza Toktaş

---

## 1. What this project does

We benchmark **ten methods** for the binary classification task
*melanoma vs. all other lesions* under a deliberately strict evaluation
protocol that is **leak-free, balanced, cross-dataset and
source-decorrelated**: melanoma and non-melanoma images are drawn from
**both HAM10000 and ISIC 2019** with the same per-class source mixture (so
the dataset source is statistically independent of the label), lesions are
split 70/15/15 by `lesion_id`, and the validation/test partitions are
exactly balanced (test set: **1,698 images, 849/849**). The benchmark
covers:

1. Logistic regression on raw 64×64 pixels (simple baseline).
2. Classical machine learning — HOG + HSV colour histogram + GLCM
   features → PCA → RBF-SVM.
3. AlexNet (2012, fully fine-tuned).
4. VGG16-BN (2014, fully fine-tuned).
5. ResNet50 (2015, fully fine-tuned).
6. EfficientNet-B3 (2019, fully fine-tuned).
7. DenseNet121 (2017, fully fine-tuned).
8. Swin-Tiny (2021, fully fine-tuned).
9. Soft-vote ensemble of methods 3–8.
10. Hybrid handcrafted–deep feature fusion (HOG + colour + GLCM + ABCD
    clinical descriptors + the penultimate-layer features of all six
    CNNs, balanced with SMOTE-Tomek, classified by XGBoost / MLP).

All deep models share the same training recipe: focal loss with
class-balanced α, weighted random sampling, conservative medical-grade
augmentation, AdamW with linear warm-up and cosine annealing,
exponential moving average of weights, eight-way test-time augmentation,
and validation-tuned decision thresholds. The split is computed once and
augmentation is applied to the training partition only, *after* the split,
so no lesion — and no augmented copy of a test image — leaks into training.

## 2. Headline results (balanced HAM10000 + ISIC 2019 test set, 1,698 images, 849/849)

All ten methods are trained and evaluated on the identical balanced split.

| Method | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic regression (raw 64×64) | 0.6084 | 0.6274 | 0.5336 | 0.5767 | 0.6550 |
| Classical ML (HOG + Color + GLCM + PCA + SVM) | 0.6938 | 0.6915 | 0.6996 | 0.6956 | 0.7792 |
| AlexNet | 0.7538 | 0.7149 | 0.8445 | 0.7743 | 0.8544 |
| VGG16-BN | 0.7850 | 0.7415 | 0.8751 | 0.8028 | 0.8867 |
| ResNet50 | 0.7803 | 0.7200 | 0.9176 | 0.8068 | 0.8915 |
| EfficientNet-B3 | 0.7574 | 0.7010 | 0.8975 | 0.7872 | 0.8720 |
| DenseNet121 | 0.8009 | 0.7589 | 0.8822 | 0.8159 | 0.9037 |
| Swin-Tiny (best single CNN) | 0.8351 | 0.8244 | 0.8516 | 0.8378 | **0.9198** |
| Soft-vote ensemble (6 CNNs) | 0.8074 | 0.7589 | **0.9011** | 0.8239 | 0.9160 |
| **Hybrid fusion (Method 10)** | 0.8404 | 0.8262 | 0.8622 | **0.8438** | 0.9172 |

The **hybrid fusion** classifier attains the best F1 (0.8438), **Swin-Tiny**
is the strongest single architecture (F1 0.8378, AUC 0.9198), and the
**soft-vote ensemble** attains the highest recall (0.9011) — the operating
characteristic preferred for clinical screening. These honest mid-0.80 F1
values are measured on a leak-free, balanced, source-decorrelated test; they
are a deliberately more conservative estimate than the 0.95–0.99 figures
common on HAM10000, which are largely inflated by image-level splits and
augment-before-split leakage.

## 3. Repository structure

```
.
├── README.md                          (this file)
├── requirements.txt                   pip dependencies
├── config.py                          all paths and hyperparameters
├── experiments_log.md                 hardware, seed, hyperparameter list
├── notebooks/                         the twelve Jupyter notebooks
│   ├── 00_data_setup.ipynb
│   ├── 01_baseline_logistic.ipynb
│   ├── 02_classical_ml_svm.ipynb
│   ├── 03_deep_learning_effnet.ipynb  EfficientNet-B0 transfer learning
│   ├── 04_alexnet.ipynb
│   ├── 05_vgg16_bn.ipynb
│   ├── 06_resnet50.ipynb
│   ├── 07_efficientnet_b3.ipynb
│   ├── 08_densenet121.ipynb
│   ├── 09_swin_tiny.ipynb
│   ├── 10_aggregation.ipynb           ensemble + headline table + ablations
│   └── 11_hybrid_fusion.ipynb         Method 10
├── src/                               reusable Python modules
│   ├── data.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── abcd.py
│   ├── features.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   └── gradcam.py
├── scripts/
│   └── build_notebooks.py             rebuilds every .ipynb from one source
└── results_csv/                       per-method metrics JSON + comparison tables
```

The raw HAM10000 images, intermediate `.npy` arrays and trained model
checkpoints are not committed (they are too large for the zip) and live
on Google Drive under `MyDrive/melanoma/`. The notebooks fetch them
from there automatically when run on Colab.

## 4. How to reproduce on Google Colab

Training the six CNNs requires a GPU. We used Google Colab with an
NVIDIA A100 40 GB. A T4 also works but is roughly three times slower
and may require the batch sizes in `config.py` to be halved.

1. **One-time setup.** Create a Drive folder `MyDrive/melanoma/` and
   place a `kaggle.json` Kaggle API token in it.
2. **Notebook 00 (`00_data_setup.ipynb`).** Open in Colab on a standard
   CPU runtime. The first cell mounts Drive and prepares the project.
   The notebook downloads HAM10000 from Kaggle, runs DullRazor hair
   removal, Otsu lesion segmentation in the LAB colour space, a 15 %
   margin bounding-box crop, and resizes everything to 448 × 448. It
   then computes a stratified lesion-grouped 70 / 15 / 15 split and
   saves the indices to `MyDrive/melanoma/data/`.
3. **Notebooks 01 → 11.** Run in order. Notebooks 01, 02, 10 and 11 are
   CPU-only; notebooks 04 to 09 need a GPU. The aggregation notebook
   (`10_aggregation.ipynb`) regenerates the headline table, ablation
   table, ROC overlay, error analysis grid and per-method curves.

All outputs land in `MyDrive/melanoma/results/`. Every CNN notebook is
disconnect-proof: the best validation-F1 checkpoint is saved to Drive
every time it improves, and a recovery cell at the bottom of each
notebook reloads the checkpoint and reruns threshold tuning and TTA
evaluation without retraining.

## 5. Reproducibility

- Global random seed `42` for `numpy`, `torch`, `random` and
  `torch.cuda`.
- The lesion-grouped split is computed once and persisted as integer
  index arrays, so every method evaluates on the same test set.
- All hyperparameters live in `config.py`. Hardware, software
  versions and per-architecture training times are listed in
  `experiments_log.md`.
- Per-method outputs (metrics JSON, predictions CSV, comparison tables)
  are committed under `results_csv/` so the headline numbers can be
  inspected without re-running anything.

## 6. Local smoke test (no GPU required)

To verify that the Python modules import cleanly outside Colab:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -c "from src import data, preprocessing, segmentation, abcd, features, models, training, evaluation, gradcam; print('ok')"
```

Local CPU runs are too slow for real training; use Colab for that.

## 7. Configuration

`config.py` is the single source of truth for paths and hyperparameters.
The two path constants can be overridden by environment variables when
running outside Colab:

```powershell
$env:MELANOMA_DRIVE_ROOT = "D:\melanoma"
$env:MELANOMA_LOCAL_SCRATCH = "D:\melanoma\scratch"
```

Inside Colab the defaults (`/content/drive/MyDrive/melanoma`,
`/content/ham10000`) work as-is.

## 8. Dataset citation

> P. Tschandl, C. Rosendahl and H. Kittler,
> *The HAM10000 dataset, a large collection of multi-source dermatoscopic
> images of common pigmented skin lesions*,
> Scientific Data, vol. 5, Art. no. 180161, 2018.

Kaggle slug: `kmader/skin-cancer-mnist-ham10000`.
