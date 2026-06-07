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

We benchmark **ten methods** on the HAM10000 dermoscopic image dataset
for the binary classification task *melanoma vs. all other lesions*
(class ratio ≈ 1:8). The benchmark covers:

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
and validation-tuned decision thresholds. Every method evaluates on the
same held-out test set under a strict lesion-grouped stratified split
(70 / 15 / 15) so that no lesion contributes images to more than one
partition.

The full report is in [`paper/melanoma_paper.pdf`](paper/melanoma_paper.pdf).

## 2. Headline results (lesion-grouped test set, natural 1:8 imbalance)

| Method | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic regression baseline | 0.7957 | 0.3132 | 0.7337 |
| Classical ML (HOG + Color + GLCM + SVM) | 0.8397 | 0.4356 | 0.8350 |
| AlexNet | 0.8802 | 0.5135 | 0.8925 |
| VGG16-BN | 0.9162 | 0.6228 | 0.9190 |
| ResNet50 | 0.9255 | 0.6836 | 0.9470 |
| EfficientNet-B3 | 0.8975 | 0.5769 | 0.9189 |
| DenseNet121 | 0.9208 | 0.6149 | 0.9369 |
| Swin-Tiny | 0.9168 | 0.6575 | 0.9400 |
| **Soft-vote ensemble (6 CNNs)** | **0.9328** | **0.6967** | **0.9540** |
| Hybrid fusion (Method 10) | 0.9261 | 0.6647 | 0.9218 |

## 3. Repository structure

```
.
├── README.md                          (this file)
├── requirements.txt                   pip dependencies
├── config.py                          all paths and hyperparameters
├── experiments_log.md                 hardware, seed, hyperparameter list
├── paper/
│   └── melanoma_paper.pdf             IEEE-formatted final report
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
