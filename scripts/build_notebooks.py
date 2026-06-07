"""
Generate every .ipynb in `notebooks/` from plain-Python definitions.

Why:  hand-editing notebook JSON is brittle (escaping, trailing newlines).
      Defining cells as Python strings keeps diffs reviewable on GitHub.

Usage:
    python scripts/build_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _split(src)}


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split(src),
    }


def _split(src: str) -> list[str]:
    """Split into list-of-lines with trailing \\n (Jupyter convention)."""
    lines = src.strip("\n").splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] = lines[-1]
    return lines or [""]


def write_nb(name: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = NB_DIR / name
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {out.relative_to(ROOT)}")


# ----------------------------------------------------------------------
# Reusable preamble used in every Colab notebook
# ----------------------------------------------------------------------
COLAB_PREAMBLE = '''
# --- Colab setup: ensure the project is on sys.path, mount Drive, load config ---
import os, sys, subprocess
from pathlib import Path

# Either the project is already on disk (uploaded zip / mounted Drive) or we
# clone it from GitHub. We never destroy local changes.
REPO_URL = "https://github.com/zkoymen/melanoma-detection-ham10000.git"
CANDIDATE_PATHS = [
    Path.cwd(),
    Path("/content/melanoma-detection-ham10000"),
    Path("/content/drive/MyDrive/melanoma-detection-ham10000"),
]

project_root = None
for p in CANDIDATE_PATHS:
    if (p / "src").exists() and (p / "config.py").exists():
        project_root = p
        break

if project_root is None:
    project_root = Path("/content/melanoma-detection-ham10000")
    subprocess.run(["git", "clone", REPO_URL, str(project_root)], check=True)

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Mount Drive (silently re-uses an existing mount on re-run)
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
except (ImportError, ModuleNotFoundError):
    pass

import config
config.ensure_drive_dirs()
print("Project root:", project_root)
print("Drive root  :", config.DRIVE_ROOT)
print("Data dir    :", config.DATA_DIR)
print("Results dir :", config.RESULTS_DIR)
'''.strip()


SEED_BLOCK = '''
import random, numpy as np, torch
random.seed(config.SEED); np.random.seed(config.SEED); torch.manual_seed(config.SEED)
torch.cuda.manual_seed_all(config.SEED)
'''.strip()


# ======================================================================
# 00_data_setup.ipynb
# ======================================================================
def nb_data_setup():
    cells = [
        md("""
# 00 — Data Setup (segmentation + 448 + lesion-grouped split)

**Inputs:** `MyDrive/melanoma/kaggle.json` (your Kaggle API token).
**Outputs (in `MyDrive/melanoma/data/`):**
- `X_all.npy` (N, 448, 448, 3) uint8 — lesion-cropped, hair-removed RGB at 448×448
- `y_all.npy` (N,) int64 — 1 = melanoma, 0 = otherwise
- `ids_all.npy` (N,) — HAM10000 `image_id` strings, same order as X
- `lesion_ids_all.npy` (N,) — HAM10000 `lesion_id` strings (used by the split)
- `seg_fallback_all.npy` (N,) bool — True where Otsu seg failed and a centre
  crop was substituted
- `idx_train.npy`, `idx_val.npy`, `idx_test.npy` — lesion-grouped stratified
  70/15/15 split (no image of a given lesion appears in more than one partition)

Pipeline per image (one-shot, run once on Colab, persisted to Drive):
1. DullRazor hair removal (morphological black-hat + Telea inpaint).
2. Otsu threshold on LAB-L (inverted) + morphological close/open + largest
   connected component. Sanity gates reject suspicious masks; on rejection a
   centred 80%-side fallback crop is used.
3. Crop to lesion bounding box with 15% margin.
4. Resize to 448×448.
5. BGR -> RGB uint8.

Re-runnable: the cell below detects a wrong-shape X_all.npy and rebuilds it.
The first run takes about 15–25 minutes (10,015 images, CPU-bound).
"""),
        code(COLAB_PREAMBLE),
        code("!pip install --quiet kaggle tqdm opencv-python-headless scikit-image timm 2>&1 | tail -n 1"),
        code("""
# --- Configure Kaggle CLI from kaggle.json on Drive ---
import os, shutil
os.makedirs("/root/.kaggle", exist_ok=True)
shutil.copy(str(config.KAGGLE_JSON_PATH), "/root/.kaggle/kaggle.json")
os.chmod("/root/.kaggle/kaggle.json", 0o600)
print("Kaggle credentials installed.")
"""),
        code("""
# --- Download + unzip HAM10000 to /content (fast SSD) ---
import os, subprocess
from pathlib import Path

scratch = Path(config.LOCAL_SCRATCH); scratch.mkdir(parents=True, exist_ok=True)
zip_path = scratch / "ham.zip"

if not zip_path.exists():
    subprocess.run(["kaggle", "datasets", "download",
                    "-d", config.KAGGLE_DATASET, "-p", str(scratch)],
                   check=True)
    # Kaggle drops a zip whose filename matches the dataset slug
    src = next(scratch.glob("*.zip"))
    src.rename(zip_path)
    subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(scratch)], check=True)

print("Files in scratch:", sorted(p.name for p in scratch.iterdir())[:8], "...")
"""),
        code("""
# --- Read metadata, build binary labels, expose lesion_id ---
import pandas as pd
meta = pd.read_csv(scratch / "HAM10000_metadata.csv")
meta["y"] = (meta["dx"] == config.POSITIVE_CLASS).astype("int64")
print("Total:", len(meta))
print(meta["dx"].value_counts())
print("Binary balance:", meta["y"].value_counts().to_dict())
n_unique_lesions = meta["lesion_id"].nunique()
print(f"Unique lesions: {n_unique_lesions}  (multi-image lesions: {len(meta) - n_unique_lesions})")
"""),
        code("""
# --- Index image_id -> filesystem path (images live in two part folders) ---
img_root_candidates = [
    scratch / "ham10000_images_part_1", scratch / "ham10000_images_part_2",
    scratch / "HAM10000_images_part_1", scratch / "HAM10000_images_part_2",
]
id_to_path = {}
for d in img_root_candidates:
    if d.exists():
        for p in d.glob("*.jpg"):
            id_to_path[p.stem] = p
print("Found", len(id_to_path), "image files.")
assert len(id_to_path) >= len(meta), "Some image_ids missing on disk!"
"""),
        code("""
# --- Build X_all.npy (hair removal + Otsu segmentation + crop + resize) ---
# Re-builds when the on-disk shape doesn't match config.IMG_SIZE.
import cv2, numpy as np
from tqdm import tqdm
from src.preprocessing import preprocess_for_storage

X_path           = config.DATA_DIR / "X_all.npy"
y_path           = config.DATA_DIR / "y_all.npy"
ids_path         = config.DATA_DIR / "ids_all.npy"
lesion_ids_path  = config.DATA_DIR / "lesion_ids_all.npy"
seg_fb_path      = config.DATA_DIR / "seg_fallback_all.npy"

N = len(meta)
expected_shape = (N, config.IMG_SIZE, config.IMG_SIZE, 3)

needs_rebuild = True
if (X_path.exists() and y_path.exists() and ids_path.exists()
        and lesion_ids_path.exists() and seg_fb_path.exists()):
    try:
        X_check = np.load(X_path, mmap_mode="r")
        if X_check.shape == expected_shape:
            needs_rebuild = False
            print(f"Drive arrays already match {expected_shape} — skipping heavy loop.")
    except Exception as exc:
        print(f"Could not inspect existing X_all.npy ({exc}); rebuilding.")

if not needs_rebuild:
    X = np.load(X_path)
    y = np.load(y_path)
    ids = np.load(ids_path, allow_pickle=True)
    lesion_ids = np.load(lesion_ids_path, allow_pickle=True)
    seg_fallback = np.load(seg_fb_path)
else:
    X = np.empty(expected_shape, dtype=np.uint8)
    y = meta["y"].to_numpy()
    ids = meta["image_id"].to_numpy()
    lesion_ids = meta["lesion_id"].to_numpy()
    seg_fallback = np.zeros(N, dtype=bool)
    for i, image_id in enumerate(tqdm(ids, desc="hair-removal + seg + crop + resize")):
        img_bgr = cv2.imread(str(id_to_path[image_id]))
        X[i], used_fb = preprocess_for_storage(
            img_bgr,
            size=config.IMG_SIZE,
            do_hair_removal=True,
            do_segmentation=True,
            seg_margin_frac=config.SEG_MARGIN_FRAC,
            seg_border_frac=config.SEG_BORDER_FRAC,
            seg_min_area_frac=config.SEG_MIN_AREA_FRAC,
            seg_max_area_frac=config.SEG_MAX_AREA_FRAC,
            seg_fallback_frac=config.SEG_FALLBACK_FRAC,
        )
        seg_fallback[i] = used_fb
    np.save(X_path, X)
    np.save(y_path, y)
    np.save(ids_path, ids)
    np.save(lesion_ids_path, lesion_ids)
    np.save(seg_fb_path, seg_fallback)

n_fb = int(seg_fallback.sum())
print(f"X: {X.shape} {X.dtype}   y: {y.shape}   ids: {ids.shape}")
print(f"Otsu segmentation success: {N - n_fb} / {N} "
      f"({100.0 * (N - n_fb) / N:.2f}%) — fallback used in {n_fb} images.")
"""),
        code("""
# --- Lesion-grouped stratified 70/15/15 split ---
# HAM10000 has multiple images per lesion (~7,470 unique lesions over 10,015
# images). A naive split stratified on `y` only would place different images
# of the same lesion into different partitions and leak information. We split
# by *lesion*, stratified on the lesion's binary label, then map lesions back
# to image indices.
import numpy as np
from sklearn.model_selection import train_test_split

# Per-lesion binary label (all images of a lesion share the same dx in HAM10000).
lesion_y = meta.groupby("lesion_id")["y"].first()
lesion_ids_unique = lesion_y.index.to_numpy()
lesion_labels = lesion_y.to_numpy()

# Stratified hold-out test (lesion-grouped)
trainval_lesions, test_lesions = train_test_split(
    lesion_ids_unique, test_size=config.TEST_FRAC,
    stratify=lesion_labels, random_state=config.SEED)
trainval_y = lesion_y.loc[trainval_lesions].to_numpy()

# Stratified val carve-out from the train+val pool (lesion-grouped)
val_rel = config.VAL_FRAC / (config.TRAIN_FRAC + config.VAL_FRAC)
train_lesions, val_lesions = train_test_split(
    trainval_lesions, test_size=val_rel,
    stratify=trainval_y, random_state=config.SEED)

# Map lesion partitions back to image-row indices
train_set = set(train_lesions); val_set = set(val_lesions); test_set = set(test_lesions)
img_lesions = meta["lesion_id"].to_numpy()
idx_train = np.where(np.isin(img_lesions, list(train_set)))[0]
idx_val   = np.where(np.isin(img_lesions, list(val_set)))[0]
idx_test  = np.where(np.isin(img_lesions, list(test_set)))[0]

# Defensive overlap check (must all be empty)
assert len(np.intersect1d(idx_train, idx_val))  == 0
assert len(np.intersect1d(idx_train, idx_test)) == 0
assert len(np.intersect1d(idx_val,   idx_test)) == 0
assert set(img_lesions[idx_train]).isdisjoint(set(img_lesions[idx_test]))
assert set(img_lesions[idx_train]).isdisjoint(set(img_lesions[idx_val]))
assert set(img_lesions[idx_val]).isdisjoint(set(img_lesions[idx_test]))

np.save(config.DATA_DIR / "idx_train.npy", idx_train)
np.save(config.DATA_DIR / "idx_val.npy",   idx_val)
np.save(config.DATA_DIR / "idx_test.npy",  idx_test)

def dist(label_arr):
    u, c = np.unique(label_arr, return_counts=True)
    return dict(zip(u.tolist(), c.tolist()))

print(f"train: {len(idx_train):>5} images  {dist(y[idx_train])}  "
      f"({len(train_lesions)} lesions)")
print(f"val  : {len(idx_val):>5} images  {dist(y[idx_val])}  "
      f"({len(val_lesions)} lesions)")
print(f"test : {len(idx_test):>5} images  {dist(y[idx_test])}  "
      f"({len(test_lesions)} lesions)")
print("Lesion-grouped split OK — no lesion appears in more than one partition.")
"""),
        code("""
# --- Final summary ---
import os
def mb(p): return os.path.getsize(p) / 1e6
total = sum(mb(p) for p in config.DATA_DIR.glob("*.npy"))
print(f"Total Drive footprint: {total:.1f} MB")
for p in sorted(config.DATA_DIR.glob('*.npy')):
    print(f"  {p.name:20s} {mb(p):7.1f} MB")
print("\\nData setup complete. You can now run notebooks 01-04 in any order.")
"""),
    ]
    write_nb("00_data_setup.ipynb", cells)


# ======================================================================
# 01_baseline_logistic.ipynb
# ======================================================================
def nb_baseline():
    cells = [
        md("""
# 01 — Baseline: Logistic Regression on raw 64×64 pixels

The dumbest possible classifier. Sets the floor for the comparison table.
Reads the same arrays produced by `00_data_setup.ipynb`.
"""),
        code(COLAB_PREAMBLE),
        code(SEED_BLOCK),
        code("""
# --- Load arrays ---
import numpy as np, cv2
from src.data import load_arrays
X, y, ids, idx_train, idx_val, idx_test = load_arrays(config.DATA_DIR)
print("X:", X.shape, "  test size:", len(idx_test))
"""),
        code("""
# --- Resize 224 -> 64 and flatten ---
def to_baseline_vec(X224):
    out = np.empty((len(X224), config.BASELINE_IMG_SIZE * config.BASELINE_IMG_SIZE * 3),
                   dtype=np.float32)
    for i, im in enumerate(X224):
        small = cv2.resize(im, (config.BASELINE_IMG_SIZE, config.BASELINE_IMG_SIZE),
                           interpolation=cv2.INTER_AREA)
        out[i] = small.reshape(-1).astype(np.float32) / 255.0
    return out

X_train = to_baseline_vec(X[idx_train])
X_val   = to_baseline_vec(X[idx_val])
X_test  = to_baseline_vec(X[idx_test])
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]
print("X_train:", X_train.shape)
"""),
        code("""
# --- Train Logistic Regression with class_weight='balanced' ---
import time
from sklearn.linear_model import LogisticRegression
hp = dict(class_weight="balanced", max_iter=1000, random_state=config.SEED, solver="lbfgs")
clf = LogisticRegression(**hp)
t0 = time.time(); clf.fit(X_train, y_train); train_time = time.time() - t0
print(f"Trained in {train_time:.1f}s")
"""),
        code("""
# --- Evaluate on test set ---
from src.evaluation import save_standard_outputs, time_inference

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

inf_ms = time_inference(lambda x: clf.predict(x), X_test[:200])

metrics = save_standard_outputs(
    method_name="baseline_logistic",
    results_dir=config.RESULTS_DIR,
    y_true=y_test, y_pred=y_pred, y_prob=y_prob,
    ids=ids[idx_test],
    hyperparameters=hp,
    train_time_sec=train_time,
    inference_time_per_image_ms=inf_ms,
)
{k: v for k, v in metrics.items() if k != "hyperparameters"}
"""),
    ]
    write_nb("01_baseline_logistic.ipynb", cells)


# ======================================================================
# 02_classical_ml_svm.ipynb
# ======================================================================
def nb_classical():
    cells = [
        md("""
# 02 — Classical ML: HOG + Color Histogram + GLCM → PCA → RBF SVM

HOG is computed on **128×128** images to keep the descriptor manageable.
PCA reduces ~3.5k → 200 dims before the SVM.
"""),
        code(COLAB_PREAMBLE),
        code(SEED_BLOCK),
        code("""
import numpy as np, cv2
from tqdm import tqdm  # plain tqdm: text-based bar, no ipywidgets — avoids the "metadata.widgets state missing" GitHub render bug
from src.data import load_arrays
from src.features import extract_all
X, y, ids, idx_train, idx_val, idx_test = load_arrays(config.DATA_DIR)
print("X:", X.shape)
"""),
        code("""
# --- Resize once to 128x128 (HOG is RAM-hungry on 224x224) ---
def to_hog_size(X224, size):
    out = np.empty((len(X224), size, size, 3), dtype=np.uint8)
    for i, im in enumerate(X224):
        out[i] = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    return out

X128 = to_hog_size(X, config.HOG_IMG_SIZE)
print("Resized:", X128.shape)
"""),
        code("""
# --- Extract concatenated descriptors ---
sample = extract_all(X128[0],
                     pixels_per_cell=config.HOG_PIXELS_PER_CELL,
                     cells_per_block=config.HOG_CELLS_PER_BLOCK,
                     color_bins=config.COLOR_HIST_BINS,
                     glcm_distances=config.GLCM_DISTANCES,
                     glcm_angles=config.GLCM_ANGLES)
print("Feature dim:", sample.shape[0])

F = np.empty((len(X128), sample.shape[0]), dtype=np.float32)
for i in tqdm(range(len(X128)), desc="features"):
    F[i] = extract_all(X128[i],
                       pixels_per_cell=config.HOG_PIXELS_PER_CELL,
                       cells_per_block=config.HOG_CELLS_PER_BLOCK,
                       color_bins=config.COLOR_HIST_BINS,
                       glcm_distances=config.GLCM_DISTANCES,
                       glcm_angles=config.GLCM_ANGLES)
print("F:", F.shape, F.dtype)
"""),
        code("""
# --- Standardize + PCA (fit on train only) ---
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler().fit(F[idx_train])
F_s = scaler.transform(F)

pca = PCA(n_components=config.PCA_COMPONENTS, random_state=config.SEED).fit(F_s[idx_train])
F_p = pca.transform(F_s)
print("PCA explained variance:", pca.explained_variance_ratio_.sum().round(3))
print("F_p:", F_p.shape)
"""),
        code("""
# --- Train RBF SVM with class_weight='balanced' ---
import time
from sklearn.svm import SVC
hp = dict(kernel="rbf", C=1.0, gamma="scale",
          class_weight="balanced", probability=True, random_state=config.SEED)
svm = SVC(**hp)
t0 = time.time(); svm.fit(F_p[idx_train], y[idx_train]); train_time = time.time() - t0
print(f"Trained in {train_time:.1f}s")
"""),
        code("""
# --- Evaluate ---
from src.evaluation import save_standard_outputs, time_inference

y_pred = svm.predict(F_p[idx_test])
y_prob = svm.predict_proba(F_p[idx_test])[:, 1]

inf_ms = time_inference(lambda x: svm.predict(x), F_p[idx_test][:200])

metrics = save_standard_outputs(
    method_name="classical_ml_svm",
    results_dir=config.RESULTS_DIR,
    y_true=y[idx_test], y_pred=y_pred, y_prob=y_prob,
    ids=ids[idx_test],
    hyperparameters={**hp, "feature_dim_raw": int(F.shape[1]), "pca_components": config.PCA_COMPONENTS},
    train_time_sec=train_time,
    inference_time_per_image_ms=inf_ms,
)
{k: v for k, v in metrics.items() if k != "hyperparameters"}
"""),
    ]
    write_nb("02_classical_ml_svm.ipynb", cells)


# ======================================================================
# 03_deep_learning_effnet.ipynb
# ======================================================================
def nb_effnet():
    cells = [
        md("""
# 03 — EfficientNet-B0 transfer learning + Grad-CAM

**Disconnect-proof:** the best-val-F1 weights are saved to Drive
(`MyDrive/melanoma/checkpoints/effnet_b0_best.pt`) the moment val F1
improves. If the runtime dies mid-training, just re-run the recovery
cell at the bottom — no retraining needed.

Two-stage training:
1. Freeze backbone, train classifier head only — lr=1e-3, 5 epochs.
2. Unfreeze last **2** EfficientNet blocks — lr=1e-4, up to 15 epochs,
   early stopping on val F1 (patience=5).

Class imbalance handled by **softened (sqrt-balanced)** weights —
full `balanced` weights overcorrect and tank precision.
"""),
        code(COLAB_PREAMBLE),
        code(SEED_BLOCK),
        code("""
# --- Build PyTorch Datasets / DataLoaders (data auto-cached to /content) ---
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.data import load_arrays, softened_class_weights

X, y, ids, idx_train, idx_val, idx_test = load_arrays(config.DATA_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, "  X:", X.shape)

class HAMDataset(Dataset):
    def __init__(self, X, y, indices, transform):
        self.X, self.y, self.idx, self.tf = X, y, indices, transform
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        k = self.idx[i]
        return self.tf(self.X[k]), int(self.y[k])

# Storage is at 448x448. EfficientNet-B0 was trained at 224 in the original
# timm recipe, so we downsample explicitly here.
LEGACY_INPUT = 224
train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((LEGACY_INPUT, LEGACY_INPUT), antialias=True),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(0.15, 0.15, 0.15, 0.07),
    transforms.RandomResizedCrop(LEGACY_INPUT, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((LEGACY_INPUT, LEGACY_INPUT), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])

bs = config.EFFNET_BATCH_SIZE
train_ld = DataLoader(HAMDataset(X, y, idx_train, train_tf), batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True)
val_ld   = DataLoader(HAMDataset(X, y, idx_val,   eval_tf),  batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
test_ld  = DataLoader(HAMDataset(X, y, idx_test,  eval_tf),  batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
"""),
        code("""
# --- Build model ---
from src.models import build_efficientnet_b0, freeze_backbone, unfreeze_last_n_blocks, trainable_params
model = build_efficientnet_b0(num_classes=2, pretrained=True).to(device)
freeze_backbone(model)
print("Stage-1 trainable params:", trainable_params(model))
"""),
        code("""
# --- Softened (sqrt-balanced) class-weighted loss ---
import torch.nn as nn
cw = softened_class_weights(y[idx_train], power=config.EFFNET_CW_POWER)
weights = torch.tensor([cw[0], cw[1]], dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=weights)
print("Softened class weights:", cw)
"""),
        code("""
# --- Stage 1: head only ---
import time, torch
from src.training import train_loop, EpochLog

CHECKPOINT_PATH = config.CHECKPOINT_DIR / "effnet_b0_best.pt"
print("Best weights will be persisted to:", CHECKPOINT_PATH)

opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.EFFNET_HEAD_LR)
log = EpochLog()
t0 = time.time()
log = train_loop(model, train_ld, val_ld, criterion, opt, device,
                 epochs=config.EFFNET_HEAD_EPOCHS, log=log,
                 checkpoint_path=CHECKPOINT_PATH)
"""),
        code("""
# --- Stage 2: unfreeze last N blocks ---
unfreeze_last_n_blocks(model, n=config.EFFNET_UNFREEZE_LAST_N_BLOCKS)
print(f"Stage-2 unfrozen blocks: {config.EFFNET_UNFREEZE_LAST_N_BLOCKS}")
print(f"Stage-2 trainable params: {trainable_params(model)}")

opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config.EFFNET_FT_LR)
log = train_loop(model, train_ld, val_ld, criterion, opt, device,
                 epochs=config.EFFNET_FT_EPOCHS,
                 early_stop_patience=config.EFFNET_EARLY_STOP_PATIENCE, log=log,
                 checkpoint_path=CHECKPOINT_PATH)
train_time = time.time() - t0
print(f"Total training time: {train_time:.1f}s. Best weights at {CHECKPOINT_PATH}")
"""),
        code("""
# --- Plot loss / F1 curves ---
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(log.train_loss, label="train"); ax[0].plot(log.val_loss, label="val")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(log.val_f1); ax[1].set_title("Val F1"); ax[1].set_xlabel("epoch")
fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "deep_learning_effnet_curves.png", dpi=120)
plt.show()
"""),
        code("""
# --- Threshold optimization on val: find the F1-maximizing decision threshold ---
import numpy as np
from sklearn.metrics import f1_score
from src.training import predict

y_val_true, _, y_val_prob = predict(model, val_ld, device)
ths = np.linspace(0.05, 0.95, 181)
f1s = [f1_score(y_val_true, (y_val_prob > t).astype(int), zero_division=0) for t in ths]
best_threshold = float(ths[int(np.argmax(f1s))])
print(f"Best val threshold: {best_threshold:.3f}  ->  val F1 = {max(f1s):.4f}")
"""),
        code("""
# --- Evaluate on test set with the optimized threshold ---
import time
from src.evaluation import save_standard_outputs

t0 = time.time()
y_true, _, y_prob = predict(model, test_ld, device)
inf_ms = (time.time() - t0) * 1000.0 / len(y_true)
y_pred = (y_prob > best_threshold).astype(int)

hp = dict(
    arch="efficientnet_b0_timm", input_size=config.IMG_SIZE, batch_size=bs,
    head_lr=config.EFFNET_HEAD_LR, head_epochs=config.EFFNET_HEAD_EPOCHS,
    ft_lr=config.EFFNET_FT_LR,     ft_epochs=config.EFFNET_FT_EPOCHS,
    early_stop_patience=config.EFFNET_EARLY_STOP_PATIENCE,
    unfreeze_last_n_blocks=config.EFFNET_UNFREEZE_LAST_N_BLOCKS,
    class_weight=f"softened (power={config.EFFNET_CW_POWER})",
    augmentation="hflip+vflip+rot30+colorjitter+resizedcrop", hair_removal=True,
    decision_threshold=best_threshold,
    threshold_selection="argmax F1 on validation set",
)
metrics = save_standard_outputs(
    method_name="deep_learning_effnet",
    results_dir=config.RESULTS_DIR,
    y_true=y_true, y_pred=y_pred, y_prob=y_prob,
    ids=ids[idx_test],
    hyperparameters=hp,
    train_time_sec=train_time,
    inference_time_per_image_ms=inf_ms,
)
{k: v for k, v in metrics.items() if k != "hyperparameters"}
"""),
        code("""
# --- Grad-CAM grid: 4 melanoma + 4 non-melanoma test images ---
import numpy as np, torch, matplotlib.pyplot as plt
from src.gradcam import GradCAM, overlay_heatmap

model.eval()
cam = GradCAM(model, model.blocks[-1])

mel_idx    = idx_test[y[idx_test] == 1][:4]
nonmel_idx = idx_test[y[idx_test] == 0][:4]
chosen = list(mel_idx) + list(nonmel_idx)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, k in zip(axes.flat, chosen):
    img = X[k]
    t = eval_tf(img).unsqueeze(0).to(device)
    heat = cam(t, class_idx=int(y[k]))
    ax.imshow(overlay_heatmap(img, heat))
    ax.set_title(f"id={ids[k]}  true={int(y[k])}")
    ax.axis("off")
cam.close()
fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "gradcam_grid.png", dpi=120)
plt.show()
"""),
        md("""
---
## Recovery cell (use only if runtime disconnected during training)

If the cell above showing curves never ran, but you saw at least one
`-> checkpoint saved` line during training, the best weights are in
`MyDrive/melanoma/checkpoints/effnet_b0_best.pt`. Run this single cell
on a fresh runtime (after re-running the preamble + dataset cells) to
finish evaluation without retraining.
"""),
        code("""
# --- Recovery: load checkpoint from Drive, evaluate, save standardized outputs ---
import torch, time, numpy as np
from sklearn.metrics import f1_score
from src.training import predict
from src.evaluation import save_standard_outputs

CHECKPOINT_PATH = config.CHECKPOINT_DIR / "effnet_b0_best.pt"
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

y_val_true, _, y_val_prob = predict(model, val_ld, device)
ths = np.linspace(0.05, 0.95, 181)
best_threshold = float(ths[int(np.argmax([f1_score(y_val_true,(y_val_prob>t).astype(int), zero_division=0) for t in ths]))])

t0 = time.time()
y_true, _, y_prob = predict(model, test_ld, device)
inf_ms = (time.time() - t0) * 1000.0 / len(y_true)
y_pred = (y_prob > best_threshold).astype(int)

metrics = save_standard_outputs(
    method_name="deep_learning_effnet",
    results_dir=config.RESULTS_DIR,
    y_true=y_true, y_pred=y_pred, y_prob=y_prob,
    ids=ids[idx_test],
    hyperparameters={"recovered_from_checkpoint": True, "decision_threshold": best_threshold},
    train_time_sec=0.0,
    inference_time_per_image_ms=inf_ms,
)
{k: v for k, v in metrics.items() if k != "hyperparameters"}
"""),
    ]
    write_nb("03_deep_learning_effnet.ipynb", cells)


# ======================================================================
# Modern CNN notebook generator (used for AlexNet/VGG/ResNet/EffNetB3/
# DenseNet/Swin-Tiny — all share the SAME pipeline, only the architecture
# string differs). Pipeline: focal loss + weighted sampler + RandAugment +
# mixup/cutmix + AdamW + warmup-cosine + EMA + TTA + threshold tuning.
# ======================================================================

# Notebook-numbering map. Keep stable across rebuilds.
ARCH_NB_NUMBER = {
    "alexnet":         "04",
    "vgg16_bn":        "05",
    "resnet50":        "06",
    "efficientnet_b3": "07",
    "densenet121":     "08",
    "swin_tiny":       "09",
}

ARCH_DISPLAY_NAME = {
    "alexnet":         "AlexNet",
    "vgg16_bn":        "VGG16-BN",
    "resnet50":        "ResNet50",
    "efficientnet_b3": "EfficientNet-B3",
    "densenet121":     "DenseNet121",
    "swin_tiny":       "Swin-Tiny",
}


def nb_modern_cnn(arch: str):
    nb_num = ARCH_NB_NUMBER[arch]
    nice = ARCH_DISPLAY_NAME[arch]
    method_name = arch       # used as filename prefix in results/
    cells = [
        md(f"""
# {nb_num} — {nice} (transfer learning)

**Pipeline** (shared across all six CNNs):

- **Input:** lesion-cropped images from `X_all.npy` (448×448 storage,
  Otsu segmentation applied in 00_data_setup).
- **Loss:** Focal Loss (γ=2.0) with class-balanced α (Cui et al. 2019, β=0.999).
- **Sampler:** WeightedRandomSampler — every batch is approximately 50/50
  melanoma/non-melanoma despite the 1:8 dataset imbalance.
- **Augmentation (conservative, medical-grade):** RandomResizedCrop(0.85, 1.0)
  + HFlip + VFlip + Rotation(±15°) + mild ColorJitter
  (brightness/contrast=0.10, saturation=0.05, hue=0.02) + small RandomErasing.
  RandAugment, Mixup and CutMix are intentionally disabled because they would
  distort or replace the lesion pixels that the ABCD diagnostic rule depends on.
- **Optimizer:** AdamW (wd=1e-4) with discriminative LR (head 3e-4, backbone 3e-5).
- **Schedule:** linear warmup (3 epochs) → cosine annealing to 1e-6.
- **Stage 1** (head only, 3 epochs, lr=1e-3) → **Stage 2** (full backbone,
  up to 25 epochs, early stopping on val F1, patience=7).
- **EMA** of weights (decay=0.999) — validation, threshold selection and TTA
  all use EMA weights.
- **Test-time augmentation:** 8-way (identity, hflip, vflip, hvflip,
  rot90/180/270, hflip+rot90); probabilities averaged.
- **Threshold:** F1-maximising threshold tuned on validation, applied once on test.

**Disconnect-proof:** the best-val-F1 weights are persisted to
`MyDrive/melanoma/checkpoints/{arch}_best.pt` every time val F1 improves.
If the runtime dies, the recovery cell at the bottom of this notebook reloads
the checkpoint, runs threshold tuning + TTA evaluation, and saves outputs —
no retraining required.
"""),
        code(COLAB_PREAMBLE),
        code(SEED_BLOCK),
        code("""
!pip install --quiet timm 2>&1 | tail -n 1
"""),
        code(f"""
# --- Architecture configuration ---
ARCH = "{arch}"
INPUT_SIZE = config.ARCH_CONFIG[ARCH]["input_size"]
BATCH_SIZE = config.ARCH_CONFIG[ARCH]["batch_size"]
print(f"Architecture: {{ARCH}}  input={{INPUT_SIZE}}  batch_size={{BATCH_SIZE}}")
"""),
        code("""
# --- Build datasets / dataloaders (data auto-cached to /content/local_data) ---
import numpy as np, torch
from torch.utils.data import DataLoader
from src.data import (load_arrays, HAMDataset,
                      make_train_transform_strong, make_eval_transform)
from src.training import make_weighted_sampler

X, y, ids, idx_train, idx_val, idx_test = load_arrays(config.DATA_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, "  X:", X.shape)

train_tf = make_train_transform_strong(
    INPUT_SIZE,
    rotation_deg=config.AUG_ROTATION_DEG,
    brightness=config.AUG_COLOR_BRIGHTNESS,
    contrast=config.AUG_COLOR_CONTRAST,
    saturation=config.AUG_COLOR_SATURATION,
    hue=config.AUG_COLOR_HUE,
    crop_scale_min=config.AUG_CROP_SCALE_MIN,
    crop_scale_max=config.AUG_CROP_SCALE_MAX,
    erasing_p=config.AUG_RANDOM_ERASING_P,
    erasing_scale=config.AUG_RANDOM_ERASING_SCALE,
)
eval_tf  = make_eval_transform(INPUT_SIZE)

train_ds = HAMDataset(X, y, idx_train, train_tf)
val_ds   = HAMDataset(X, y, idx_val,   eval_tf)
test_ds  = HAMDataset(X, y, idx_test,  eval_tf)

# Weighted sampler -> ~50/50 batches (1:8 imbalance otherwise)
sampler = make_weighted_sampler(y[idx_train]) if config.USE_WEIGHTED_SAMPLER else None

train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE,
                      sampler=sampler, shuffle=(sampler is None),
                      num_workers=2, pin_memory=True, drop_last=True)
val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=2, pin_memory=True)
test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=2, pin_memory=True)
print(f"Batches: train={len(train_ld)}  val={len(val_ld)}  test={len(test_ld)}")
"""),
        code("""
# --- Build model ---
from src.models import (BUILDERS, freeze_backbone, unfreeze_all,
                        trainable_params, discriminative_param_groups,
                        gradcam_target_layer)

model = BUILDERS[ARCH](num_classes=2, pretrained=True).to(device)
freeze_backbone(model)
print(f"Stage-1 trainable params: {trainable_params(model):,}")
"""),
        code("""
# --- Focal loss (class-balanced alpha) ---
from src.training import build_focal_loss
criterion = build_focal_loss(y[idx_train],
                             gamma=config.FOCAL_GAMMA,
                             beta=config.FOCAL_BETA,
                             device=device)
print(f"Focal loss: gamma={config.FOCAL_GAMMA}  beta={config.FOCAL_BETA}")
print(f"Class-balanced alpha: {criterion.alpha.cpu().numpy()}")
"""),
        code(f"""
# --- Stage 1: head only ---
import time, torch
from src.training import train_loop_v2, EpochLog, EMA, WarmupCosineSchedule

CHECKPOINT_PATH = config.CHECKPOINT_DIR / f"{{ARCH}}_best.pt"
print("Best weights -> ", CHECKPOINT_PATH)

# EMA shadow tracks the live model from the start
ema = EMA(model, decay=config.CNN_EMA_DECAY)

opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=config.CNN_HEAD_LR, weight_decay=config.CNN_WEIGHT_DECAY,
)
log = EpochLog()
t0 = time.time()
log = train_loop_v2(
    model, train_ld, val_ld, criterion, opt, device,
    epochs=config.CNN_HEAD_EPOCHS,
    num_classes=2,
    ema=ema,
    # No mixup/cutmix in stage 1 — head needs clean signal
    mixup_alpha=0.0, cutmix_alpha=0.0, mixup_prob=0.0, cutmix_prob=0.0,
    log=log, checkpoint_path=CHECKPOINT_PATH,
)
"""),
        code("""
# --- Stage 2: unfreeze full backbone with discriminative LR + warmup-cosine ---
unfreeze_all(model)
print(f"Stage-2 trainable params: {trainable_params(model):,}")

groups = discriminative_param_groups(
    model,
    head_lr=config.CNN_FT_LR_HEAD,
    backbone_lr=config.CNN_FT_LR_BACKBONE,
    weight_decay=config.CNN_WEIGHT_DECAY,
)
opt = torch.optim.AdamW(groups)
sched = WarmupCosineSchedule(
    opt,
    warmup_epochs=config.CNN_WARMUP_EPOCHS,
    total_epochs=config.CNN_FT_EPOCHS,
    base_lrs=[config.CNN_FT_LR_BACKBONE, config.CNN_FT_LR_HEAD],
    min_lr=1e-6,
)

log = train_loop_v2(
    model, train_ld, val_ld, criterion, opt, device,
    epochs=config.CNN_FT_EPOCHS,
    num_classes=2,
    scheduler=sched, ema=ema,
    # Mixup/CutMix intentionally disabled (see notebook header).
    mixup_alpha=0.0, cutmix_alpha=0.0, mixup_prob=0.0, cutmix_prob=0.0,
    early_stop_patience=config.CNN_EARLY_STOP_PATIENCE,
    log=log, checkpoint_path=CHECKPOINT_PATH,
)
train_time = time.time() - t0
print(f"Total training time: {train_time:.1f}s. Best weights at {CHECKPOINT_PATH}")
"""),
        code("""
# --- Plot loss / F1 curves and save the raw arrays for the aggregation overlay ---
import numpy as np, matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(log.train_loss, label="train"); ax[0].plot(log.val_loss, label="val")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(log.val_f1); ax[1].set_title("Val F1"); ax[1].set_xlabel("epoch")
fig.tight_layout()
fig.savefig(config.RESULTS_DIR / f"{ARCH}_curves.png", dpi=120)
np.savez(config.RESULTS_DIR / f"{ARCH}_curves.npz",
         train_loss=np.array(log.train_loss),
         val_loss=np.array(log.val_loss),
         val_f1=np.array(log.val_f1))
plt.show()
"""),
        code("""
# --- Apply EMA weights for evaluation ---
ema.apply_shadow(model)
"""),
        code("""
# --- Single-pass val probabilities (for threshold tuning + 'no TTA' ablation) ---
import numpy as np
from src.training import predict, tune_threshold

y_val_true, _, y_val_prob = predict(model, val_ld, device)
best_t, best_val_f1 = tune_threshold(y_val_true, y_val_prob)
print(f"Threshold (no TTA) best on val: t={best_t:.3f}  F1={best_val_f1:.4f}")
"""),
        code("""
# --- TTA val probabilities + threshold ---
from src.training import tta_predict
y_val_true_tta, y_val_prob_tta = tta_predict(model, val_ld, device, config.TTA_TRANSFORMS)
best_t_tta, best_val_f1_tta = tune_threshold(y_val_true_tta, y_val_prob_tta)
print(f"Threshold (TTA)   best on val: t={best_t_tta:.3f}  F1={best_val_f1_tta:.4f}")
"""),
        code("""
# --- Test evaluation: single-pass + TTA, with both thresholds ---
import time
import numpy as np

# Single-pass test
t0 = time.time()
y_test_true, _, y_test_prob = predict(model, test_ld, device)
inf_ms_single = (time.time() - t0) * 1000.0 / len(y_test_true)

# TTA test
t0 = time.time()
y_test_true2, y_test_prob_tta = tta_predict(model, test_ld, device, config.TTA_TRANSFORMS)
inf_ms_tta = (time.time() - t0) * 1000.0 / len(y_test_true2)
assert np.array_equal(y_test_true, y_test_true2)

y_pred_single = (y_test_prob > best_t).astype(int)
y_pred_tta    = (y_test_prob_tta > best_t_tta).astype(int)

print(f"Single-pass inf: {inf_ms_single:.2f} ms/img")
print(f"TTA (8-way) inf: {inf_ms_tta:.2f} ms/img")
"""),
        code("""
# --- Save standardized outputs (TTA result is the headline, but both columns persist) ---
import pandas as pd
from src.evaluation import save_standard_outputs, compute_metrics

# Headline metrics use TTA + tuned threshold
hp = dict(
    arch=ARCH,
    input_size=int(INPUT_SIZE),
    batch_size=int(BATCH_SIZE),
    head_lr=config.CNN_HEAD_LR,
    head_epochs=config.CNN_HEAD_EPOCHS,
    ft_lr_head=config.CNN_FT_LR_HEAD,
    ft_lr_backbone=config.CNN_FT_LR_BACKBONE,
    ft_epochs=config.CNN_FT_EPOCHS,
    warmup_epochs=config.CNN_WARMUP_EPOCHS,
    early_stop_patience=config.CNN_EARLY_STOP_PATIENCE,
    weight_decay=config.CNN_WEIGHT_DECAY,
    ema_decay=config.CNN_EMA_DECAY,
    focal_gamma=config.FOCAL_GAMMA,
    focal_beta=config.FOCAL_BETA,
    weighted_sampler=bool(config.USE_WEIGHTED_SAMPLER),
    augmentation=("RandomResizedCrop(0.85,1.0)+HFlip+VFlip+"
                  f"Rotation({config.AUG_ROTATION_DEG})+mildColorJitter+"
                  f"RandomErasing(p={config.AUG_RANDOM_ERASING_P}); "
                  "no Mixup/CutMix/RandAugment"),
    stored_image_size=int(config.IMG_SIZE),
    segmentation="Otsu(LAB-L invert)+morph close/open+largest CC+15% margin crop",
    split="lesion-grouped stratified 70/15/15",
    decision_threshold_tta=best_t_tta,
    decision_threshold_single=best_t,
    threshold_selection="argmax F1 on validation set",
    tta_transforms=list(config.TTA_TRANSFORMS),
)

metrics = save_standard_outputs(
    method_name=ARCH,
    results_dir=config.RESULTS_DIR,
    y_true=y_test_true,
    y_pred=y_pred_tta,         # headline = TTA + tuned threshold
    y_prob=y_test_prob_tta,    # headline probability = TTA-averaged
    ids=ids[idx_test],
    hyperparameters=hp,
    train_time_sec=train_time,
    inference_time_per_image_ms=inf_ms_tta,
)

# Also save the rich TEST predictions CSV with both single and TTA columns
# (the aggregation notebook needs this to compute the TTA ablation row).
pd.DataFrame({
    "image_id": ids[idx_test],
    "y_true":   y_test_true,
    "y_pred_single": y_pred_single,
    "y_prob_single": y_test_prob,
    "y_pred_tta":    y_pred_tta,
    "y_prob_tta":    y_test_prob_tta,
    "best_t_single": best_t,
    "best_t_tta":    best_t_tta,
}).to_csv(config.RESULTS_DIR / f"{ARCH}_predictions.csv", index=False)

# VAL predictions CSV — required by the ensemble (method 9) for threshold
# tuning on a held-out set. The aggregation notebook averages these
# probabilities across all 6 CNNs, sweeps thresholds on val, and applies
# the best threshold once on the test ensemble.
pd.DataFrame({
    "image_id": ids[idx_val],
    "y_true":   y_val_true,
    "y_prob_single": y_val_prob,
    "y_prob_tta":    y_val_prob_tta,
}).to_csv(config.RESULTS_DIR / f"{ARCH}_val_predictions.csv", index=False)

# Side-by-side diagnostic
m_single = compute_metrics(y_test_true, y_pred_single, y_test_prob)
m_tta    = compute_metrics(y_test_true, y_pred_tta,    y_test_prob_tta)
print("Single-pass test metrics:", {k: round(v, 4) for k, v in m_single.items()})
print("TTA-headline   test metrics:", {k: round(v, 4) for k, v in m_tta.items()})
"""),
        code("""
# --- Grad-CAM grid: 4 melanoma + 4 non-melanoma test images ---
import numpy as np, torch, matplotlib.pyplot as plt
from src.gradcam import GradCAM, overlay_heatmap

target_layer = gradcam_target_layer(model, ARCH)
cam = GradCAM(model, target_layer)
mel_idx    = idx_test[y[idx_test] == 1][:4]
nonmel_idx = idx_test[y[idx_test] == 0][:4]
chosen = list(mel_idx) + list(nonmel_idx)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, k in zip(axes.flat, chosen):
    img = X[k]
    t = eval_tf(img).unsqueeze(0).to(device)
    try:
        heat = cam(t, class_idx=int(y[k]))
        ax.imshow(overlay_heatmap(img, heat))
    except Exception:
        # Some attention archs (e.g. Swin) may not produce a valid CAM via
        # plain gradcam; fall back to showing the raw image.
        ax.imshow(img); ax.set_title(f"Grad-CAM N/A for {ARCH}")
    ax.set_title(f"id={ids[k]}  true={int(y[k])}")
    ax.axis("off")
cam.close()
fig.tight_layout()
fig.savefig(config.RESULTS_DIR / f"{ARCH}_gradcam_grid.png", dpi=120)
plt.show()
"""),
        md(f"""
---
## Recovery cell (use only if the runtime disconnected during training)

If the cells above never finished but you saw at least one
`-> checkpoint saved` line during training, the best weights are at
`MyDrive/melanoma/checkpoints/{arch}_best.pt`. Run **only the recovery cell
below** on a fresh runtime (after re-running the preamble + dataset cells)
to finish evaluation without retraining.
"""),
        code("""
# --- Recovery: load checkpoint, re-run threshold + TTA, save outputs ---
import torch, time, numpy as np, pandas as pd
from src.training import predict, tune_threshold, tta_predict
from src.evaluation import save_standard_outputs

CHECKPOINT_PATH = config.CHECKPOINT_DIR / f"{ARCH}_best.pt"
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

y_val_true, _, y_val_prob = predict(model, val_ld, device)
best_t, _ = tune_threshold(y_val_true, y_val_prob)
y_val_true_tta, y_val_prob_tta = tta_predict(model, val_ld, device, config.TTA_TRANSFORMS)
best_t_tta, _ = tune_threshold(y_val_true_tta, y_val_prob_tta)

t0 = time.time()
y_test_true, _, y_test_prob = predict(model, test_ld, device)
inf_ms_single = (time.time() - t0) * 1000.0 / len(y_test_true)
t0 = time.time()
_, y_test_prob_tta = tta_predict(model, test_ld, device, config.TTA_TRANSFORMS)
inf_ms_tta = (time.time() - t0) * 1000.0 / len(y_test_true)

y_pred_tta = (y_test_prob_tta > best_t_tta).astype(int)
y_pred_single = (y_test_prob > best_t).astype(int)
metrics = save_standard_outputs(
    method_name=ARCH,
    results_dir=config.RESULTS_DIR,
    y_true=y_test_true, y_pred=y_pred_tta, y_prob=y_test_prob_tta,
    ids=ids[idx_test],
    hyperparameters={"recovered_from_checkpoint": True,
                     "decision_threshold_tta": best_t_tta,
                     "decision_threshold_single": best_t},
    train_time_sec=0.0,
    inference_time_per_image_ms=inf_ms_tta,
)

# Save the rich predictions CSVs (test + val) — same layout as the main
# pipeline cell so the aggregation notebook can read either source.
pd.DataFrame({
    "image_id": ids[idx_test],
    "y_true":   y_test_true,
    "y_pred_single": y_pred_single,
    "y_prob_single": y_test_prob,
    "y_pred_tta":    y_pred_tta,
    "y_prob_tta":    y_test_prob_tta,
    "best_t_single": best_t,
    "best_t_tta":    best_t_tta,
}).to_csv(config.RESULTS_DIR / f"{ARCH}_predictions.csv", index=False)
pd.DataFrame({
    "image_id": ids[idx_val],
    "y_true":   y_val_true,
    "y_prob_single": y_val_prob,
    "y_prob_tta":    y_val_prob_tta,
}).to_csv(config.RESULTS_DIR / f"{ARCH}_val_predictions.csv", index=False)
{k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))}
"""),
    ]
    nb_filename = f"{nb_num}_{arch}.ipynb"
    write_nb(nb_filename, cells)


# ======================================================================
# 04_ablation_study.ipynb (legacy — kept for old EfficientNet-B0 ablation)
# ======================================================================
def nb_ablation():
    cells = [
        md("""
# 04 — Ablation Study (EfficientNet-B0, frozen backbone, 5 epochs each)

Four cumulative variants — A → D — to isolate the contribution of each ingredient.
Output: `results/ablation_table.csv`.
"""),
        code(COLAB_PREAMBLE),
        code(SEED_BLOCK),
        code("""
import numpy as np, torch, time, pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.data import load_arrays, class_weights, softened_class_weights
from src.models import build_efficientnet_b0, freeze_backbone

# Note: variant C requires a "no hair removal" version. Since 00_data_setup.ipynb
# applied hair removal, this ablation flag is informative only — for a fully
# rigorous C, rebuild X with `do_hair_removal=False`. We document this clearly
# in the paper.
X, y, ids, idx_train, idx_val, idx_test = load_arrays(config.DATA_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

class DS(Dataset):
    def __init__(self, X, y, idx, tf): self.X, self.y, self.idx, self.tf = X, y, idx, tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, i): k = self.idx[i]; return self.tf(self.X[k]), int(self.y[k])
"""),
        code("""
norm = transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
plain_tf = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), norm])
aug_tf   = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
    transforms.ToTensor(), norm,
])

def loaders(train_tf, eval_tf):
    bs = config.EFFNET_BATCH_SIZE
    return (
        DataLoader(DS(X, y, idx_train, train_tf), batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True),
        DataLoader(DS(X, y, idx_val,   eval_tf),  batch_size=bs, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(DS(X, y, idx_test,  eval_tf),  batch_size=bs, shuffle=False, num_workers=2, pin_memory=True),
    )
"""),
        code("""
def run_variant(name, augment, class_weighting, epochs=5):
    train_tf = aug_tf if augment else plain_tf
    train_ld, val_ld, test_ld = loaders(train_tf, plain_tf)

    m = build_efficientnet_b0(num_classes=2, pretrained=True).to(device)
    freeze_backbone(m)

    if class_weighting:
        cw = class_weights(y[idx_train])
        w = torch.tensor([cw[0], cw[1]], dtype=torch.float32, device=device)
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=config.EFFNET_HEAD_LR)

    for epoch in range(epochs):
        m.train()
        for x_, t in train_ld:
            x_, t = x_.to(device), t.to(device)
            opt.zero_grad(); loss = crit(m(x_), t); loss.backward(); opt.step()

    m.eval()
    ys, ps, prs = [], [], []
    with torch.no_grad():
        for x_, t in test_ld:
            logits = m(x_.to(device))
            prs.append(torch.softmax(logits, 1)[:, 1].cpu().numpy())
            ps.append(logits.argmax(1).cpu().numpy())
            ys.append(t.numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps); prs = np.concatenate(prs)
    return dict(variant=name,
                accuracy=float(accuracy_score(ys, ps)),
                f1=float(f1_score(ys, ps, zero_division=0)),
                roc_auc=float(roc_auc_score(ys, prs)))
"""),
        code("""
rows = []
rows.append(run_variant("A: vanilla",           augment=False, class_weighting=False))
rows.append(run_variant("B: + augmentation",    augment=True,  class_weighting=False))
rows.append(run_variant("C: + hair-removed data (already in X_all)",
                        augment=True,  class_weighting=False))
rows.append(run_variant("D: + class weighting", augment=True,  class_weighting=True))
df = pd.DataFrame(rows)
df.to_csv(config.RESULTS_DIR / "ablation_table.csv", index=False)
df
"""),
    ]
    write_nb("04_ablation_study.ipynb", cells)


# ======================================================================
# 05_results_aggregation.ipynb
# ======================================================================
def nb_aggregation():
    cells = [
        md("""
# 05 — Results Aggregation

Reads every `results/*_metrics.json`, builds the final comparison table,
the literature-comparison table, and the 8-image error-analysis figure.
"""),
        code(COLAB_PREAMBLE),
        code("""
import json, pandas as pd, numpy as np
from pathlib import Path

rows = []
for p in sorted(config.RESULTS_DIR.glob("*_metrics.json")):
    m = json.loads(p.read_text())
    rows.append({
        "method":  p.stem.replace("_metrics", ""),
        "accuracy":  round(m["accuracy"], 4),
        "precision": round(m["precision"], 4),
        "recall":    round(m["recall"], 4),
        "f1":        round(m["f1"], 4),
        "roc_auc":   round(m["roc_auc"], 4),
        "train_time_sec": round(m["train_time_sec"], 1),
        "inference_ms_per_image": round(m["inference_time_per_image_ms"], 3),
    })
df = pd.DataFrame(rows)
df.to_csv(config.RESULTS_DIR / "comparison_table.csv", index=False)
df
"""),
        code("""
# --- Write markdown comparison table into paper/tables_and_figures.md ---
md_lines = ["## Method comparison (auto-generated)\\n",
            "| " + " | ".join(df.columns) + " |",
            "|" + "|".join(["---"] * len(df.columns)) + "|"]
for _, r in df.iterrows():
    md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
out = config.PAPER_DIR / "tables_and_figures.md"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\\n".join(md_lines))
print("Wrote", out)
"""),
        code("""
# --- Error analysis: 4 false positives + 4 false negatives from EfficientNet ---
import pandas as pd, numpy as np, matplotlib.pyplot as plt, cv2
from src.data import load_arrays
X, y, ids, idx_train, idx_val, idx_test = load_arrays(config.DATA_DIR)

pred_csv = config.RESULTS_DIR / "deep_learning_effnet_predictions.csv"
P = pd.read_csv(pred_csv)
fp = P[(P.y_true == 0) & (P.y_pred == 1)].sort_values("y_prob", ascending=False).head(4)
fn = P[(P.y_true == 1) & (P.y_pred == 0)].sort_values("y_prob", ascending=True).head(4)
chosen = pd.concat([fp, fn])

# image_id -> array index
id_to_idx = {iid: i for i, iid in enumerate(ids)}
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, (_, r) in zip(axes.flat, chosen.iterrows()):
    k = id_to_idx[r.image_id]
    ax.imshow(X[k])
    ax.set_title(f"true={int(r.y_true)}  pred={int(r.y_pred)}  p={r.y_prob:.2f}")
    ax.axis("off")
fig.suptitle("Top row: false positives (predicted melanoma, actually benign).  Bottom row: false negatives.")
fig.tight_layout(); fig.savefig(config.RESULTS_DIR / "error_analysis.png", dpi=120)
plt.show()
"""),
        md("""
### Failure-mode commentary (paper-ready, edit as needed)

- **False positives** are typically benign nevi with **dark, irregular pigment
  patterns** that mimic melanoma. The model latches on to high-frequency
  texture without weighting border smoothness.
- **False negatives** tend to be melanomas with **low contrast against
  surrounding skin** or partial occlusion by **residual hair** that survived
  the DullRazor mask.
- A subset of misclassified images shows **specular reflections** from the
  dermoscope's polariser, which the augmentation pipeline does not simulate.
- Finally, a few errors are simply **ambiguous borders** — cases where even
  expert dermatologists disagree without histology.
"""),
    ]
    write_nb("05_results_aggregation.ipynb", cells)


# ======================================================================
# 10_aggregation.ipynb — 9-method comparison + ensemble + ablations
#
# Reads:
#   results/baseline_logistic_metrics.json + _predictions.csv
#   results/classical_ml_svm_metrics.json  + _predictions.csv
#   results/{arch}_metrics.json            + _predictions.csv  (6 CNNs)
#   results/{arch}_val_predictions.csv     (6 CNNs; required for ensemble)
#   results/{arch}_curves.npz              (6 CNNs; required for epoch overlay)
#
# Produces:
#   results/comparison_table.csv           # 9-row headline
#   results/epoch_curves.png               # val F1 vs epoch overlay (6 CNNs)
#   results/roc_overlay.png                # 9 ROC curves on one axis
#   results/error_analysis.png             # 4 FP + 4 FN from best CNN
#   results/ablation_table.csv             # focal/sampler vs softened CW; TTA on/off; threshold; ensemble
#   results/literature_comparison.csv      # our rows + prior-work numbers
#   results/ensemble_metrics.json          # ensemble (method 9)
#   results/ensemble_predictions.csv       # ensemble test predictions
#   paper/tables_and_figures.md            # markdown tables for Overleaf
#
# The ensemble reloads ResNet50 from its checkpoint to recover the missing
# val_predictions (Phase 1 notebook didn't save it). Other archs save val
# predictions during their main run; if any are missing, this notebook
# also recomputes them via checkpoint reload.
# ======================================================================
def nb_aggregation_v2():
    cells = [
        md("""
# 10 — Aggregation, Ensemble, Ablations

This is the final notebook of the benchmark. It does **not** train any
model. It reads everything the previous notebooks left in
`MyDrive/melanoma/results/` and `MyDrive/melanoma/checkpoints/` and
produces every table and figure the IEEE paper needs.

Run on a **CPU runtime** to save GPU compute units. The only GPU step is
the ensemble's val/test inference for any CNN whose `*_val_predictions.csv`
is missing — set the runtime to **A100 / L4** if you want that step to
run in seconds rather than minutes. (It can also run on CPU; just slower.)
"""),
        code(COLAB_PREAMBLE),
        code(SEED_BLOCK),
        code("""
!pip install --quiet timm 2>&1 | tail -n 1
"""),
        code("""
# --- Inputs we expect ---
import json, numpy as np, pandas as pd, torch
from pathlib import Path

ARCHES = ["alexnet", "vgg16_bn", "resnet50", "efficientnet_b3", "densenet121", "swin_tiny"]
NON_DEEP = ["baseline_logistic", "classical_ml_svm"]
LEGACY_FOR_ABLATION = "deep_learning_effnet"  # old EfficientNet-B0, softened CW + light aug

print("Listing results dir:")
for p in sorted(config.RESULTS_DIR.iterdir()):
    print(" ", p.name)
"""),
        code("""
# --- Build a working DataFrame of every method's headline metrics ---
def load_metrics(name):
    p = config.RESULTS_DIR / f"{name}_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

rows = []
for name in NON_DEEP + ARCHES:
    m = load_metrics(name)
    if m is None:
        print(f"  WARN: missing {name}_metrics.json")
        continue
    rows.append({
        "method":    name,
        "accuracy":  m.get("accuracy"),
        "precision": m.get("precision"),
        "recall":    m.get("recall"),
        "f1":        m.get("f1"),
        "roc_auc":   m.get("roc_auc"),
        "train_time_sec": m.get("train_time_sec"),
        "inference_ms_per_img": m.get("inference_time_per_image_ms"),
    })
df_pre = pd.DataFrame(rows)

# Method 10 (hybrid fusion) is optional — append if its metrics file exists.
hf = load_metrics("hybrid_fusion")
if hf is not None:
    df_pre = pd.concat([df_pre, pd.DataFrame([{
        "method":    "hybrid_fusion",
        "accuracy":  hf.get("accuracy"),
        "precision": hf.get("precision"),
        "recall":    hf.get("recall"),
        "f1":        hf.get("f1"),
        "roc_auc":   hf.get("roc_auc"),
        "train_time_sec": hf.get("train_time_sec"),
        "inference_ms_per_img": hf.get("inference_time_per_image_ms"),
    }])], ignore_index=True)
    print("  + hybrid_fusion row included.")

print(df_pre.to_string(index=False))
"""),
        code("""
# --- Compute the soft-vote ensemble (method 9) ---
# We need val + test probabilities from each CNN. Test probs are in
# {arch}_predictions.csv (column y_prob_tta). Val probs are in
# {arch}_val_predictions.csv. If a val CSV is missing (e.g. ResNet50 from
# Phase 1), reload that arch's checkpoint and recompute val + test predictions.

import torch
from torch.utils.data import DataLoader
from src.data import load_arrays, HAMDataset, make_eval_transform
from src.models import BUILDERS
from src.training import predict, tta_predict, tune_threshold

X, y, ids, idx_train, idx_val, idx_test = load_arrays(config.DATA_DIR)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

def recompute_val_predictions(arch):
    \"\"\"Reload checkpoint and re-do val + test passes; save both CSVs.\"\"\"
    print(f"  recomputing val/test predictions for {arch}...")
    cfg = config.ARCH_CONFIG[arch]
    model = BUILDERS[arch](num_classes=2, pretrained=False).to(device)
    ckpt = torch.load(config.CHECKPOINT_DIR / f"{arch}_best.pt", map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    eval_tf = make_eval_transform(cfg["input_size"])
    val_ld  = DataLoader(HAMDataset(X, y, idx_val,  eval_tf), batch_size=cfg["batch_size"],
                         shuffle=False, num_workers=2, pin_memory=True)
    test_ld = DataLoader(HAMDataset(X, y, idx_test, eval_tf), batch_size=cfg["batch_size"],
                         shuffle=False, num_workers=2, pin_memory=True)

    yv, _, pv = predict(model, val_ld, device)
    yv2, pv_tta = tta_predict(model, val_ld, device, config.TTA_TRANSFORMS)
    yt, _, pt = predict(model, test_ld, device)
    yt2, pt_tta = tta_predict(model, test_ld, device, config.TTA_TRANSFORMS)

    pd.DataFrame({"image_id": ids[idx_val], "y_true": yv,
                  "y_prob_single": pv, "y_prob_tta": pv_tta}
                ).to_csv(config.RESULTS_DIR / f"{arch}_val_predictions.csv", index=False)

    # Refresh the test CSV too — but only if it doesn't already have y_prob_tta
    test_csv = config.RESULTS_DIR / f"{arch}_predictions.csv"
    if test_csv.exists():
        existing = pd.read_csv(test_csv)
        if "y_prob_tta" in existing.columns:
            return  # already rich; don't overwrite
    bt, _ = tune_threshold(yv, pv)
    bt_tta, _ = tune_threshold(yv2, pv_tta)
    pd.DataFrame({"image_id": ids[idx_test], "y_true": yt,
                  "y_pred_single": (pt > bt).astype(int),     "y_prob_single": pt,
                  "y_pred_tta":    (pt_tta > bt_tta).astype(int), "y_prob_tta":    pt_tta,
                  "best_t_single": bt, "best_t_tta": bt_tta,
                 }).to_csv(test_csv, index=False)

# Verify each CNN has val_predictions.csv; recompute from checkpoint if missing.
# CNNs whose checkpoint ALSO doesn't exist are dropped from ARCHES so the
# ensemble row and downstream tables build on whatever is actually trained.
TRAINED_ARCHES = []
for arch in ARCHES:
    val_csv = config.RESULTS_DIR / f"{arch}_val_predictions.csv"
    ckpt    = config.CHECKPOINT_DIR / f"{arch}_best.pt"
    test_csv = config.RESULTS_DIR / f"{arch}_predictions.csv"
    if val_csv.exists() and test_csv.exists():
        TRAINED_ARCHES.append(arch)
        print(f"  ok: {arch}_val_predictions.csv present")
    elif ckpt.exists() and test_csv.exists():
        recompute_val_predictions(arch)
        TRAINED_ARCHES.append(arch)
    else:
        print(f"  SKIP: {arch} (no checkpoint and no predictions on Drive)")

if not TRAINED_ARCHES:
    raise RuntimeError("No CNN checkpoints or predictions found on Drive. "
                       "Run at least one of notebooks 04-09 before this aggregation.")
print(f"\\nEnsemble will average over {len(TRAINED_ARCHES)} CNN(s): {TRAINED_ARCHES}")
ARCHES = TRAINED_ARCHES   # rebind so downstream cells use only trained models
"""),
        code("""
# --- Build the ensemble: average TTA probabilities across the 6 CNNs ---
val_probs_list, test_probs_list = [], []
y_val_true_ref, y_test_true_ref = None, None

for arch in ARCHES:
    vdf = pd.read_csv(config.RESULTS_DIR / f"{arch}_val_predictions.csv")
    tdf = pd.read_csv(config.RESULTS_DIR / f"{arch}_predictions.csv")
    val_probs_list.append(vdf["y_prob_tta"].to_numpy())
    test_probs_list.append(tdf["y_prob_tta"].to_numpy())
    if y_val_true_ref  is None: y_val_true_ref  = vdf["y_true"].to_numpy()
    if y_test_true_ref is None: y_test_true_ref = tdf["y_true"].to_numpy()

ens_val_prob  = np.mean(val_probs_list,  axis=0)
ens_test_prob = np.mean(test_probs_list, axis=0)

best_t_ens, val_f1_ens = tune_threshold(y_val_true_ref, ens_val_prob)
print(f"Ensemble val F1 = {val_f1_ens:.4f}  at threshold = {best_t_ens:.3f}")

ens_test_pred = (ens_test_prob > best_t_ens).astype(int)
"""),
        code("""
# --- Save ensemble outputs in the standard layout ---
from src.evaluation import save_standard_outputs, compute_metrics

ens_metrics = save_standard_outputs(
    method_name="ensemble",
    results_dir=config.RESULTS_DIR,
    y_true=y_test_true_ref,
    y_pred=ens_test_pred,
    y_prob=ens_test_prob,
    ids=pd.read_csv(config.RESULTS_DIR / f"{ARCHES[0]}_predictions.csv")["image_id"].to_numpy(),
    hyperparameters={
        "members": ARCHES,
        "weights": "uniform soft-vote",
        "decision_threshold_tta": float(best_t_ens),
    },
    train_time_sec=0.0,
    inference_time_per_image_ms=float(np.mean(
        [load_metrics(a).get("inference_time_per_image_ms", 0.0) for a in ARCHES])),
)
print({k: round(v, 4) for k, v in ens_metrics.items() if isinstance(v, (int, float))})

# Append to the dataframe
df_pre = pd.concat([df_pre, pd.DataFrame([{
    "method":   "ensemble",
    "accuracy": ens_metrics["accuracy"],
    "precision":ens_metrics["precision"],
    "recall":   ens_metrics["recall"],
    "f1":       ens_metrics["f1"],
    "roc_auc":  ens_metrics["roc_auc"],
    "train_time_sec": 0.0,
    "inference_ms_per_img": ens_metrics["inference_time_per_image_ms"],
}])], ignore_index=True)
"""),
        code("""
# --- Save the headline 9-row comparison table ---
df_pre = df_pre.round(4)
df_pre.to_csv(config.RESULTS_DIR / "comparison_table.csv", index=False)
print(df_pre.to_string(index=False))
"""),
        code("""
# --- Build the val-F1 vs epoch overlay (figure required by the syllabus) ---
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4))
for arch in ARCHES:
    npz = config.RESULTS_DIR / f"{arch}_curves.npz"
    if not npz.exists():
        print(f"  missing {npz.name}")
        continue
    d = np.load(npz)
    ax.plot(np.arange(1, len(d["val_f1"]) + 1), d["val_f1"], label=arch)
ax.set_xlabel("Epoch (stage-2 fine-tuning)")
ax.set_ylabel("Validation F1")
ax.set_title("Per-epoch validation F1 — 6 CNN architectures")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(config.RESULTS_DIR / "epoch_curves.png", dpi=120)
plt.show()
"""),
        code("""
# --- Build the 9-curve ROC overlay ---
from sklearn.metrics import roc_curve, roc_auc_score
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
for name in NON_DEEP + ARCHES + ["ensemble", "hybrid_fusion"]:
    csv = config.RESULTS_DIR / f"{name}_predictions.csv"
    if not csv.exists():
        continue
    pdf = pd.read_csv(csv)
    if "y_prob_tta" in pdf.columns:
        prob = pdf["y_prob_tta"].to_numpy()
    elif "y_prob" in pdf.columns:
        prob = pdf["y_prob"].to_numpy()
    else:
        continue
    fpr, tpr, _ = roc_curve(pdf["y_true"], prob)
    auc = roc_auc_score(pdf["y_true"], prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC — 9-method comparison on HAM10000 binary test set")
ax.legend(loc="lower right", fontsize=7)
fig.tight_layout()
fig.savefig(config.RESULTS_DIR / "roc_overlay.png", dpi=120)
plt.show()
"""),
        code("""
# --- Error analysis: 4 FP + 4 FN from the best single CNN ---
best_arch = df_pre[df_pre["method"].isin(ARCHES)].sort_values("f1", ascending=False).iloc[0]["method"]
print("Best single CNN:", best_arch)
pred_csv = config.RESULTS_DIR / f"{best_arch}_predictions.csv"
pdf = pd.read_csv(pred_csv)
fp_rows = pdf[(pdf["y_pred_tta"] == 1) & (pdf["y_true"] == 0)].sort_values("y_prob_tta", ascending=False).head(4)
fn_rows = pdf[(pdf["y_pred_tta"] == 0) & (pdf["y_true"] == 1)].sort_values("y_prob_tta", ascending=True ).head(4)

id_to_arr_idx = {str(s): i for i, s in enumerate(ids)}
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, (_, r) in zip(axes[0], fp_rows.iterrows()):
    k = id_to_arr_idx[str(r["image_id"])]
    ax.imshow(X[k]); ax.set_title(f"FP  p={r['y_prob_tta']:.2f}", fontsize=9); ax.axis("off")
for ax, (_, r) in zip(axes[1], fn_rows.iterrows()):
    k = id_to_arr_idx[str(r["image_id"])]
    ax.imshow(X[k]); ax.set_title(f"FN  p={r['y_prob_tta']:.2f}", fontsize=9); ax.axis("off")
fig.suptitle(f"Error analysis on test set — {best_arch} (top row: false positives, bottom: false negatives)")
fig.tight_layout()
fig.savefig(config.RESULTS_DIR / "error_analysis.png", dpi=120)
plt.show()
"""),
        code("""
# --- Ablation table (4 rows; no extra training) ---
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

def metrics_at_threshold(y_true, y_prob, t):
    y_pred = (y_prob > t).astype(int)
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_prob),
    )

ab_rows = []

# A) Loss/sampler: legacy softened-CW EfficientNet-B0 vs new focal+sampler EfficientNet-B3
m_legacy = load_metrics(LEGACY_FOR_ABLATION)
m_b3     = load_metrics("efficientnet_b3")
if m_legacy and m_b3:
    ab_rows.append({"ablation": "Loss + sampling: softened class-weight (legacy B0)",
                    **{k: m_legacy.get(k) for k in ["accuracy","precision","recall","f1","roc_auc"]}})
    ab_rows.append({"ablation": "Loss + sampling: Focal + WeightedRandomSampler (B3)",
                    **{k: m_b3.get(k) for k in ["accuracy","precision","recall","f1","roc_auc"]}})

# B) TTA on/off — averaged across the 6 CNNs
single_f1s, tta_f1s = [], []
for arch in ARCHES:
    pdf = pd.read_csv(config.RESULTS_DIR / f"{arch}_predictions.csv")
    single_f1s.append(f1_score(pdf["y_true"], pdf["y_pred_single"], zero_division=0))
    tta_f1s   .append(f1_score(pdf["y_true"], pdf["y_pred_tta"],    zero_division=0))
ab_rows.append({"ablation": "TTA OFF (mean across 6 CNNs)", "f1": float(np.mean(single_f1s))})
ab_rows.append({"ablation": "TTA ON  (mean across 6 CNNs)", "f1": float(np.mean(tta_f1s))})

# C) Threshold tuned vs t=0.5 — averaged across the 6 CNNs (TTA probs)
def_f1s, tuned_f1s = [], []
for arch in ARCHES:
    pdf = pd.read_csv(config.RESULTS_DIR / f"{arch}_predictions.csv")
    def_f1s.append(f1_score(pdf["y_true"], (pdf["y_prob_tta"] > 0.5).astype(int), zero_division=0))
    tuned_f1s.append(f1_score(pdf["y_true"], pdf["y_pred_tta"], zero_division=0))
ab_rows.append({"ablation": "Threshold = 0.5 (mean across 6 CNNs)",         "f1": float(np.mean(def_f1s))})
ab_rows.append({"ablation": "Threshold tuned on val (mean across 6 CNNs)",  "f1": float(np.mean(tuned_f1s))})

# D) Best single CNN vs ensemble
best_single = df_pre[df_pre["method"].isin(ARCHES)]["f1"].max()
ens_f1      = df_pre[df_pre["method"] == "ensemble"]["f1"].iloc[0]
ab_rows.append({"ablation": "Best single CNN (test F1)", "f1": float(best_single)})
ab_rows.append({"ablation": "Ensemble (soft-vote, 6 CNNs)", "f1": float(ens_f1)})

ab_df = pd.DataFrame(ab_rows).round(4)
ab_df.to_csv(config.RESULTS_DIR / "ablation_table.csv", index=False)
print(ab_df.to_string(index=False))
"""),
        code("""
# --- Literature comparison table ---
lit_rows = [
    {"reference": "Al-Waisy et al. 2025 [1]", "method": "Skin-DeepNet (HRNet+DBN+XGBoost fusion)",
     "dataset": "ISIC 2019 / HAM10000", "accuracy": "0.9965 / 1.000", "f1": "0.9954 / —"},
    {"reference": "Kaur et al. 2025 [2]", "method": "N-DCNN with hair removal + ACNN segmentation",
     "dataset": "ISIC 2020", "accuracy": "0.9340", "f1": "0.9398"},
    {"reference": "Naeem et al. 2024 [3]", "method": "SNC_Net Inception V3 + handcrafted entropy fusion",
     "dataset": "ISIC 2019", "accuracy": "0.9781", "f1": "0.9810"},
    {"reference": "Shahzaib et al. 2025 [25]", "method": "EHR + Deep Residual U-Net + DenseNet169",
     "dataset": "ISIC 2019", "accuracy": "0.9774", "f1": "—"},
    {"reference": "Albahli 2025 [37]", "method": "YOLOv8 multi-dataset",
     "dataset": "ISIC2020+HAM10000+PH2", "accuracy": "—", "f1": "0.905"},
    {"reference": "Bansal et al. 2022 [21]", "method": "Handcrafted + EfficientNet-B0 + ANN",
     "dataset": "HAM10000", "accuracy": "0.949", "f1": "—"},
]
# Append our rows
def _method_label(name):
    if name in ARCHES:
        return "single CNN (TL+focal+TTA)"
    if name == "ensemble":
        return "soft-vote ensemble"
    if name == "hybrid_fusion":
        return "handcrafted + ABCD + deep + SMOTE-Tomek + MLP/XGBoost"
    if name == "baseline_logistic":
        return "logistic regression on raw pixels"
    if name == "classical_ml_svm":
        return "HOG + Color + GLCM + PCA + RBF-SVM"
    return "n/a"

for _, r in df_pre.iterrows():
    lit_rows.append({"reference": "This work — " + r["method"],
                     "method": _method_label(r["method"]),
                     "dataset": "HAM10000 (binary, lesion-grouped)",
                     "accuracy": f'{r["accuracy"]:.4f}' if pd.notna(r["accuracy"]) else "—",
                     "f1":       f'{r["f1"]:.4f}'       if pd.notna(r["f1"])       else "—"})
lit_df = pd.DataFrame(lit_rows)
lit_df.to_csv(config.RESULTS_DIR / "literature_comparison.csv", index=False)
print(lit_df.to_string(index=False))
"""),
        code("""
# --- Write everything as Markdown into paper/tables_and_figures.md ---
from pathlib import Path

md_path = Path("/content/melanoma-detection-ham10000/paper/tables_and_figures.md")
local_md = config.PAPER_DIR / "tables_and_figures.md"

def df_to_md(df):
    return df.to_markdown(index=False)

def section(title, body):
    return f"## {title}\\n\\n{body}\\n\\n"

content = "# Tables and Figures (auto-generated by 10_aggregation.ipynb)\\n\\n"
content += section("Table I — Headline 9-method comparison (HAM10000 binary, test set)",
                   df_to_md(df_pre))
content += section("Table II — Ablations", df_to_md(ab_df))
content += section("Table III — Literature comparison", df_to_md(lit_df))
content += section("Figures (saved as PNG in MyDrive/melanoma/results/)",
                   "- `epoch_curves.png` — val F1 vs epoch overlay (6 CNNs)\\n"
                   "- `roc_overlay.png` — 9 ROC curves on one axis\\n"
                   "- `error_analysis.png` — 4 false positives + 4 false negatives from "
                   f"the best single CNN ({best_arch})\\n"
                   "- per-arch `*_curves.png`, `*_confusion_matrix.png`, `*_roc_curve.png`, "
                   "`*_gradcam_grid.png`")

# Save in two places: Drive (for the user) and local repo (for git)
local_md.parent.mkdir(parents=True, exist_ok=True)
local_md.write_text(content, encoding="utf-8")
md_path.parent.mkdir(parents=True, exist_ok=True)
md_path.write_text(content, encoding="utf-8")
print("Wrote:", local_md)
print("Wrote:", md_path)
"""),
        md("""
## Final summary

You should now have, in `MyDrive/melanoma/results/`:

- `comparison_table.csv` — the 9-row headline table for the IEEE paper.
- `ablation_table.csv` — 8 rows of ablation evidence (loss, TTA, threshold, ensemble).
- `literature_comparison.csv` — our six rows next to six prior-work numbers.
- `epoch_curves.png` — val F1 vs epoch overlay for the six CNNs.
- `roc_overlay.png` — nine ROC curves on one axis with AUC values in the legend.
- `error_analysis.png` — four false positives and four false negatives from
  the best single CNN, with the model's predicted probability shown.
- per-arch artefacts (curves, confusion matrices, Grad-CAM grids).

Plus, in `paper/tables_and_figures.md` (in this repo and on Drive), a
markdown rendering of all three tables ready to drop into Overleaf.
"""),
    ]
    write_nb("10_aggregation.ipynb", cells)


# ======================================================================
# 11_hybrid_fusion.ipynb — Method 10 (Bansal-style hybrid)
#
# Pipeline:
#   handcrafted (HOG + Color hist + GLCM)
#   + ABCD clinical features (asymmetry, border, color, diameter)
#   + deep features (penultimate layer of one or more trained CNNs)
#   -> StandardScaler -> TruncatedSVD
#   -> SMOTE-Tomek balancing on train features
#   -> MLP and XGBoost classifiers; pick whichever wins on val
#   -> tune decision threshold on val; evaluate on test
#
# Defaults to ResNet50 features; auto-falls back to whichever CNN
# checkpoints are present on Drive. Robust to partial training (uses
# the latest best-F1 checkpoint, whatever epoch it stopped at).
# ======================================================================

def nb_hybrid_fusion():
    cells = [
        md("""
# 11 — Hybrid Fusion (Method 10): handcrafted + ABCD + deep + SMOTE + MLP/XGBoost

This is the **Bansal 2022 recipe** adapted to our lesion-cropped data. Concretely:

1. **Handcrafted features** — HOG + HSV color histogram + GLCM (same as the
   classical-ML pipeline in notebook 02), computed on the 128×128 downsample
   of the lesion-cropped 448 image.
2. **ABCD clinical features** — asymmetry (PCA-aligned XOR), border
   irregularity (perimeter² / 4π·area), color diversity (per-channel std in
   RGB+HSV+Lab + L-channel entropy), diameter (fitted-ellipse major axis /
   image diagonal). Mask is recomputed via Otsu on each cropped image at
   evaluation time (no extra storage required).
3. **Deep features** — penultimate layer of one or more trained CNNs. The
   notebook prefers ResNet50 by default but auto-falls back to any CNN
   checkpoint present in `MyDrive/melanoma/checkpoints/`. If multiple
   checkpoints are present, their features are concatenated
   (Bansal-style multi-CNN fusion).
4. **Scaling + PCA(500)** — standardise then reduce; PCA fit on train only.
5. **SMOTE-Tomek** — synthetic minority oversampling + Tomek-link cleanup
   on the TRAIN feature matrix only. Matches the SMOTE-Tomek protocol used
   by Naeem 2024 (SNC_Net), which reached F1 0.981 on ISIC 2019.
6. **Shallow classifier** — train an MLP (Bansal style) and an XGBoost
   classifier (Naeem style) on the balanced training features; pick the
   one with higher val F1.
7. **Threshold tuning + test evaluation** — sweep thresholds on val,
   apply the F1-maximising threshold once on test; save the standard
   `hybrid_fusion_*` outputs so the aggregation notebook ingests this
   as a new row in the comparison table.

**Expected output:** test F1 in the 0.80-0.90 band on HAM10000 binary
with the lesion-grouped split. Bansal 2022 (same dataset, same binary
task, similar recipe but image-grouped split) reports 94.9% acc.

This notebook requires `imbalanced-learn` (auto-installed below) and at
least one CNN checkpoint. **CPU-only run**, around 10–20 minutes.
"""),
        code(COLAB_PREAMBLE),
        code(SEED_BLOCK),
        code("""
# Install only what Colab doesn't already ship with.
!pip install --quiet imbalanced-learn xgboost timm 2>&1 | tail -n 1
"""),
        code("""
# --- Inspect which CNN checkpoints exist on Drive ---
import time
from pathlib import Path

# /content/local_data is reused for intermediate feature caches so a
# disconnect in the middle of this notebook doesn't waste the slow CPU work.
CACHE_DIR = Path("/content/local_data")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ARCHES = ["resnet50", "densenet121", "efficientnet_b3", "swin_tiny", "vgg16_bn", "alexnet"]
available = [a for a in ARCHES
             if (config.CHECKPOINT_DIR / f"{a}_best.pt").exists()]
if not available:
    raise RuntimeError(
        "No CNN checkpoints found in MyDrive/melanoma/checkpoints/. "
        "Run at least one of notebooks 04-09 before this hybrid fusion notebook."
    )
print("Checkpoints available for deep-feature extraction:")
for a in available:
    p = config.CHECKPOINT_DIR / f"{a}_best.pt"
    size_mb = p.stat().st_size / 1e6
    print(f"  - {a:18s}  ({size_mb:.1f} MB)")

# Use EVERY available CNN's penultimate features concatenated — this is
# the Bansal 2022 multi-CNN fusion recipe (ResNet50V2 + EfficientNet-B0
# + handcrafted -> ANN, 94.9% on HAM10000). The single-ResNet50 default
# we shipped first gave a 2048-D deep block; the all-CNN default below
# stacks up to ~13.5K dims (AlexNet 4096 + VGG 4096 + ResNet50 2048 +
# EffNetB3 1536 + DenseNet 1024 + Swin 768) which PCA then projects to
# 500 components, exactly matching the published recipe.
DEEP_FEATURE_SOURCES = available
print(f"\\nWill extract deep features from: {DEEP_FEATURE_SOURCES}")
"""),
        code("""
# --- Load arrays + indices ---
import numpy as np
from src.data import load_arrays

t0 = time.time()
X, y, ids, idx_train, idx_val, idx_test = load_arrays(config.DATA_DIR)
print(f"Loaded arrays in {time.time()-t0:.1f}s.  "
      f"X: {X.shape}  splits: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")
"""),
        code("""
# --- Handcrafted features (HOG + Color hist + GLCM) on 128x128 downsample ---
# Parallelised with joblib (A100 runtime has ~12 vCPUs); cached so a
# re-run after interruption skips this slow step.
import cv2
from joblib import Parallel, delayed
from tqdm import tqdm
from src.features import extract_all

F_hand_path = CACHE_DIR / "F_hand.npy"
if F_hand_path.exists():
    F_hand = np.load(F_hand_path)
    print(f"Loaded cached F_hand: {F_hand.shape}  (delete {F_hand_path} to recompute)")
else:
    t0 = time.time()
    # Resize first (cheap), then extract features in parallel.
    print("Resizing 10,015 images to 128x128 ...")
    X128 = np.empty((len(X), config.HOG_IMG_SIZE, config.HOG_IMG_SIZE, 3), dtype=np.uint8)
    for i in range(len(X)):
        X128[i] = cv2.resize(X[i], (config.HOG_IMG_SIZE, config.HOG_IMG_SIZE),
                              interpolation=cv2.INTER_AREA)

    # Bind hyperparameters to local variables so joblib workers receive them
    # without dragging the whole config module across the pickle boundary.
    PPC = config.HOG_PIXELS_PER_CELL
    CPB = config.HOG_CELLS_PER_BLOCK
    CHB = config.COLOR_HIST_BINS
    GD = config.GLCM_DISTANCES
    GA = config.GLCM_ANGLES

    sample = extract_all(X128[0], pixels_per_cell=PPC, cells_per_block=CPB,
                         color_bins=CHB, glcm_distances=GD, glcm_angles=GA)
    print(f"Handcrafted feature dim per image: {sample.shape[0]}  "
          f"(HOG + Color + GLCM)")
    print("Computing handcrafted features in parallel (threading backend) ...")

    feats_list = Parallel(n_jobs=-1, verbose=5, backend="threading", batch_size=128)(
        delayed(extract_all)(X128[i],
                             pixels_per_cell=PPC,
                             cells_per_block=CPB,
                             color_bins=CHB,
                             glcm_distances=GD,
                             glcm_angles=GA)
        for i in range(len(X128))
    )
    F_hand = np.asarray(feats_list, dtype=np.float32)
    del X128, feats_list  # free RAM

    np.save(F_hand_path, F_hand)
    print(f"F_hand: {F_hand.shape}  (computed in {time.time()-t0:.1f}s, "
          f"cached to {F_hand_path})")
"""),
        code("""
# --- ABCD clinical features (Otsu mask recomputed per cropped image) ---
# Also parallelised + cached.
import cv2
from joblib import Parallel, delayed
from src.abcd import abcd_features, ABCD_FEATURE_NAMES
from src.segmentation import segment_lesion

F_abcd_path = CACHE_DIR / "F_abcd.npy"
if F_abcd_path.exists():
    F_abcd = np.load(F_abcd_path)
    print(f"Loaded cached F_abcd: {F_abcd.shape}  (delete {F_abcd_path} to recompute)")
else:
    t0 = time.time()
    from src.abcd import abcd_features_from_image
    print(f"ABCD feature names ({len(ABCD_FEATURE_NAMES)}): {ABCD_FEATURE_NAMES}")
    print(f"Computing ABCD features in parallel (threading) on {len(X)} images ...")

    results = Parallel(n_jobs=-1, verbose=5, backend="threading", batch_size=128)(
        delayed(abcd_features_from_image)(X[i]) for i in range(len(X))
    )
    F_abcd = np.asarray([r[0] for r in results], dtype=np.float32)
    n_mask_fail = sum(1 for r in results if r[1])
    np.save(F_abcd_path, F_abcd)
    print(f"F_abcd: {F_abcd.shape}  (Otsu fallback in {n_mask_fail} crops, "
          f"computed in {time.time()-t0:.1f}s, cached to {F_abcd_path})")
"""),
        code("""
# --- Deep features: penultimate layer of each available CNN ---
# Cached so a re-run after interruption skips the GPU pass.
import torch
from torch.utils.data import DataLoader
from src.data import HAMDataset, make_eval_transform
from src.models import BUILDERS, chop_head_for_features, PENULTIMATE_DIMS

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device for deep-feature extraction: {device}")

@torch.no_grad()
def extract_deep_features(arch):
    cache_path = CACHE_DIR / f"F_deep_{arch}.npy"
    if cache_path.exists():
        feats = np.load(cache_path)
        print(f"  loaded cached {arch}: {feats.shape}")
        return feats
    t0 = time.time()
    cfg = config.ARCH_CONFIG[arch]
    model = BUILDERS[arch](num_classes=2, pretrained=False).to(device)
    ckpt = torch.load(config.CHECKPOINT_DIR / f"{arch}_best.pt", map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model = chop_head_for_features(model, arch).to(device).eval()
    eval_tf = make_eval_transform(cfg["input_size"])
    ds_all = HAMDataset(X, y, np.arange(len(X)), eval_tf)
    ld = DataLoader(ds_all, batch_size=cfg["batch_size"], shuffle=False,
                    num_workers=2, pin_memory=True)
    feats = np.empty((len(X), PENULTIMATE_DIMS[arch]), dtype=np.float32)
    off = 0
    for x_batch, _ in tqdm(ld, desc=f"deep features [{arch}]"):
        x_batch = x_batch.to(device, non_blocking=True)
        out = model(x_batch)
        if out.dim() > 2:
            out = out.flatten(start_dim=1)
        n = out.shape[0]
        feats[off:off + n] = out.cpu().numpy()
        off += n
    np.save(cache_path, feats)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"  {arch}: {feats.shape}  (extracted in {time.time()-t0:.1f}s, "
          f"cached to {cache_path})")
    return feats

F_deep_parts = []
deep_names = []
for arch in DEEP_FEATURE_SOURCES:
    feats = extract_deep_features(arch)
    F_deep_parts.append(feats)
    deep_names.append(f"{arch}({PENULTIMATE_DIMS[arch]})")
F_deep = np.concatenate(F_deep_parts, axis=1)
print(f"F_deep ({' + '.join(deep_names)}): {F_deep.shape}")
"""),
        code("""
# --- Concatenate all feature blocks ---
F_all = np.concatenate([F_hand, F_abcd, F_deep], axis=1).astype(np.float32)
print(f"F_all: {F_all.shape}  "
      f"(hand={F_hand.shape[1]}, abcd={F_abcd.shape[1]}, deep={F_deep.shape[1]})  "
      f"size in RAM: {F_all.nbytes / 1e9:.2f} GB")
"""),
        code("""
# --- StandardScaler (no centering) + TruncatedSVD, fit on TRAIN only ---
# Why TruncatedSVD instead of PCA:
#   With the Bansal multi-CNN deep block, F_all is (~10K, ~16K) — too
#   wide for sklearn's randomized-PCA path on Colab CPU (hangs > 8 min).
#   TruncatedSVD skips the zero-centering step that makes the matrix
#   dense and slow; it finishes in ~30-90s on the same shape with
#   negligible quality loss for the downstream MLP/XGBoost.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

t0 = time.time()
print("Fitting StandardScaler (with_mean=False) on train ...")
scaler = StandardScaler(with_mean=False).fit(F_all[idx_train])
F_scaled = scaler.transform(F_all).astype(np.float32)
print(f"  scaler done in {time.time()-t0:.1f}s.  F_scaled: {F_scaled.shape}")

# Cap the component count at min(300, n_features-1, n_train-1).
# 300 is enough to capture the variance once the multi-CNN deep block
# stacks 13.5K features through PCA — a 500-component target was the
# old default that triggered the hang.
n_comp = int(min(300, F_scaled.shape[1] - 1, len(idx_train) - 1))
t0 = time.time()
print(f"Fitting TruncatedSVD(n={n_comp}, n_iter=5) on train ...")
tsvd = TruncatedSVD(
    n_components=n_comp,
    n_iter=5,
    random_state=config.SEED,
).fit(F_scaled[idx_train])
F_pca = tsvd.transform(F_scaled).astype(np.float32)
print(f"TruncatedSVD done in {time.time()-t0:.1f}s.  kept {n_comp} components, "
      f"explained variance = {tsvd.explained_variance_ratio_.sum():.3f}")
print(f"F_pca: {F_pca.shape}")
"""),
        code("""
# --- SMOTE-Tomek on TRAIN ONLY ---
# (val/test must remain at the natural 1:8 distribution for honest evaluation.)
from imblearn.combine import SMOTETomek

t0 = time.time()
Xtr_pca = F_pca[idx_train]
ytr     = y[idx_train]
print(f"Before SMOTE-Tomek: train={Xtr_pca.shape}, "
      f"class counts={dict(zip(*np.unique(ytr, return_counts=True)))}")
print("Running SMOTE-Tomek ...")

smt = SMOTETomek(random_state=config.SEED, n_jobs=-1)
Xtr_bal, ytr_bal = smt.fit_resample(Xtr_pca, ytr)
print(f"After  SMOTE-Tomek: train={Xtr_bal.shape}, "
      f"class counts={dict(zip(*np.unique(ytr_bal, return_counts=True)))}  "
      f"({time.time()-t0:.1f}s)")

Xv_pca = F_pca[idx_val];  yv = y[idx_val]
Xt_pca = F_pca[idx_test]; yt = y[idx_test]
"""),
        code("""
# --- Train MLP (Bansal-style ANN) ---
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score

print("Training MLP [256 -> 128] with early stopping ...")
t0 = time.time()
mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=128,
    learning_rate_init=1e-3,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=config.SEED,
    verbose=False,
)
mlp.fit(Xtr_bal, ytr_bal)
mlp_train_time = time.time() - t0

pv_mlp = mlp.predict_proba(Xv_pca)[:, 1]
pt_mlp = mlp.predict_proba(Xt_pca)[:, 1]
print(f"MLP trained in {mlp_train_time:.1f}s ({mlp.n_iter_} iters).  "
      f"val AUC = {roc_auc_score(yv, pv_mlp):.4f}")
"""),
        code("""
# --- Train XGBoost (Naeem-style booster) ---
import xgboost as xgb

print("Training XGBoost (500 trees, depth=6, hist) ...")
t0 = time.time()
xgbc = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=config.SEED,
    n_jobs=-1,
    tree_method="hist",
)
xgbc.fit(Xtr_bal, ytr_bal,
         eval_set=[(Xv_pca, yv)],
         verbose=False)
xgb_train_time = time.time() - t0

pv_xgb = xgbc.predict_proba(Xv_pca)[:, 1]
pt_xgb = xgbc.predict_proba(Xt_pca)[:, 1]
print(f"XGBoost trained in {xgb_train_time:.1f}s.  val AUC = {roc_auc_score(yv, pv_xgb):.4f}")
"""),
        code("""
# --- Pick the better classifier on val F1 (at tuned threshold) ---
from src.training import tune_threshold

best_t_mlp, val_f1_mlp = tune_threshold(yv, pv_mlp)
best_t_xgb, val_f1_xgb = tune_threshold(yv, pv_xgb)
print(f"MLP    : val F1 = {val_f1_mlp:.4f} at t = {best_t_mlp:.3f}")
print(f"XGBoost: val F1 = {val_f1_xgb:.4f} at t = {best_t_xgb:.3f}")

if val_f1_mlp >= val_f1_xgb:
    chosen, pv, pt, best_t = "MLP", pv_mlp, pt_mlp, best_t_mlp
    train_time = mlp_train_time
else:
    chosen, pv, pt, best_t = "XGBoost", pv_xgb, pt_xgb, best_t_xgb
    train_time = xgb_train_time

print(f"\\nChosen classifier: {chosen}  (val F1 = {max(val_f1_mlp, val_f1_xgb):.4f})")
"""),
        code("""
# --- Final test evaluation with the chosen classifier + tuned threshold ---
import time
from src.evaluation import save_standard_outputs, compute_metrics

t0 = time.time()
yp_test = (pt > best_t).astype(int)
inf_ms_per_image = (time.time() - t0) * 1000.0 / max(1, len(pt))

hp = dict(
    classifier=chosen,
    deep_feature_sources=list(DEEP_FEATURE_SOURCES),
    handcrafted_dim=int(F_hand.shape[1]),
    abcd_dim=int(F_abcd.shape[1]),
    deep_dim=int(F_deep.shape[1]),
    pca_components=int(n_comp),
    pca_explained_variance=float(tsvd.explained_variance_ratio_.sum()),
    balancing="SMOTE-Tomek (train only)",
    train_class_counts_before=dict(zip(*[a.tolist() for a in np.unique(ytr, return_counts=True)])),
    train_class_counts_after=dict(zip(*[a.tolist() for a in np.unique(ytr_bal, return_counts=True)])),
    decision_threshold=float(best_t),
    threshold_selection="argmax F1 on validation set",
    val_f1_mlp=float(val_f1_mlp),
    val_f1_xgb=float(val_f1_xgb),
    notes="Bansal2022 + Naeem2024 inspired hybrid fusion (handcrafted + ABCD + deep features).",
)

metrics = save_standard_outputs(
    method_name="hybrid_fusion",
    results_dir=config.RESULTS_DIR,
    y_true=yt,
    y_pred=yp_test,
    y_prob=pt,
    ids=ids[idx_test],
    hyperparameters=hp,
    train_time_sec=float(train_time),
    inference_time_per_image_ms=float(inf_ms_per_image),
)

# Save val probs too — useful for later soft-voting against the CNN ensemble.
import pandas as pd
pd.DataFrame({
    "image_id": ids[idx_val], "y_true": yv,
    "y_prob_mlp": pv_mlp, "y_prob_xgb": pv_xgb,
    "y_prob_chosen": pv, "best_t": best_t,
}).to_csv(config.RESULTS_DIR / "hybrid_fusion_val_predictions.csv", index=False)

print({k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))})
"""),
        md("""
---

## What this produces

- `results/hybrid_fusion_metrics.json` — headline metrics (acc/prec/recall/F1/AUC).
- `results/hybrid_fusion_predictions.csv` — per-test-image probabilities and the
  tuned-threshold prediction.
- `results/hybrid_fusion_val_predictions.csv` — per-val-image probabilities from
  both MLP and XGBoost, so you can also try a CNN-ensemble + hybrid soft-vote
  if you want one more row.
- `results/hybrid_fusion_confusion_matrix.png`, `_roc_curve.png`.

After this notebook finishes, re-run **`10_aggregation.ipynb`** — it auto-detects
the new `hybrid_fusion` entry and appends it to the comparison table. The IEEE
paper draft (`paper/paper_draft.tex`) then gets a 10th row.

## If you also want a hybrid + CNN-ensemble super-fusion

After both 10_aggregation and this notebook have finished, you can compute an
additional soft-vote between `ensemble` (6 CNNs) and `hybrid_fusion` (this
notebook) by averaging the two test probabilities at their respective tuned
thresholds. That's a single extra cell.
"""),
    ]
    write_nb("11_hybrid_fusion.ipynb", cells)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    # Done in earlier sessions, kept stable:
    nb_data_setup()
    # nb_baseline()    # 01 — executed version is on origin/main; uncomment to regenerate
    # nb_classical()   # 02 — executed version is on origin/main; uncomment to regenerate
    nb_effnet()        # 03 — legacy EfficientNet-B0; kept for old ablation row

    # Phase 2 — full benchmark
    nb_modern_cnn("alexnet")          # 04
    nb_modern_cnn("vgg16_bn")         # 05
    nb_modern_cnn("resnet50")         # 06 (Phase 1 proof-of-concept)
    nb_modern_cnn("efficientnet_b3")  # 07
    nb_modern_cnn("densenet121")      # 08
    nb_modern_cnn("swin_tiny")        # 09
    nb_aggregation_v2()               # 10 — comparison + ensemble + ablations
    nb_hybrid_fusion()                # 11 — Method 10: Bansal-style hybrid fusion

    print("All notebooks generated.")
