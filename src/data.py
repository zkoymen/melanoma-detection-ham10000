"""
Dataset loading and augmentation helpers.

Two layers:
  1. `load_arrays()` reads the .npy artefacts produced by 00_data_setup,
     optionally caching them to a fast local SSD on first call.
  2. `HAMDataset`, `make_train_transform_strong`, `make_eval_transform`
     compose the conservative medical-image augmentation pipeline shared by
     every CNN notebook (flips, mild rotation, mild colour jitter, tight
     RandomResizedCrop, ImageNet normalize).

The legacy class-weight helpers (`class_weights`, `softened_class_weights`)
remain so notebook 03 still runs; new notebooks use focal loss instead.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


_LOCAL_CACHE_DIR = Path(os.environ.get(
    "MELANOMA_LOCAL_CACHE", "/content/local_data"))
_REQUIRED_FILES = (
    "X_all.npy", "y_all.npy", "ids_all.npy",
    "idx_train.npy", "idx_val.npy", "idx_test.npy",
)
_ISIC2019_X   = "X_isic2019_mel.npy"
_ISIC2019_IDS = "ids_isic2019_mel.npy"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------
# Array loading (Drive -> local SSD cache)
# ----------------------------------------------------------------------------

def _sync_local_cache(drive_dir: Path) -> Path:
    _LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for name in _REQUIRED_FILES:
        src = drive_dir / name
        dst = _LOCAL_CACHE_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"Missing on Drive: {src}")
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            print(f"  copying {name} ({src.stat().st_size / 1e6:.1f} MB) Drive -> local SSD")
            shutil.copy(src, dst)
    # Also cache ISIC 2019 arrays if they exist on Drive (optional)
    for name in (_ISIC2019_X, _ISIC2019_IDS):
        src = drive_dir / name
        dst = _LOCAL_CACHE_DIR / name
        if src.exists() and (not dst.exists() or dst.stat().st_size != src.stat().st_size):
            print(f"  copying {name} ({src.stat().st_size / 1e6:.1f} MB) Drive -> local SSD")
            shutil.copy(src, dst)
    return _LOCAL_CACHE_DIR


def load_arrays(data_dir: Path, use_local_cache: bool = True):
    """Load the preprocessed arrays produced by 00_data_setup.ipynb.

    On Colab, the default `use_local_cache=True` first copies the .npy files
    to a fast local SSD. This avoids Drive FUSE timeouts on the 1.5 GB
    X_all.npy. Set `use_local_cache=False` for local development.

    Returns:
        X     : (N, IMG_SIZE, IMG_SIZE, 3) uint8
        y     : (N,) int64
        ids   : (N,) object array of image_id strings
        idx_train, idx_val, idx_test : 1-D int arrays
    """
    drive_dir = Path(data_dir)
    if use_local_cache and Path("/content").exists():
        load_dir = _sync_local_cache(drive_dir)
    else:
        load_dir = drive_dir
    X = np.load(load_dir / "X_all.npy")
    y = np.load(load_dir / "y_all.npy")
    ids = np.load(load_dir / "ids_all.npy", allow_pickle=True)
    idx_train = np.load(load_dir / "idx_train.npy")
    idx_val = np.load(load_dir / "idx_val.npy")
    idx_test = np.load(load_dir / "idx_test.npy")
    return X, y, ids, idx_train, idx_val, idx_test


def load_arrays_extended(data_dir: Path, use_local_cache: bool = True):
    """Like load_arrays(), but appends ISIC 2019 melanoma images to training.

    If X_isic2019_mel.npy exists in data_dir, it is appended after the
    HAM10000 array. The new training indices point to the appended rows.
    Val and test indices are unchanged (pure HAM10000 — fair comparison).

    Returns same tuple as load_arrays():
        X, y, ids, idx_train, idx_val, idx_test
    """
    X, y, ids, idx_train, idx_val, idx_test = load_arrays(data_dir, use_local_cache)

    load_dir = _LOCAL_CACHE_DIR if (use_local_cache and Path("/content").exists()) else Path(data_dir)
    x19_path = load_dir / _ISIC2019_X
    ids19_path = load_dir / _ISIC2019_IDS

    if not x19_path.exists():
        print("[load_arrays_extended] X_isic2019_mel.npy not found — using HAM10000 only.")
        print("  Run notebooks/12_isic2019_prep.ipynb first to build it.")
        return X, y, ids, idx_train, idx_val, idx_test

    X19 = np.load(x19_path)          # (N19, H, H, 3) uint8
    ids19 = np.load(ids19_path, allow_pickle=True) if ids19_path.exists() else np.array([f"isic19_{i}" for i in range(len(X19))])

    # Resize X19 if its spatial dims differ from HAM10000 (e.g. 448 vs 224)
    target_size = X.shape[1]
    if X19.shape[1] != target_size or X19.shape[2] != target_size:
        import cv2
        print(f"  Resizing ISIC 2019 from {X19.shape[1]}px → {target_size}px ...")
        X19_r = np.zeros((len(X19), target_size, target_size, 3), dtype=np.uint8)
        for i, img in enumerate(X19):
            X19_r[i] = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        X19 = X19_r
        print(f"  Resize done. X19 shape: {X19.shape}")

    n_ham = len(X)
    n_isic = len(X19)

    X_ext   = np.concatenate([X, X19], axis=0)
    y_ext   = np.concatenate([y, np.ones(n_isic, dtype=y.dtype)], axis=0)
    ids_ext = np.concatenate([ids, ids19], axis=0)

    # ISIC 2019 images are training-only; append their indices after HAM10000
    new_isic_idx = np.arange(n_ham, n_ham + n_isic, dtype=idx_train.dtype)
    idx_train_ext = np.concatenate([idx_train, new_isic_idx], axis=0)

    n_mel_ham  = int((y[idx_train] == 1).sum())
    n_mel_isic = n_isic
    print(f"[load_arrays_extended] HAM10000 train mel: {n_mel_ham}  +  ISIC2019 mel: {n_mel_isic}"
          f"  =  {n_mel_ham + n_mel_isic} total mel in training")
    print(f"  Extended X shape: {X_ext.shape}")

    return X_ext, y_ext, ids_ext, idx_train_ext, idx_val, idx_test


# ----------------------------------------------------------------------------
# Class weighting (used by notebook 03 only)
# ----------------------------------------------------------------------------

def class_weights(y: np.ndarray) -> dict[int, float]:
    """`class_weight='balanced'` formula: n_samples / (n_classes * count(c))."""
    classes, counts = np.unique(y, return_counts=True)
    n = y.shape[0]
    return {int(c): float(n / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}


def softened_class_weights(y: np.ndarray, power: float = 0.5) -> dict[int, float]:
    """Sqrt-balanced weights — fully balanced weights tend to overcorrect."""
    cw = class_weights(y)
    soft = {k: v ** power for k, v in cw.items()}
    avg = np.mean(list(soft.values()))
    return {k: v / avg for k, v in soft.items()}


# ----------------------------------------------------------------------------
# Conservative medical-image augmentation pipeline (used by methods 3–8)
# ----------------------------------------------------------------------------

def make_train_transform_strong(input_size: int,
                                rotation_deg: float = 15.0,
                                brightness: float = 0.10,
                                contrast: float = 0.10,
                                saturation: float = 0.05,
                                hue: float = 0.02,
                                crop_scale_min: float = 0.85,
                                crop_scale_max: float = 1.00,
                                erasing_p: float = 0.10,
                                erasing_scale: tuple[float, float] = (0.02, 0.10),
                                randaug_num_ops: int | None = None,
                                randaug_magnitude: int | None = None):
    """Conservative augmentation tailored to dermoscopic lesion crops.

    Order: resize with a small slack, tight RandomResizedCrop, HFlip + VFlip,
    small rotation, mild ColorJitter, ToTensor + ImageNet normalize, small
    RandomErasing. The `randaug_*` arguments are accepted but ignored so older
    notebook cells that pass them continue to import without raising.
    """
    del randaug_num_ops, randaug_magnitude

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(input_size * 1.10), antialias=True),
        transforms.RandomResizedCrop(input_size,
                                     scale=(crop_scale_min, crop_scale_max),
                                     antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(rotation_deg, fill=0),
        transforms.RandomAffine(degrees=0, shear=10, fill=0),
        transforms.ColorJitter(brightness=brightness, contrast=contrast,
                               saturation=saturation, hue=hue),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=erasing_p, scale=erasing_scale,
                                 ratio=(0.3, 3.3), value=0.0),
    ])


def make_eval_transform(input_size: int):
    """Resize + ToTensor + Normalize. No randomness."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((input_size, input_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class HAMDataset(Dataset):
    """Wraps the in-memory uint8 array X with an indexed view + transform.

    X is held once in RAM (about 1.5 GB for the full uint8 set). The dataset
    only stores a view; no copies are made.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, transform):
        self.X = X
        self.y = y
        self.idx = indices
        self.transform = transform

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k = int(self.idx[i])
        img = self.X[k]
        return self.transform(img), int(self.y[k])
