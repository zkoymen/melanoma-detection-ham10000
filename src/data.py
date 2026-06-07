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
        transforms.ColorJitter(brightness=brightness, contrast=contrast,
                               saturation=saturation, hue=hue),
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
