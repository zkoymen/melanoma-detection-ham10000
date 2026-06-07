"""
Project-wide configuration.

Paths default to a Google Colab layout (Drive mounted at /content/drive). They
can be overridden by environment variables for local runs:

    MELANOMA_DRIVE_ROOT    base folder that holds data/, results/, checkpoints/
    MELANOMA_LOCAL_SCRATCH fast local SSD used to unzip the raw Kaggle images

Expected folder layout under DRIVE_ROOT:

    melanoma/
        kaggle.json         Kaggle API token (download from kaggle.com/account)
        data/               processed .npy arrays (created automatically)
        results/            metrics, predictions, plots (created automatically)
        checkpoints/        trained model weights (created automatically)
"""

import os
from pathlib import Path


# Reproducibility
SEED = 42


# Drive layout (override via environment variable for local runs)
DRIVE_ROOT = Path(os.environ.get(
    "MELANOMA_DRIVE_ROOT", "/content/drive/MyDrive/melanoma"))
KAGGLE_JSON_PATH = DRIVE_ROOT / "kaggle.json"
DATA_DIR = DRIVE_ROOT / "data"
RESULTS_DIR = DRIVE_ROOT / "results"
PAPER_DIR = DRIVE_ROOT / "paper"
CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints"


# Local scratch space — fast SSD, used to unzip the raw Kaggle images.
LOCAL_SCRATCH = Path(os.environ.get(
    "MELANOMA_LOCAL_SCRATCH", "/content/ham10000"))


# Kaggle dataset slug
KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"


# Image sizes used at different stages
IMG_SIZE = 448              # stored arrays — lesion-cropped square
HOG_IMG_SIZE = 128          # smaller for HOG to keep vectors manageable
BASELINE_IMG_SIZE = 64      # tiny for the raw-pixel logistic baseline


# Segmentation (Otsu on inverted LAB-L + morphology + largest CC + sanity gate)
SEG_BORDER_FRAC = 0.04
SEG_MIN_AREA_FRAC = 0.02
SEG_MAX_AREA_FRAC = 0.85
SEG_MARGIN_FRAC = 0.15
SEG_FALLBACK_FRAC = 0.80


# Train / val / test split fractions (lesion-grouped, stratified)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


# Shared CNN training pipeline (methods 3–8)
CNN_HEAD_LR = 1e-3              # stage 1 learning rate (head only)
CNN_HEAD_EPOCHS = 3
CNN_FT_LR_HEAD = 3e-4           # stage 2 head learning rate
CNN_FT_LR_BACKBONE = 3e-5       # stage 2 backbone learning rate (10x lower)
CNN_FT_EPOCHS = 25
CNN_WARMUP_EPOCHS = 3
CNN_EARLY_STOP_PATIENCE = 7     # early stopping on val F1
CNN_WEIGHT_DECAY = 1e-4
CNN_EMA_DECAY = 0.999           # exponential moving average of weights


# Focal loss with class-balanced alpha + weighted sampler
FOCAL_GAMMA = 2.0
FOCAL_BETA = 0.999              # effective number of samples (Cui et al. 2019)
USE_WEIGHTED_SAMPLER = True     # WeightedRandomSampler -> roughly 50/50 batches


# Conservative augmentation tuned for dermoscopic images. Mixup, CutMix and
# RandAugment are intentionally disabled because they distort or replace the
# lesion pixels that the ABCD diagnostic criteria depend on.
AUG_ROTATION_DEG = 15
AUG_COLOR_BRIGHTNESS = 0.10
AUG_COLOR_CONTRAST = 0.10
AUG_COLOR_SATURATION = 0.05
AUG_COLOR_HUE = 0.02
AUG_CROP_SCALE_MIN = 0.85
AUG_CROP_SCALE_MAX = 1.00
AUG_RANDOM_ERASING_P = 0.10
AUG_RANDOM_ERASING_SCALE = (0.02, 0.10)

RANDAUG_NUM_OPS = 0
RANDAUG_MAGNITUDE = 0
MIXUP_ALPHA = 0.0
CUTMIX_ALPHA = 0.0
MIXUP_PROB = 0.0
CUTMIX_PROB = 0.0


# Per-architecture (input size, batch size).
ARCH_CONFIG = {
    "alexnet":         {"input_size": 224, "batch_size": 64},
    "vgg16_bn":        {"input_size": 224, "batch_size": 16},
    "resnet50":        {"input_size": 320, "batch_size": 32},
    "efficientnet_b3": {"input_size": 320, "batch_size": 24},
    "densenet121":     {"input_size": 320, "batch_size": 32},
    "swin_tiny":       {"input_size": 224, "batch_size": 32},
}


# Test-time augmentation
TTA_TRANSFORMS = (
    "identity", "hflip", "vflip", "hvflip",
    "rot90", "rot180", "rot270", "hflip_rot90",
)


# ImageNet normalization stats
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Classical ML (HOG + Color + GLCM + PCA + SVM)
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
COLOR_HIST_BINS = 8
GLCM_DISTANCES = (1,)
GLCM_ANGLES = (0.0, 0.7853981633974483, 1.5707963267948966, 2.356194490192345)
PCA_COMPONENTS = 200


# Class label mapping (binary)
LABEL_NAMES = ("non_melanoma", "melanoma")
POSITIVE_CLASS = "mel"


# EfficientNet-B0 hyperparameters used by notebook 03 only. The main CNN
# benchmark (notebooks 04–09) uses the CNN_* block above plus ARCH_CONFIG.
EFFNET_BATCH_SIZE = 32
EFFNET_HEAD_LR = 1e-3
EFFNET_HEAD_EPOCHS = 5
EFFNET_FT_LR = 1e-4
EFFNET_FT_EPOCHS = 15
EFFNET_EARLY_STOP_PATIENCE = 5
EFFNET_UNFREEZE_LAST_N_BLOCKS = 2
EFFNET_CW_POWER = 0.5


def ensure_drive_dirs():
    """Create the Drive folders if they don't yet exist."""
    for d in (DATA_DIR, RESULTS_DIR, PAPER_DIR, CHECKPOINT_DIR):
        d.mkdir(parents=True, exist_ok=True)
