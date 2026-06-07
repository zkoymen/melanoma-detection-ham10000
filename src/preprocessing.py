"""
Preprocessing utilities used across all notebooks.

Pipeline applied once during 00_data_setup:
    hair removal (DullRazor) -> lesion segmentation (Otsu+morphology in LAB-L)
    -> crop to lesion bbox + margin -> resize square -> BGR -> RGB uint8

The atomic helpers (`remove_hair`, `resize_image`) are pure so they can be
imported in isolation. `preprocess_for_storage` composes them and is the
function 00_data_setup calls per image.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.segmentation import (
    DEFAULT_BORDER_FRAC,
    DEFAULT_FALLBACK_FRAC,
    DEFAULT_MARGIN_FRAC,
    DEFAULT_MAX_AREA_FRAC,
    DEFAULT_MIN_AREA_FRAC,
    segment_and_crop,
)


def remove_hair(img_bgr: np.ndarray) -> np.ndarray:
    """DullRazor-inspired hair removal.

    Grayscale -> morphological black-hat with a cross-shaped 17x17 kernel
    (highlights thin dark structures, i.e. hair) -> threshold to a binary
    mask -> Telea inpainting fills the masked pixels with their neighbourhood.
    Works in BGR.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img_bgr, mask, 1, cv2.INPAINT_TELEA)


def resize_image(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """Resize to a square (size, size). Uses INTER_AREA for downscaling."""
    h, w = img_bgr.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    return cv2.resize(img_bgr, (size, size), interpolation=interp)


def preprocess_for_storage(
    img_bgr: np.ndarray,
    size: int,
    do_hair_removal: bool = True,
    do_segmentation: bool = True,
    seg_margin_frac: float = DEFAULT_MARGIN_FRAC,
    seg_border_frac: float = DEFAULT_BORDER_FRAC,
    seg_min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
    seg_max_area_frac: float = DEFAULT_MAX_AREA_FRAC,
    seg_fallback_frac: float = DEFAULT_FALLBACK_FRAC,
) -> tuple[np.ndarray, bool]:
    """One-shot preprocessing applied during dataset build.

    Returns:
        rgb_uint8:     (size, size, 3) RGB uint8 image, lesion-cropped.
        used_fallback: True if Otsu segmentation was rejected and a centred
                       fallback crop was used instead. The caller can sum
                       this across the dataset to report the segmentation
                       success rate.
    """
    if do_hair_removal:
        img_bgr = remove_hair(img_bgr)

    used_fallback = False
    if do_segmentation:
        img_bgr, used_fallback = segment_and_crop(
            img_bgr,
            margin_frac=seg_margin_frac,
            border_frac=seg_border_frac,
            min_area_frac=seg_min_area_frac,
            max_area_frac=seg_max_area_frac,
            fallback_frac=seg_fallback_frac,
        )

    img_bgr = resize_image(img_bgr, size)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return rgb, used_fallback
