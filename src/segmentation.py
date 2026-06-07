"""
Lesion segmentation and cropping utilities.

A classical Otsu-on-LAB-L pipeline that crops each image to the lesion
bounding box (with a small margin) before resizing to the stored array.
Pure DIP, no learned model.

Pipeline per image (called from src.preprocessing.preprocess_for_storage):
    1. (Upstream) DullRazor hair removal.
    2. Convert BGR -> LAB, take L channel, Gaussian blur to denoise.
    3. Otsu threshold on inverted L (lesion is darker than skin in LAB-L,
       so inversion puts the lesion in the "bright" class).
    4. Morphological close+open with an ellipse kernel; zero out a thin
       border to suppress dermoscope-ring artefacts.
    5. Largest connected component as the lesion mask.
    6. Sanity gates (area fraction within [min, max]; the mask must not
       touch all four edges). If any gate fails, fall back to a centred
       square crop.
    7. Bounding box of the kept component, expanded by `margin_frac`,
       clipped to image bounds. Return the cropped BGR image plus the
       bbox so the caller can log how often the fallback triggered.

Everything below operates on BGR uint8 arrays for consistency with
`src.preprocessing.remove_hair`.
"""
from __future__ import annotations

import cv2
import numpy as np


DEFAULT_BORDER_FRAC = 0.04
DEFAULT_MIN_AREA_FRAC = 0.02
DEFAULT_MAX_AREA_FRAC = 0.85
DEFAULT_MARGIN_FRAC = 0.15
DEFAULT_FALLBACK_FRAC = 0.80


def _zero_border(mask: np.ndarray, border_frac: float) -> np.ndarray:
    h, w = mask.shape
    bh = max(1, int(h * border_frac))
    bw = max(1, int(w * border_frac))
    mask = mask.copy()
    mask[:bh, :] = 0
    mask[-bh:, :] = 0
    mask[:, :bw] = 0
    mask[:, -bw:] = 0
    return mask


def _touches_all_edges(mask: np.ndarray) -> bool:
    return (mask[0, :].any() and mask[-1, :].any()
            and mask[:, 0].any() and mask[:, -1].any())


def _fallback_center_bbox(h: int, w: int, frac: float) -> tuple[int, int, int, int]:
    side = int(min(h, w) * frac)
    x = (w - side) // 2
    y = (h - side) // 2
    return x, y, side, side


def segment_lesion(
    img_bgr: np.ndarray,
    border_frac: float = DEFAULT_BORDER_FRAC,
    min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
    max_area_frac: float = DEFAULT_MAX_AREA_FRAC,
    fallback_frac: float = DEFAULT_FALLBACK_FRAC,
) -> tuple[np.ndarray, tuple[int, int, int, int], bool]:
    """Compute a binary lesion mask and its bounding box.

    Returns:
        mask:          (H, W) uint8 in {0, 255}.
        bbox:          (x, y, w, h) of the kept component (or fallback box).
        used_fallback: True if the Otsu+CC step was rejected and a centred
                       square crop was substituted instead.
    """
    assert img_bgr.ndim == 3 and img_bgr.shape[2] == 3, "expected BGR HxWx3"
    h, w = img_bgr.shape[:2]
    total = float(h * w)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    L_blur = cv2.GaussianBlur(L, (5, 5), 0)

    _, mask = cv2.threshold(255 - L_blur, 0, 255,
                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = _zero_border(mask, border_frac)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        bbox = _fallback_center_bbox(h, w, fallback_frac)
        fb_mask = np.zeros((h, w), dtype=np.uint8)
        x, y, bw, bh = bbox
        fb_mask[y:y + bh, x:x + bw] = 255
        return fb_mask, bbox, True

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    area = float(areas[largest_label - 1])

    cleaned = (labels == largest_label).astype(np.uint8) * 255

    if (area / total) < min_area_frac or (area / total) > max_area_frac or _touches_all_edges(cleaned):
        bbox = _fallback_center_bbox(h, w, fallback_frac)
        fb_mask = np.zeros((h, w), dtype=np.uint8)
        x, y, bw, bh = bbox
        fb_mask[y:y + bh, x:x + bw] = 255
        return fb_mask, bbox, True

    x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])
    bw = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    bh = int(stats[largest_label, cv2.CC_STAT_HEIGHT])
    return cleaned, (x, y, bw, bh), False


def crop_to_lesion_bbox(
    img: np.ndarray,
    bbox: tuple[int, int, int, int],
    margin_frac: float = DEFAULT_MARGIN_FRAC,
) -> np.ndarray:
    """Crop `img` to a margin-expanded bbox. Works for both BGR and RGB."""
    h, w = img.shape[:2]
    x, y, bw, bh = bbox
    mx = int(round(bw * margin_frac))
    my = int(round(bh * margin_frac))
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + bw + mx)
    y2 = min(h, y + bh + my)
    return img[y1:y2, x1:x2]


def segment_and_crop(
    img_bgr: np.ndarray,
    margin_frac: float = DEFAULT_MARGIN_FRAC,
    border_frac: float = DEFAULT_BORDER_FRAC,
    min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
    max_area_frac: float = DEFAULT_MAX_AREA_FRAC,
    fallback_frac: float = DEFAULT_FALLBACK_FRAC,
) -> tuple[np.ndarray, bool]:
    """End-to-end: segment the lesion, crop to bbox + margin. Returns BGR + fallback flag."""
    _, bbox, used_fallback = segment_lesion(
        img_bgr,
        border_frac=border_frac,
        min_area_frac=min_area_frac,
        max_area_frac=max_area_frac,
        fallback_frac=fallback_frac,
    )
    cropped = crop_to_lesion_bbox(img_bgr, bbox, margin_frac=margin_frac)
    return cropped, used_fallback
