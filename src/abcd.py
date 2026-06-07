"""
ABCD clinical-rule features for dermoscopic lesions (Stolz 1994).

The four pillars of clinical visual diagnosis:
    A - Asymmetry  (PCA-aligned XOR of mask against its flips)
    B - Border     (perimeter^2 / 4*pi*Area, compactness / irregularity)
    C - Color      (per-channel std in RGB/HSV/Lab inside the mask plus
                    L-channel histogram entropy as a colour-variety proxy)
    D - Diameter   (major-axis length of fitted ellipse / image diagonal)

Features are computed from the same Otsu mask produced by `src.segmentation`.
Each function is robust to degenerate masks (empty, single-pixel, etc.) and
returns sensible defaults instead of NaN. `abcd_features(img_rgb, mask)` is
the entry point and returns a single 13-D float32 vector.
"""
from __future__ import annotations

import cv2
import numpy as np


# ============================================================================
# A — Asymmetry
# ============================================================================

def _principal_angle_deg(mask: np.ndarray) -> float:
    """Orientation of the major axis (degrees) via image moments."""
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] < 1e-6:
        return 0.0
    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    mu11 = moments["mu11"] / moments["m00"]
    return float(0.5 * np.degrees(np.arctan2(2.0 * mu11, mu20 - mu02 + 1e-12)))


def _rotate_to_principal(mask: np.ndarray) -> np.ndarray:
    """Rotate mask so its major axis is vertical (Y-axis aligned)."""
    angle = _principal_angle_deg(mask)
    h, w = mask.shape
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(mask.astype(np.uint8), M, (w, h),
                          flags=cv2.INTER_NEAREST, borderValue=0)


def asymmetry_pca(mask: np.ndarray) -> float:
    """0 = perfect symmetry, 1 = max asymmetry. Mean of H and V XOR after PCA-align."""
    aligned = _rotate_to_principal(mask) > 0
    total = aligned.sum()
    if total < 10:
        return 1.0
    h_flip = aligned[::-1, :]
    v_flip = aligned[:, ::-1]
    h_xor = np.logical_xor(aligned, h_flip).sum()
    v_xor = np.logical_xor(aligned, v_flip).sum()
    return float((h_xor + v_xor) / (2.0 * total))


# ============================================================================
# B — Border irregularity
# ============================================================================

def border_irregularity(mask: np.ndarray) -> float:
    """Compactness index. 1.0 = perfect circle, higher = more irregular."""
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 1.0:
        return 0.0
    perim = cv2.arcLength(cnt, closed=True)
    return float((perim * perim) / (4.0 * np.pi * area))


# ============================================================================
# C — Color diversity inside the lesion
# ============================================================================

def color_diversity(img_rgb: np.ndarray, mask: np.ndarray) -> list[float]:
    """Per-channel std in RGB+HSV+Lab (9-D) plus L-channel histogram entropy (1-D)."""
    mask_bool = mask > 0
    if not mask_bool.any():
        return [0.0] * 10

    rgb = img_rgb
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    feats: list[float] = []
    for arr in (rgb, hsv, lab):
        for c in range(3):
            vals = arr[:, :, c][mask_bool].astype(np.float32)
            feats.append(float(vals.std()))

    # L-channel intensity entropy as a colour-variety proxy
    L_vals = lab[:, :, 0][mask_bool]
    hist, _ = np.histogram(L_vals, bins=16, range=(0, 255), density=True)
    hist = hist + 1e-8
    entropy = float(-(hist * np.log2(hist)).sum())
    feats.append(entropy)
    return feats


# ============================================================================
# D — Diameter
# ============================================================================

def diameter_ratio(mask: np.ndarray) -> float:
    """Major-axis length of the fitted ellipse divided by image diagonal."""
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        # fitEllipse needs >= 5 points; fall back to bounding-box diagonal
        x, y, w, h = cv2.boundingRect(cnt)
        major = np.sqrt(w * w + h * h)
    else:
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        major = max(MA, ma)
    img_h, img_w = mask.shape
    img_diag = float(np.sqrt(img_h * img_h + img_w * img_w))
    if img_diag < 1.0:
        return 0.0
    return float(major / img_diag)


# ============================================================================
# Public API
# ============================================================================

ABCD_FEATURE_NAMES = (
    "asymmetry_pca",
    "border_irregularity",
    "color_std_R", "color_std_G", "color_std_B",
    "color_std_H", "color_std_S", "color_std_V",
    "color_std_L", "color_std_a", "color_std_b",
    "color_entropy_L",
    "diameter_ratio",
)


def abcd_features(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a single 13-D float32 vector of ABCD clinical features.

    img_rgb: (H, W, 3) uint8 RGB image, lesion-cropped.
    mask:    (H, W)    uint8 in {0, 255} from src.segmentation.
    """
    feats = [
        asymmetry_pca(mask),
        border_irregularity(mask),
        *color_diversity(img_rgb, mask),
        diameter_ratio(mask),
    ]
    return np.asarray(feats, dtype=np.float32)


def abcd_features_from_image(img_rgb: np.ndarray) -> tuple[np.ndarray, bool]:
    """End-to-end: recompute the Otsu mask on the cropped image, then ABCD.

    Module-level so joblib can pickle it across workers. Returns
    (13-D feature vector, fallback flag).
    """
    from src.segmentation import segment_lesion
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask, _bbox, used_fallback = segment_lesion(img_bgr)
    return abcd_features(img_rgb, mask), used_fallback
