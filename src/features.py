"""
Hand-crafted features for the classical-ML method.

Concatenated descriptor = HOG + HSV color histogram + GLCM stats.
Vectors are ~3500-D before PCA.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import hog, graycomatrix, graycoprops


def hog_features(img_rgb: np.ndarray,
                 pixels_per_cell=(16, 16),
                 cells_per_block=(2, 2)) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return hog(gray,
               orientations=9,
               pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block,
               block_norm="L2-Hys",
               feature_vector=True).astype(np.float32)


def color_histogram(img_rgb: np.ndarray, bins: int = 8) -> np.ndarray:
    """HSV histogram, `bins` per channel, L1-normalized, flattened."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None,
                        [bins, bins, bins],
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(np.float32)


def glcm_features(img_rgb: np.ndarray,
                  distances=(1,),
                  angles=(0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)) -> np.ndarray:
    """Contrast, homogeneity, energy, correlation -> 4 * |dist| * |ang| values."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = (gray // 32).astype(np.uint8)  # 8-level quantization for speed
    glcm = graycomatrix(gray,
                        distances=list(distances),
                        angles=list(angles),
                        levels=8,
                        symmetric=True,
                        normed=True)
    feats = []
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        feats.append(graycoprops(glcm, prop).flatten())
    return np.concatenate(feats).astype(np.float32)


def extract_all(img_rgb: np.ndarray,
                pixels_per_cell=(16, 16),
                cells_per_block=(2, 2),
                color_bins: int = 8,
                glcm_distances=(1,),
                glcm_angles=(0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)) -> np.ndarray:
    """Concatenate the three descriptors into one feature vector."""
    return np.concatenate([
        hog_features(img_rgb, pixels_per_cell, cells_per_block),
        color_histogram(img_rgb, color_bins),
        glcm_features(img_rgb, glcm_distances, glcm_angles),
    ])
