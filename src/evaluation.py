"""
Standardised metrics and plotting helpers.

Every method calls `compute_metrics` and `save_standard_outputs`, which
guarantees a uniform `results/<method>_*` file layout that the aggregation
notebook can ingest blindly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_prob)),
    }


def plot_confusion(y_true, y_pred, out_path: Path, class_names=("non-mel", "mel")):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], class_names)
    ax.set_yticks([0, 1], class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc(y_true, y_prob, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_standard_outputs(method_name: str,
                          results_dir: Path,
                          y_true,
                          y_pred,
                          y_prob,
                          ids,
                          hyperparameters: dict,
                          train_time_sec: float,
                          inference_time_per_image_ms: float):
    """Write metrics.json, confusion_matrix.png, roc_curve.png, predictions.csv."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["train_time_sec"] = float(train_time_sec)
    metrics["inference_time_per_image_ms"] = float(inference_time_per_image_ms)
    metrics["hyperparameters"] = hyperparameters

    (results_dir / f"{method_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )
    plot_confusion(y_true, y_pred, results_dir / f"{method_name}_confusion_matrix.png")
    plot_roc(y_true, y_prob, results_dir / f"{method_name}_roc_curve.png")
    pd.DataFrame({
        "image_id": ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }).to_csv(results_dir / f"{method_name}_predictions.csv", index=False)
    return metrics


def time_inference(predict_fn, x_sample, n_iter: int = 5) -> float:
    """Average per-image inference latency in milliseconds."""
    n = len(x_sample)
    durations = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        predict_fn(x_sample)
        durations.append((time.perf_counter() - t0) * 1000.0 / n)
    return float(np.median(durations))
