"""
PyTorch training utilities.

Includes:
 - FocalLoss with class-balanced alpha (Lin 2017, Cui 2019)
 - make_weighted_sampler — roughly 50/50 batches under the 1:8 imbalance
 - EMA shadow weights (default decay 0.999)
 - WarmupCosineSchedule
 - mixup / cutmix per-batch mixer (kept available; disabled by default
   because it swaps lesion pixels with background and hurt F1 on this task)
 - tta_predict that averages probabilities over a list of TTA transforms

The legacy `train_loop` (used by notebook 03) keeps its old behaviour. The
modern `train_loop_v2` is used by methods 3–8.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler


# ----------------------------------------------------------------------------
# Loss functions
# ----------------------------------------------------------------------------

def _class_balanced_alpha(y: np.ndarray, beta: float = 0.999) -> np.ndarray:
    """Effective number of samples (Cui et al. 2019).

    For class c with count n_c, the per-class weight is
        alpha_c = (1 - beta) / (1 - beta**n_c)
    then normalized so it averages to 1.
    """
    classes, counts = np.unique(y, return_counts=True)
    eff_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(eff_num, 1e-12)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional class-balanced alpha.

    L = -alpha_c * (1 - p_c)^gamma * log(p_c)

    Supports both hard integer labels (shape [B]) and soft labels (shape
    [B, C], used by mixup/cutmix).
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.dim() == 1:
            return self._forward_hard(logits, target)
        return self._forward_soft(logits, target)

    def _forward_hard(self, logits, target):
        log_p = F.log_softmax(logits, dim=-1)
        log_p_t = log_p.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        loss = -((1 - p_t) ** self.gamma) * log_p_t
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def _forward_soft(self, logits, target):
        log_p = F.log_softmax(logits, dim=-1)
        p = log_p.exp()
        focal = (1 - p) ** self.gamma * log_p
        if self.alpha is not None:
            focal = focal * self.alpha
        loss = -(target * focal).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_focal_loss(y_train: np.ndarray, gamma: float, beta: float, device) -> FocalLoss:
    """Convenience constructor used by all CNN notebooks."""
    alpha = _class_balanced_alpha(y_train, beta=beta)
    return FocalLoss(alpha=torch.tensor(alpha, device=device), gamma=gamma).to(device)


# ----------------------------------------------------------------------------
# Sampler
# ----------------------------------------------------------------------------

def make_weighted_sampler(y_train: np.ndarray) -> WeightedRandomSampler:
    """Each sample weighted by 1 / count(its class) -> roughly 50/50 batches."""
    classes, counts = np.unique(y_train, return_counts=True)
    cls_to_w = {int(c): 1.0 / cnt for c, cnt in zip(classes, counts)}
    sample_weights = np.array([cls_to_w[int(yi)] for yi in y_train], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


# ----------------------------------------------------------------------------
# EMA
# ----------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model weights.

    Track a shadow copy that updates each optimizer step. Use
    `apply_shadow()` before evaluation, `restore()` after.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if k in self.shadow}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)

    def restore(self, model: nn.Module):
        sd = model.state_dict()
        for k, v in self.backup.items():
            sd[k].copy_(v)
        self.backup = {}


# ----------------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------------

class WarmupCosineSchedule:
    """Linear warmup for `warmup_epochs`, then cosine annealing to `min_lr`."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int,
                 total_epochs: int, base_lrs: list[float], min_lr: float = 1e-6):
        self.opt = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.base = base_lrs
        self.min_lr = min_lr

    def step(self, epoch: int):
        e = max(0, epoch - 1)
        for i, group in enumerate(self.opt.param_groups):
            base = self.base[i] if i < len(self.base) else self.base[-1]
            if e < self.warmup:
                lr = base * (e + 1) / max(1, self.warmup)
            else:
                progress = (e - self.warmup) / max(1, self.total - self.warmup)
                lr = self.min_lr + 0.5 * (base - self.min_lr) * (1 + math.cos(math.pi * progress))
            group["lr"] = lr


# ----------------------------------------------------------------------------
# Mixup / CutMix (available but disabled by default in the notebooks)
# ----------------------------------------------------------------------------

def _rand_bbox(size, lam):
    _, _, H, W = size
    cut_rat = math.sqrt(1.0 - lam)
    cw, ch = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cw // 2, 0, W)
    bby1 = np.clip(cy - ch // 2, 0, H)
    bbx2 = np.clip(cx + cw // 2, 0, W)
    bby2 = np.clip(cy + ch // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup_cutmix(x: torch.Tensor, y: torch.Tensor, num_classes: int,
                 mixup_alpha: float, cutmix_alpha: float,
                 mixup_prob: float, cutmix_prob: float):
    """Apply mixup OR cutmix OR neither, return (x, soft_target).

    `y` is hard integer labels [B]; the returned target is a soft [B, C]
    tensor. On 'no augmentation' branches the soft target is the one-hot
    encoding so the downstream loss path is identical.
    """
    one_hot = F.one_hot(y, num_classes=num_classes).float()
    r = np.random.rand()
    if r < mixup_prob:
        lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
        idx = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[idx]
        target = lam * one_hot + (1 - lam) * one_hot[idx]
        return x, target
    if r < mixup_prob + cutmix_prob:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha) if cutmix_alpha > 0 else 1.0
        idx = torch.randperm(x.size(0), device=x.device)
        bbx1, bby1, bbx2, bby2 = _rand_bbox(x.size(), lam)
        x[:, :, bby1:bby2, bbx1:bbx2] = x[idx, :, bby1:bby2, bbx1:bbx2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        target = lam * one_hot + (1 - lam) * one_hot[idx]
        return x, target
    return x, one_hot


# ----------------------------------------------------------------------------
# Train / val loops
# ----------------------------------------------------------------------------

@dataclass
class EpochLog:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)


def _val_epoch(model, loader, criterion, device, ema: EMA | None):
    if ema is not None:
        ema.apply_shadow(model)
    model.eval()
    total_loss, n = 0.0, 0
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            all_pred.append(logits.argmax(1).cpu().numpy())
            all_true.append(y.cpu().numpy())
    if ema is not None:
        ema.restore(model)
    return total_loss / max(1, n), np.concatenate(all_true), np.concatenate(all_pred)


def train_loop_v2(model: nn.Module,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  criterion: nn.Module,
                  optimizer: torch.optim.Optimizer,
                  device: str,
                  epochs: int,
                  num_classes: int = 2,
                  scheduler: WarmupCosineSchedule | None = None,
                  ema: EMA | None = None,
                  mixup_alpha: float = 0.0,
                  cutmix_alpha: float = 0.0,
                  mixup_prob: float = 0.0,
                  cutmix_prob: float = 0.0,
                  early_stop_patience: int | None = None,
                  log: EpochLog | None = None,
                  checkpoint_path=None,
                  grad_clip: float | None = 1.0) -> EpochLog:
    """Train loop with focal loss, EMA, optional mixup/cutmix, and a schedule.

    `criterion` must accept either hard-int targets [B] or soft [B,C] —
    `FocalLoss` does. `mixup_*` and `cutmix_*` should be 0 in stage 1.
    EMA (if provided) is updated each step.
    """
    log = log or EpochLog()
    best_f1, best_state, since_improved = -1.0, None, 0

    for epoch in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step(epoch)
        t0 = time.time()
        model.train()
        total_loss, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if mixup_prob > 0 or cutmix_prob > 0:
                x, target = mixup_cutmix(x, y, num_classes,
                                         mixup_alpha, cutmix_alpha,
                                         mixup_prob, cutmix_prob)
            else:
                target = y
            logits = model(x)
            loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        tr_loss = total_loss / max(1, n)

        val_loss, y_true, y_pred = _val_epoch(
            model, val_loader,
            criterion=lambda lg, yy: F.cross_entropy(lg, yy),
            device=device, ema=ema,
        )
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        log.train_loss.append(tr_loss)
        log.val_loss.append(val_loss)
        log.val_f1.append(f1)
        lr = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:02d}  lr={lr:.2e}  tr_loss={tr_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_f1={f1:.4f}  ({time.time()-t0:.1f}s)")
        if f1 > best_f1:
            best_f1 = f1
            if ema is not None:
                ema.apply_shadow(model)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                ema.restore(model)
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improved = 0
            if checkpoint_path is not None:
                torch.save(best_state, str(checkpoint_path))
                print(f"  -> checkpoint saved to {checkpoint_path}")
        else:
            since_improved += 1
            if early_stop_patience is not None and since_improved >= early_stop_patience:
                print(f"  early stop at epoch {epoch} (best val_f1={best_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return log


# ----------------------------------------------------------------------------
# Inference (single-pass + TTA)
# ----------------------------------------------------------------------------

@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str):
    """Return (y_true, y_pred, y_prob_positive) — single forward pass."""
    model.eval()
    ys, preds, probs = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1]
        preds.append(logits.argmax(1).cpu().numpy())
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


def _apply_tta(x: torch.Tensor, name: str) -> torch.Tensor:
    """Apply a named TTA transform to a [B, C, H, W] batch on GPU."""
    if name == "identity":
        return x
    if name == "hflip":
        return x.flip(dims=[3])
    if name == "vflip":
        return x.flip(dims=[2])
    if name == "hvflip":
        return x.flip(dims=[2, 3])
    if name == "rot90":
        return torch.rot90(x, k=1, dims=[2, 3])
    if name == "rot180":
        return torch.rot90(x, k=2, dims=[2, 3])
    if name == "rot270":
        return torch.rot90(x, k=3, dims=[2, 3])
    if name == "hflip_rot90":
        return torch.rot90(x.flip(dims=[3]), k=1, dims=[2, 3])
    raise ValueError(f"Unknown TTA transform: {name}")


@torch.no_grad()
def tta_predict(model: nn.Module, loader: DataLoader, device: str,
                tta_names: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob_tta) by averaging post-softmax probabilities
    across the given TTA transforms. Square-input architectures only.
    """
    model.eval()
    tta_names = list(tta_names)
    ys, prob_sum, count = [], None, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        batch_probs = None
        for name in tta_names:
            xt = _apply_tta(x, name)
            logits = model(xt)
            p = torch.softmax(logits, dim=1)[:, 1]
            batch_probs = p if batch_probs is None else batch_probs + p
        batch_probs = (batch_probs / len(tta_names)).cpu().numpy()
        if prob_sum is None:
            prob_sum = [batch_probs]
        else:
            prob_sum.append(batch_probs)
        ys.append(y.numpy())
        count += 1
    return np.concatenate(ys), np.concatenate(prob_sum)


def tune_threshold(y_val: np.ndarray, y_prob_val: np.ndarray,
                   ths: np.ndarray | None = None) -> tuple[float, float]:
    """Sweep thresholds on validation, return (best_t, best_val_f1)."""
    if ths is None:
        ths = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_val, (y_prob_val > t).astype(int), zero_division=0) for t in ths]
    i = int(np.argmax(f1s))
    return float(ths[i]), float(f1s[i])


# ----------------------------------------------------------------------------
# Legacy train_loop used by notebook 03
# ----------------------------------------------------------------------------

def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    all_pred, all_true = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            all_pred.append(logits.argmax(1).detach().cpu().numpy())
            all_true.append(y.detach().cpu().numpy())
    return total_loss / n, np.concatenate(all_true), np.concatenate(all_pred)


def train_loop(model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               criterion: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str,
               epochs: int,
               early_stop_patience: int | None = None,
               log: EpochLog | None = None,
               checkpoint_path=None) -> EpochLog:
    """Simple train loop used by notebook 03 (EfficientNet-B0)."""
    log = log or EpochLog()
    best_f1, best_state, since_improved = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, y_true, y_pred = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        log.train_loss.append(tr_loss)
        log.val_loss.append(val_loss)
        log.val_f1.append(f1)
        print(f"epoch {epoch:02d}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_f1={f1:.4f}  ({time.time()-t0:.1f}s)")
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_improved = 0
            if checkpoint_path is not None:
                torch.save(best_state, str(checkpoint_path))
                print(f"  -> checkpoint saved to {checkpoint_path}")
        else:
            since_improved += 1
            if early_stop_patience is not None and since_improved >= early_stop_patience:
                print(f"  early stop at epoch {epoch} (best val_f1={best_f1:.4f})")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return log
