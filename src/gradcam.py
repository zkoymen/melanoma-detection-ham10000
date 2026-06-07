"""
Minimal Grad-CAM implementation.

Reference: Selvaraju et al. 2017, "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization".
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """Pass any conv layer to compute a class-activation heatmap.

    Usage:
        cam = GradCAM(model, model.blocks[-1])
        heatmap = cam(img_tensor, class_idx=1)   # (H, W) float in [0, 1]
        cam.close()
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, out):
        self._activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """`x` is a single (1, 3, H, W) tensor on the same device as the model."""
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        # Global-average-pool the gradients to get one weight per channel.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def close(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(img_rgb_uint8: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Superimpose the Grad-CAM heatmap on the original image."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return ((1 - alpha) * img_rgb_uint8 + alpha * heat).clip(0, 255).astype(np.uint8)
