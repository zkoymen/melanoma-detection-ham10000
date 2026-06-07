"""
Model factories for the six CNN architectures plus shared freeze/unfreeze
helpers. Every factory returns a 2-class classifier head; the training
pipeline (focal loss + EMA + cosine schedule) lives in `src/training.py`.

Architectures and pretrained sources:
  - AlexNet         : torchvision (IMAGENET1K_V1)
  - VGG16-BN        : torchvision (IMAGENET1K_V1)
  - ResNet50        : torchvision (IMAGENET1K_V2)
  - EfficientNet-B3 : timm        (pretrained=True)
  - DenseNet121     : torchvision (IMAGENET1K_V1)
  - Swin-Tiny       : timm        (swin_tiny_patch4_window7_224)

`gradcam_target_layer(model, name)` returns the right conv layer for
Grad-CAM per architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import timm
import torchvision.models as tvm


# ============================================================================
# Factories
# ============================================================================

def build_alexnet(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = tvm.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.alexnet(weights=weights)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def build_vgg16_bn(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = tvm.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.vgg16_bn(weights=weights)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


def build_resnet50(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = tvm.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b3(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    return timm.create_model("efficientnet_b3",
                             pretrained=pretrained,
                             num_classes=num_classes)


def build_densenet121(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
    model = tvm.densenet121(weights=weights)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def build_swin_tiny(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    return timm.create_model("swin_tiny_patch4_window7_224",
                             pretrained=pretrained,
                             num_classes=num_classes)


# Mapping from arch name -> factory. Used by the notebook template.
BUILDERS = {
    "alexnet":         build_alexnet,
    "vgg16_bn":        build_vgg16_bn,
    "resnet50":        build_resnet50,
    "efficientnet_b3": build_efficientnet_b3,
    "densenet121":     build_densenet121,
    "swin_tiny":       build_swin_tiny,
}


# ============================================================================
# Generic freeze / unfreeze
# ============================================================================

def _classifier_module(model: nn.Module) -> nn.Module:
    """Return the head module (the layer we replaced in the factory).

    timm models expose `get_classifier()`; torchvision uses different
    attributes per arch (`fc` for ResNet, `classifier` for AlexNet/VGG/DenseNet).
    """
    if hasattr(model, "get_classifier"):
        return model.get_classifier()
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        return model.fc
    if hasattr(model, "classifier"):
        return model.classifier
    raise AttributeError("Cannot locate classifier head on this model.")


def freeze_backbone(model: nn.Module) -> None:
    """Freeze every parameter, then unfreeze the head only."""
    for p in model.parameters():
        p.requires_grad = False
    head = _classifier_module(model)
    for p in head.parameters():
        p.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze every parameter."""
    for p in model.parameters():
        p.requires_grad = True


def trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Discriminative-LR parameter groups for stage 2
# ============================================================================

def discriminative_param_groups(model: nn.Module,
                                head_lr: float,
                                backbone_lr: float,
                                weight_decay: float) -> list[dict]:
    """Return two AdamW param groups: head (high LR) + backbone (low LR).

    Anything reachable through `_classifier_module(model)` is the head.
    Everything else is the backbone.
    """
    head = _classifier_module(model)
    head_ids = set(id(p) for p in head.parameters())
    head_params, backbone_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (head_params if id(p) in head_ids else backbone_params).append(p)
    return [
        {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": head_params,     "lr": head_lr,     "weight_decay": weight_decay},
    ]


# ============================================================================
# Penultimate-feature extraction (Method 10 — hybrid fusion)
# ============================================================================

PENULTIMATE_DIMS = {
    "alexnet":         4096,
    "vgg16_bn":        4096,
    "resnet50":        2048,
    "efficientnet_b3": 1536,
    "densenet121":     1024,
    "swin_tiny":       768,
}


def chop_head_for_features(model: nn.Module, arch: str) -> nn.Module:
    """Replace the classifier head so forward(x) returns penultimate features.

    Used by the hybrid-fusion notebook to extract deep features that are
    concatenated with handcrafted + ABCD features before a shallow classifier.
    """
    if arch == "alexnet":
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    elif arch == "vgg16_bn":
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    elif arch == "resnet50":
        model.fc = nn.Identity()
    elif arch == "densenet121":
        model.classifier = nn.Identity()
    elif arch in ("efficientnet_b3", "swin_tiny"):
        model.reset_classifier(0)
    else:
        raise ValueError(f"Unknown arch for feature extraction: {arch}")
    return model


# ============================================================================
# Grad-CAM target layers
# ============================================================================

def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Return the convolutional layer to use for Grad-CAM.

    Swin-Tiny is an attention-based architecture; we pick the last LayerNorm
    before the head, which the GradCAM utility handles even though
    Attention-Rollout would technically be more appropriate.
    """
    if arch == "alexnet":
        return model.features[-3]
    if arch == "vgg16_bn":
        return model.features[-1]
    if arch == "resnet50":
        return model.layer4[-1]
    if arch == "efficientnet_b3":
        return model.blocks[-1]
    if arch == "densenet121":
        return model.features.norm5
    if arch == "swin_tiny":
        return model.norm if hasattr(model, "norm") else model.layers[-1]
    raise ValueError(f"Unknown arch for Grad-CAM: {arch}")


# ============================================================================
# EfficientNet-B0 helpers (used by notebook 03)
# ============================================================================

def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    return timm.create_model("efficientnet_b0",
                             pretrained=pretrained,
                             num_classes=num_classes)


def unfreeze_last_block(model: nn.Module) -> None:
    unfreeze_last_n_blocks(model, n=1)


def unfreeze_last_n_blocks(model: nn.Module, n: int = 2) -> None:
    """EfficientNet-specific (assumes model.blocks list)."""
    for p in model.parameters():
        p.requires_grad = False
    for block in model.blocks[-n:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.get_classifier().parameters():
        p.requires_grad = True
