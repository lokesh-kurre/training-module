from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from training.utils.logger import get_logger


LOGGER = get_logger("training.nets.backbone")


SUPPORTED_TORCHVISION_ARCHITECTURES: tuple[str, ...] = (
    "alexnet",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b3",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
    "convnext_tiny",
    "vit_b_16",
    "swin_t",
)


def _remote_model_name(architecture: str) -> str:
    """Return canonical remote model identifier used by upstream model libs."""
    name_map = {
        # timm id for mobilenet_v1
        "mobilenet_v1": "mobilenetv1_100",
    }
    return name_map.get(architecture, architecture)


def _build_tiny_cnn_with_head(num_classes: int = 1000) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, num_classes),
    )


def _strip_classifier(model: Any) -> Any:
    """Strip classifier layers for feature-only backbone use."""
    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential) and len(classifier) > 0:
            classifier[-1] = nn.Identity()
            model.classifier = classifier
            return model
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Identity()
            return model

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Identity()
        return model

    if hasattr(model, "heads"):
        heads = model.heads
        if isinstance(heads, nn.Sequential) and len(heads) > 0:
            heads[-1] = nn.Identity()
            model.heads = heads
            return model
        if isinstance(heads, nn.Linear):
            model.heads = nn.Identity()
            return model

    if isinstance(model, nn.Sequential) and len(model) > 0 and isinstance(model[-1], nn.Linear):
        model[-1] = nn.Identity()
        return model

    raise ValueError("Unsupported model head for backbone stripping")


def _build_torchvision_model(builder: Any, pretrained: bool) -> Any:
    if pretrained:
        try:
            return builder(weights="DEFAULT")
        except (TypeError, ValueError):
            return builder(pretrained=True)

    try:
        return builder(weights=None)
    except TypeError:
        return builder(pretrained=False)


def _extract_state_dict(raw_obj: Any) -> dict[str, Any]:
    if isinstance(raw_obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            value = raw_obj.get(key)
            if isinstance(value, dict):
                return value
        if all(isinstance(k, str) for k in raw_obj.keys()):
            return raw_obj
    raise ValueError("Unable to extract a valid state_dict from checkpoint")


def _candidate_weight_paths(pretrained_dir: Path, architecture: str) -> list[Path]:
    names = [architecture]
    remote_name = _remote_model_name(architecture)
    if remote_name not in names:
        names.append(remote_name)

    candidates: list[Path] = []
    for name in names:
        base = pretrained_dir / name
        candidates.extend(
            [
                base.with_suffix(".pt"),
                base.with_suffix(".pth"),
                base.with_suffix(".bin"),
                base,
            ]
        )
    return candidates


def _load_local_pretrained(
    model: Any,
    architecture: str,
    pretrained_dir: str | Path = "pretrained_weights",
    pretrained_strict: bool = False,
) -> bool:
    weight_dir = Path(pretrained_dir)
    candidates = _candidate_weight_paths(weight_dir, architecture)
    selected = next((p for p in candidates if p.exists() and p.is_file()), None)
    if selected is None:
        return False

    checkpoint = torch.load(selected, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=bool(pretrained_strict))

    if missing:
        LOGGER.warning("Missing keys while loading pretrained '%s': %s", selected, missing)
    if unexpected:
        LOGGER.warning("Unexpected keys while loading pretrained '%s': %s", selected, unexpected)

    LOGGER.info("Loaded local pretrained weights for %s from %s", architecture, selected)
    return True


def build_backbone(
    architecture: str,
    pretrained: bool = False,
    pretrained_dir: str | Path = "pretrained_weights",
    pretrained_strict: bool = False,
    classification_head: bool = False,
) -> Any:
    """Build model backbone.

    Args:
        architecture: Backbone architecture name.
        pretrained: Whether to load pretrained weights.
        classification_head: If True, return full architecture with native classifier head.
            If False (default), strip classifier and return feature backbone only.
    """
    architecture = architecture.lower()

    if architecture == "tiny_cnn":
        model = _build_tiny_cnn_with_head(num_classes=1000)
        model = model if classification_head else _strip_classifier(model)
        if pretrained:
            _load_local_pretrained(
                model=model,
                architecture=architecture,
                pretrained_dir=pretrained_dir,
                pretrained_strict=pretrained_strict,
            )
        return model

    local_available = False
    if pretrained:
        local_available = any(
            p.exists() and p.is_file()
            for p in _candidate_weight_paths(Path(pretrained_dir), architecture)
        )
        if local_available:
            LOGGER.info(
                "Found local pretrained weights for %s in %s; skipping remote download",
                architecture,
                pretrained_dir,
            )

    if architecture == "mobilenet_v1":
        try:
            import timm
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "timm is required for mobilenet_v1. "
                "Install with: uv pip install --python .venv/bin/python timm"
            ) from exc
        remote_pretrained = bool(pretrained and not local_available)
        if classification_head:
            model = timm.create_model("mobilenetv1_100", pretrained=remote_pretrained, num_classes=1000)
        else:
            model = timm.create_model("mobilenetv1_100", pretrained=remote_pretrained, num_classes=0)

        if pretrained and local_available:
            _load_local_pretrained(
                model=model,
                architecture=architecture,
                pretrained_dir=pretrained_dir,
                pretrained_strict=pretrained_strict,
            )
        return model

    try:
        from torchvision import models as tv_models
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "torch/torchvision are required for this backbone. Install with: "
            "uv pip install --python .venv/bin/python --index-url "
            "https://download.pytorch.org/whl/cpu torch torchvision"
        ) from exc

    if architecture not in SUPPORTED_TORCHVISION_ARCHITECTURES:
        supported = ", ".join(sorted(["tiny_cnn", "mobilenet_v1", *SUPPORTED_TORCHVISION_ARCHITECTURES]))
        raise ValueError(f"Unknown architecture '{architecture}'. Supported: {supported}")

    builder = getattr(tv_models, architecture, None)
    if builder is None:
        raise ValueError(f"torchvision.models has no attribute '{architecture}'")

    remote_pretrained = bool(pretrained and not local_available)
    model = _build_torchvision_model(builder=builder, pretrained=remote_pretrained)
    model = model if classification_head else _strip_classifier(model)

    if pretrained and local_available:
        _load_local_pretrained(
            model=model,
            architecture=architecture,
            pretrained_dir=pretrained_dir,
            pretrained_strict=pretrained_strict,
        )

    return model
