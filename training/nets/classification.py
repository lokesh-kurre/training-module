from __future__ import annotations

from typing import Any

import torch.nn as nn

from training.nets.backbone import build_backbone


def _replace_classifier(model: Any, num_classes: int, dropout: float = 0.0) -> Any:
    """Replace native classifier head with requested output classes."""
    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Sequential) and len(classifier) > 0:
            last_idx = len(classifier) - 1
            last_layer = classifier[last_idx]
            if isinstance(last_layer, nn.Linear):
                if dropout > 0:
                    classifier[last_idx] = nn.Sequential(
                        nn.Dropout(p=float(dropout)),
                        nn.Linear(last_layer.in_features, num_classes),
                    )
                else:
                    classifier[last_idx] = nn.Linear(last_layer.in_features, num_classes)
                model.classifier = classifier
                return model
        if isinstance(classifier, nn.Linear):
            if dropout > 0:
                model.classifier = nn.Sequential(
                    nn.Dropout(p=float(dropout)),
                    nn.Linear(classifier.in_features, num_classes),
                )
            else:
                model.classifier = nn.Linear(classifier.in_features, num_classes)
            return model

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        if dropout > 0:
            model.fc = nn.Sequential(
                nn.Dropout(p=float(dropout)),
                nn.Linear(model.fc.in_features, num_classes),
            )
        else:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if hasattr(model, "heads"):
        heads = model.heads
        if isinstance(heads, nn.Sequential) and len(heads) > 0:
            last_idx = len(heads) - 1
            last_layer = heads[last_idx]
            if isinstance(last_layer, nn.Linear):
                if dropout > 0:
                    heads[last_idx] = nn.Sequential(
                        nn.Dropout(p=float(dropout)),
                        nn.Linear(last_layer.in_features, num_classes),
                    )
                else:
                    heads[last_idx] = nn.Linear(last_layer.in_features, num_classes)
                model.heads = heads
                return model
        if isinstance(heads, nn.Linear):
            if dropout > 0:
                model.heads = nn.Sequential(
                    nn.Dropout(p=float(dropout)),
                    nn.Linear(heads.in_features, num_classes),
                )
            else:
                model.heads = nn.Linear(heads.in_features, num_classes)
            return model

    if isinstance(model, nn.Sequential) and len(model) > 0 and isinstance(model[-1], nn.Linear):
        if dropout > 0:
            model[-1] = nn.Sequential(
                nn.Dropout(p=float(dropout)),
                nn.Linear(model[-1].in_features, num_classes),
            )
        else:
            model[-1] = nn.Linear(model[-1].in_features, num_classes)
        return model

    raise ValueError("Unsupported model head for classifier replacement")


def build_torchvision_backbone(
    architecture: str,
    num_classes: int | None = None,
    pretrained: bool = False,
    pretrained_dir: str = "pretrained_weights",
    pretrained_strict: bool = False,
    dropout: float = 0.0,
) -> Any:
    """Build classification network from reusable backbone module.

    Behavior:
    - num_classes is None: returns feature-only backbone.
    - num_classes is set: returns full architecture with replaced classifier head.
    """
    if num_classes is None:
        return build_backbone(
            architecture=architecture,
            pretrained=pretrained,
            pretrained_dir=pretrained_dir,
            pretrained_strict=pretrained_strict,
            classification_head=False,
        )

    model = build_backbone(
        architecture=architecture,
        pretrained=pretrained,
        pretrained_dir=pretrained_dir,
        pretrained_strict=pretrained_strict,
        classification_head=True,
    )
    return _replace_classifier(model=model, num_classes=int(num_classes), dropout=dropout)
