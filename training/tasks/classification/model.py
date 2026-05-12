from training.nets.classification import build_torchvision_backbone
from training.utils.input_spec import resolve_input_spec


def _get_value(container, key, default=None):
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def build_model(cfg):
    """Build classification model from config-driven architecture."""
    model_cfg = _get_value(cfg, "model", {})
    model_name = _get_value(model_cfg, "name", "torchvision_backbone")

    if model_name != "torchvision_backbone":
        raise ValueError(
            f"Unsupported classification model builder '{model_name}'. "
            "Use model.name=torchvision_backbone"
        )

    architecture = _get_value(model_cfg, "architecture", "resnet18")
    supported = _get_value(model_cfg, "architectures_supported", _get_value(cfg, "architectures_supported", None))
    if isinstance(supported, list) and supported:
        if architecture not in supported:
            raise ValueError(
                f"Unsupported architecture '{architecture}'. "
                f"Allowed: {supported}"
            )
    num_classes = int(_get_value(model_cfg, "num_classes", 2))
    pretrained = bool(_get_value(model_cfg, "pretrained", False))
    pretrained_dir = _get_value(model_cfg, "pretrained_dir", _get_value(cfg, "pretrained_dir", "pretrained_weights"))
    pretrained_strict = bool(_get_value(model_cfg, "pretrained_strict", _get_value(cfg, "pretrained_strict", False)))
    out_size, layout = resolve_input_spec(cfg)
    model = build_torchvision_backbone(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        pretrained_dir=pretrained_dir,
        pretrained_strict=pretrained_strict,
    )
    setattr(model, "input_size", out_size)
    setattr(model, "input_layout", layout)
    return model
