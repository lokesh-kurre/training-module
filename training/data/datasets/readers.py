from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from training.utils.io import read_image
from training.utils.input_spec import out_size_to_chw


def _extract_record_fields(sample: Any) -> tuple[str, str]:
    """Return (filepath, class_label) from list/tuple/dict-like sample."""
    if isinstance(sample, dict):
        path = sample.get("filepath")
        label = sample.get("class")
        if path is None or label is None:
            raise ValueError("dict sample must contain keys 'filepath' and 'class'")
        return str(path), str(label)

    if isinstance(sample, (list, tuple)) and len(sample) >= 2:
        return str(sample[0]), str(sample[1])

    raise TypeError("sample must be dict or list/tuple with [filepath, class]")


def _to_uint8_hwc(image: Any) -> np.ndarray:
    if image is None:
        raise ValueError("empty image")

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"expected 3D image array, got shape={arr.shape}")

    if arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _random_image_hwc() -> np.ndarray:
    h = random.randint(201, 249)
    w = random.randint(201, 299)
    return np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)


def _resize_hwc(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_hw
    try:
        import cv2  # type: ignore[import-not-found]

        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return resized
    except ModuleNotFoundError:
        from PIL import Image

        pil = Image.fromarray(image)
        return np.asarray(pil.resize((target_w, target_h)))


def _augment_train_hwc(image: np.ndarray) -> np.ndarray:
    return _augment_train_hwc_cfg(image, cfg=None)


def _cfg_value(cfg: dict[str, Any] | None, key: str, default: float) -> float:
    if cfg is None:
        return default
    value = cfg.get(key, default)
    try:
        return float(value)
    except Exception:
        return default


def _augment_hsv(image: np.ndarray, cfg: dict[str, Any] | None) -> np.ndarray:
    hsv_h = max(0.0, _cfg_value(cfg, "hsv_h", 0.0))
    hsv_s = max(0.0, _cfg_value(cfg, "hsv_s", 0.0))
    hsv_v = max(0.0, _cfg_value(cfg, "hsv_v", 0.0))
    if hsv_h <= 0 and hsv_s <= 0 and hsv_v <= 0:
        return image

    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return image

    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    # Hue range in OpenCV HSV is [0, 179].
    hue_shift = random.uniform(-hsv_h, hsv_h) * 179.0
    sat_scale = 1.0 + random.uniform(-hsv_s, hsv_s)
    val_scale = 1.0 + random.uniform(-hsv_v, hsv_v)

    img_hsv[..., 0] = (img_hsv[..., 0] + hue_shift) % 180.0
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] * sat_scale, 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] * val_scale, 0, 255)

    return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def _augment_geom(image: np.ndarray, cfg: dict[str, Any] | None) -> np.ndarray:
    degrees = _cfg_value(cfg, "degrees", 0.0)
    translate = max(0.0, _cfg_value(cfg, "translate", 0.0))
    scale = max(0.0, _cfg_value(cfg, "scale", 0.0))
    shear = _cfg_value(cfg, "shear", 0.0)
    perspective = max(0.0, _cfg_value(cfg, "perspective", 0.0))
    if degrees == 0.0 and translate == 0.0 and scale == 0.0 and shear == 0.0 and perspective == 0.0:
        return image

    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return image

    h, w = image.shape[:2]
    angle = random.uniform(-degrees, degrees)
    scale_factor = random.uniform(max(0.1, 1.0 - scale), 1.0 + scale)
    tx = random.uniform(-translate, translate) * w
    ty = random.uniform(-translate, translate) * h
    shear_x = math.tan(math.radians(random.uniform(-shear, shear)))
    shear_y = math.tan(math.radians(random.uniform(-shear, shear)))

    mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale_factor)
    mat[0, 1] += shear_x
    mat[1, 0] += shear_y
    mat[0, 2] += tx
    mat[1, 2] += ty
    out = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    if perspective > 0.0:
        d = perspective * min(h, w)
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        dst = src + np.float32(
            [
                [random.uniform(-d, d), random.uniform(-d, d)],
                [random.uniform(-d, d), random.uniform(-d, d)],
                [random.uniform(-d, d), random.uniform(-d, d)],
                [random.uniform(-d, d), random.uniform(-d, d)],
            ]
        )
        pmat = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, pmat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return out


def _augment_erasing(image: np.ndarray, cfg: dict[str, Any] | None) -> np.ndarray:
    p = max(0.0, min(1.0, _cfg_value(cfg, "erasing", 0.0)))
    if p <= 0.0 or random.random() >= p:
        return image

    h, w = image.shape[:2]
    area = float(h * w)
    target_area = random.uniform(0.02, 0.15) * area
    aspect = random.uniform(0.3, 3.0)
    erase_h = int(round(math.sqrt(target_area * aspect)))
    erase_w = int(round(math.sqrt(target_area / max(1e-6, aspect))))
    erase_h = max(1, min(h, erase_h))
    erase_w = max(1, min(w, erase_w))

    top = random.randint(0, max(0, h - erase_h))
    left = random.randint(0, max(0, w - erase_w))
    fill = np.random.randint(0, 256, size=(erase_h, erase_w, 3), dtype=np.uint8)
    image[top : top + erase_h, left : left + erase_w] = fill
    return image


def _augment_train_hwc_cfg(image: np.ndarray, cfg: dict[str, Any] | None) -> np.ndarray:
    fliplr = max(0.0, min(1.0, _cfg_value(cfg, "fliplr", 0.5)))
    flipud = max(0.0, min(1.0, _cfg_value(cfg, "flipud", 0.0)))

    if random.random() < fliplr:
        image = image[:, ::-1, :]
    if random.random() < flipud:
        image = image[::-1, :, :]

    image = _augment_geom(image, cfg=cfg)
    image = _augment_hsv(image, cfg=cfg)
    image = _augment_erasing(image, cfg=cfg)
    return image


def _imagenet_normalize_chw(image_chw: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    return (image_chw - mean) / std


def _label_to_index(label: Any, label_to_index: dict[str, int] | None = None) -> int:
    if label_to_index is not None:
        key = str(label)
        if key in label_to_index:
            return int(label_to_index[key])

    try:
        return int(label)
    except Exception as exc:
        raise ValueError(f"unsupported class label '{label}', provide numeric labels or label_to_index") from exc


def classification_read_uri_image(
    sample: Any,
    out_size: tuple[int, ...],
    layout: str,
    split: str = "train",
    label_to_index: dict[str, int] | None = None,
    s3_client: Any = None,
    cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.int64]:
    """Read URI image, fallback to random image, preprocess, return image + numeric label.

    Expected sample columns:
      filepath: URI/path
      class: label1|label2|label3
    """
    filepath, class_label = _extract_record_fields(sample)

    raw = read_image(filepath=filepath, as_pillow=False, verbose=False, s3_client=s3_client)
    if raw is None:
        image_hwc = _random_image_hwc()
    else:
        try:
            image_hwc = _to_uint8_hwc(raw)
        except Exception:
            image_hwc = _random_image_hwc()

    channels, out_h, out_w = out_size_to_chw(out_size, layout)
    if channels != 3:
        raise ValueError(f"classification_read_uri_image expects 3 channels, got {channels}")

    image_hwc = _resize_hwc(image_hwc, (out_h, out_w))

    if split == "train":
        image_hwc = _augment_train_hwc_cfg(image_hwc, cfg=cfg)

    image_chw = np.transpose(image_hwc.astype(np.float32) / 255.0, (2, 0, 1))
    image_chw = _imagenet_normalize_chw(image_chw).astype(np.float32)

    if layout.upper() == "HWC":
        image = np.transpose(image_chw, (1, 2, 0)).astype(np.float32)
    else:
        image = image_chw

    label = np.int64(_label_to_index(class_label, label_to_index=label_to_index))
    return image, label
