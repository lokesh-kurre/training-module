"""Compatibility wrappers for the old data module path."""

from training.data.datasets.factory import build_dataset_from_config
from training.data.datasets.synthetic import build_synthetic_dataset

__all__ = ["build_dataset_from_config", "build_synthetic_dataset"]
