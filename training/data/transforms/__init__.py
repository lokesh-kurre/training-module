"""Transforms package.

Intentionally lightweight in the skeleton. Real projects can register
task/dataset-specific augmentation pipelines here.
"""


def identity_transform(batch):
    return batch
