"""Pytest configuration and fixtures for training-module tests."""

import shutil
from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def test_config_dir():
    """Return path to test config directory."""
    return Path(__file__).parent / "configs"


@pytest.fixture
def tmp_run_dir(tmp_path):
    """Create a temporary run directory for test outputs."""
    run_dir = tmp_path / "run" / "test_run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    yield run_dir
    # Cleanup after test
    if run_dir.exists():
        shutil.rmtree(run_dir.parent)


@pytest.fixture
def mock_config():
    """Return a minimal mock config for testing."""
    return {
        "task": "classification",
        "data": "test_data",
        "model": {"architecture": "tiny_cnn", "num_classes": 2},
        "batch": 4,
        "epochs": 1,
        "device": "cpu",
        "trainer": {"epochs": 1, "log_level": "info"},
        "optimizer": {"name": "adam", "lr": 0.001},
        "scheduler": {"name": "constant"},
        "run": {"root_dir": "run", "log_level": "info"},
    }
