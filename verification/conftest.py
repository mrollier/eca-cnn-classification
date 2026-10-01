"""Shared fixtures for the verification suite.

Claim tests are named ``test_c<N>_<topic>.py``; the claims they check are
listed in ``docs/provenance.md`` §6. Tests that need the full generated
datasets are marked ``slow`` and skip when the data are absent.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # the figure tests draw off-screen

import numpy as np
import pytest

from eca_cnn.config import load_config
from eca_cnn.data import dataset_path, generate_set
from eca_cnn.lookup import load_lookup


@pytest.fixture
def rng():
    return np.random.default_rng(20261001)


@pytest.fixture(scope="session")
def lookup():
    return load_lookup()


@pytest.fixture(scope="session")
def small_set():
    """Four 64 x 64 diagrams per rule (1024 in total), seeded."""
    return generate_set(4, 64, 64, seed=12345)


@pytest.fixture(scope="session")
def base_config():
    return load_config("configs/base.yaml")


@pytest.fixture(scope="session")
def full_data(base_config):
    """The full generated datasets, or skip if ``make_dataset.py`` has not run."""
    from eca_cnn.data import load_datasets

    if not all(dataset_path(base_config, w).exists() for w in ("trainval", "test")):
        pytest.skip("generated datasets absent; run scripts/make_dataset.py")
    return load_datasets(base_config)
