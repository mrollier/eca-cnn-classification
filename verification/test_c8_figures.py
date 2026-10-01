"""C8: every figure function draws from small inputs without error.

The figures' content is checked by eye (``output/figures``). These tests
guard against a figure breaking silently when the code around it changes.
"""

import numpy as np
import pytest

from eca_cnn import figures as F
from eca_cnn.benchmark import scan_set


@pytest.fixture(scope="module")
def scans(small_set, lookup):
    return scan_set(small_set, lookup), scan_set(small_set, lookup, faithful=True)


def _close(fig):
    F.plt.close(fig)


def test_data_figures(small_set, lookup, scans):
    d = small_set.diagrams(slice(0, 5))
    _close(F.lp_classes_examples(list(d), F.CLASS_EXAMPLE_RULES))
    for palette in F.PALETTES:
        _close(F.benchmark_undetermined(scans[0], scans[1], lookup, palette))
    _close(F.benchmark_undetermined_examples([{"diagram": d[0], "rule": 0, "class": "null", "missing": ["000"]}]))
    _close(F.example_convolution_and_relu(d[1]))
    _close(F.convolutional_decomposition(d[2], 0))
    _close(F.augmentation_examples(d[3]))


def test_model_figures(small_set, lookup):
    rng = np.random.default_rng(0)
    n = 300
    rules = small_set.rules[:n]
    pred = {"rule": rules, "label": rules, "pred": np.where(rng.random(n) < 0.1, 0, rules)}
    _close(F.default_model_misidentified(pred, lookup))
    _close(F.default_model_all_wrong(small_set.diagrams(slice(0, 3)), np.array([0, 1, 2]), np.array([1, 2, 1])))
    _close(F.default_model_all_wrong(np.zeros((0, 64, 64)), np.array([]), np.array([])))
    cm = rng.integers(0, 100, size=(5, 5))
    _close(F.confusion_matrix_final(cm))
    _close(F.confusion_matrix_final(cm, rng.integers(0, 5, size=(88, 88))))


def test_confusion_normalised_over_predictions():
    from eca_cnn.evaluate import confusion, normalise

    cm = confusion(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 1]), 3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]
    n = normalise(cm, "pred")
    assert np.allclose(n.sum(axis=0), [1, 1, 0])  # an empty column stays 0
    assert np.isclose(n[0, 1], 1 / 3)
