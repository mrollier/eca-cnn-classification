"""C4: the networks.

- Parameter counts quoted in the chapter: default 6016 / 7301, exact 2360,
  trimmed 389 / 1504 / 664 (and the two MLPs of the archived notebooks).
- The exact CNN returns the right rule for each of the 256 rules on a diagram
  that contains all eight neighbourhoods.
- On any diagram, the exact CNN's answer equals the rule table read off by the
  scan with missing entries set to 0. So it can only fail on undetermined
  diagrams, and only where the missing entry maps to 1 (chapter Sec. 3.1).
- The output weights follow the chapter's formulas.
"""

import numpy as np
import pytest

from eca_cnn.benchmark import read_rule_tables, table_to_rule
from eca_cnn.models import count_parameters, default_cnn, exact_cnn, exact_weights, mlp, trimmed_cnn
from eca_cnn.simulate import evolve


@pytest.mark.parametrize("build, output, expected", [
    (default_cnn, "rules", 6016),
    (default_cnn, "classes", 7301),
    (trimmed_cnn, "classes", 389),
    (trimmed_cnn, "rules", 1504),
    (trimmed_cnn, "independent", 664),
])
def test_parameter_counts(build, output, expected):
    assert count_parameters(build(output)) == expected


def test_exact_cnn_parameter_count():
    assert count_parameters(exact_cnn()) == 2360


def test_mlp_parameter_counts():
    # MLP-verification: 64 x 64 -> 256 -> 256 rules (saved summary: 1,114,624).
    assert count_parameters(mlp("rules", (64, 64, 1), [256])) == 1_114_624
    # MLP-verification_cropped: 28 x 28 -> 1024 -> 512 -> 256 -> 5 (saved summary: 1,461,253).
    assert count_parameters(mlp("classes", (28, 28, 1), [1024, 512, 256])) == 1_461_253


def test_receptive_field_layer_shapes():
    # Chapter Sec. 2.2: channels of 62 x 63, then 60 x 62 (space x time);
    # Keras reports (time, space).
    m = default_cnn("rules")
    shapes = [tuple(l.output.shape[1:3]) for l in m.layers if l.__class__.__name__ == "Conv2D"]
    assert shapes == [(63, 62), (62, 60)]


def _all_neighbourhood_row():
    # A de Bruijn sequence of order 3: every triplet occurs once, cyclically.
    seq = [0, 0, 0, 1, 0, 1, 1, 1]
    return np.array(seq * 8, dtype=np.uint8)


def test_exact_cnn_identifies_every_rule():
    row = _all_neighbourhood_row()
    diagrams = np.stack([evolve(row, r, 64) for r in range(256)]).astype(np.float32)[..., None]
    pred = exact_cnn().predict(diagrams, verbose=0).argmax(axis=1)
    assert pred.tolist() == list(range(256))


def test_exact_cnn_equals_scan_with_missing_as_zero(small_set):
    d = small_set.diagrams()
    pred = exact_cnn().predict(d.astype(np.float32)[..., None], batch_size=256, verbose=0).argmax(axis=1)
    tables = read_rule_tables(d)
    assert np.array_equal(pred, table_to_rule(tables, missing_as=0))
    wrong = pred != small_set.rules
    # Every error is an undetermined diagram with a missing 1-entry.
    assert ((tables == -1).any(axis=1) | ~wrong).all()


def test_exact_output_weights_match_chapter_formula():
    _, _, w, b = exact_weights(weight_factor=1.0)
    for i in (0, 1, 30, 110, 255):
        bits = [(i >> j) & 1 for j in range(8)]
        assert w[:, i].tolist() == [2 * x - 1 for x in bits]
        assert b[i] == 1 - sum(bits)
    # The node of the actual rule receives 1, every other node 1 - Hamming distance.
    x = np.array([(110 >> j) & 1 for j in range(8)], dtype=float)
    scores = x @ w + b
    assert scores[110] == 1 and np.sort(scores)[-2] == 0
