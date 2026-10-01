"""C5: data augmentation (chapter Sec. 3.3) and the coarse-graining bug (§3.1).

- Corrected coarse-graining stays in [0, 1], and a block of 1s stays 1.
- Faithful coarse-graining reproduces the notebook: maximum 1/4096.
- Inverting a diagram of rule r gives a diagram of the complement rule, and
  mirroring gives a diagram of the mirror rule. So both augmentations change
  the rule but not the equivalence class.
- Each diagram receives one of n augmentations with probability n/(n+1).
"""

import numpy as np

from eca_cnn.augment import Augmenter, coarse_grain, invert, mirror, salt_and_pepper
from eca_cnn.lookup import complement_rule, mirror_rule
from eca_cnn.simulate import evolve


def _batch(rng, n=8):
    return rng.integers(0, 2, size=(n, 64, 64, 1)).astype(np.float32)


def test_coarse_grain_corrected_range(rng):
    x = _batch(rng)
    y = coarse_grain(x)
    assert y.shape == x.shape
    assert y.min() >= 0 and y.max() <= 1
    assert set(np.unique(y)) <= {0, 0.25, 0.5, 0.75, 1}
    ones = np.ones((1, 64, 64, 1), np.float32)
    assert (coarse_grain(ones) == 1).all()
    # Each 2 x 2 block holds its own mean.
    assert y[0, 0, 0, 0] == x[0, :2, :2, 0].mean()


def test_coarse_grain_faithful_blanks_the_image():
    ones = np.ones((1, 64, 64, 1), np.float32)
    y = coarse_grain(ones, faithful=True)
    assert np.isclose(y.max(), 1 / 4096)


def test_invert_and_mirror_change_the_rule(rng):
    row = rng.integers(0, 2, size=64, dtype=np.uint8)
    for rule in (30, 54, 110, 184):
        d = evolve(row, rule, 64)
        inv = invert(d[None, ..., None])[0, ..., 0]
        mir = mirror(d[None, ..., None])[0, ..., 0]
        assert np.array_equal(inv, evolve(inv[0], complement_rule(rule), 64))
        assert np.array_equal(mir, evolve(mir[0], mirror_rule(rule), 64))


def test_salt_and_pepper_fraction(rng):
    x = np.zeros((16, 64, 64, 1), np.float32)
    y = salt_and_pepper(x, 0.01, rng)
    assert set(np.unique(y)) <= {0, 1}
    assert abs(y.mean() - 0.01) < 0.002


def test_one_of_n_sampling(rng):
    aug = Augmenter(["invert", "mirror", "coarse_grain"])
    choice = aug.choose(200_000, rng)
    freq = np.bincount(choice, minlength=4) / len(choice)
    assert np.allclose(freq, 0.25, atol=0.005)  # n/(n+1) = 3/4 augmented, 1/4 each


def test_augmenter_changes_only_chosen_diagrams(rng):
    x = _batch(rng, n=256)
    aug = Augmenter(["invert"])
    y = aug(x, np.random.default_rng(1))
    changed = (y != x).any(axis=(1, 2, 3))
    assert np.array_equal(y[changed], 1 - x[changed])
    assert 0.4 < changed.mean() < 0.6
    # Deterministic for a given generator state.
    assert np.array_equal(y, aug(x, np.random.default_rng(1)))


def test_no_augmentation_is_identity(rng):
    x = _batch(rng)
    assert Augmenter([])(x, rng) is x
