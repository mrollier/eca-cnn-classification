"""Online data augmentation (chapter Sec. 3.3).

The point is not to fight overfitting but to change *what* the network can
learn. Each augmentation alters a diagram at the level of single cells, so
that the rule can no longer be read off reliably, while the mesoscopic pattern
(and hence the Li-Packard class) stays recognisable. The label is never
changed.

``invert``
    Swap black and white. The diagram of rule r becomes a diagram of its
    complement rule (:func:`eca_cnn.lookup.complement_rule`).
``mirror``
    Flip left-right. The diagram becomes one of the mirror rule
    (:func:`eca_cnn.lookup.mirror_rule`).
``coarse_grain``
    Replace each 2 x 2 block by its mean (values 0, 1/4, 1/2, 3/4, 1).
``salt_and_pepper``
    Flip each cell independently with a small probability (1%).

As in ``data-augmentation`` (``CustomDataGenerator.actions``), every diagram
of a batch receives exactly one of the n selected augmentations with
probability n/(n+1), or stays unchanged with probability 1/(n+1).
Augmentations are never combined on one diagram.

The notebook did this inside a Keras ``ImageDataGenerator``. Here
:class:`Augmenter` acts on whole batches with an explicit random generator,
which gives the same distribution and makes it reproducible.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import AUGMENTATIONS, Config

# All functions take a batch of shape (batch, time, space, channels) and
# return a new array of the same shape and dtype.


def invert(x: np.ndarray) -> np.ndarray:
    """Swap the two states: ``1 - x``."""
    return 1 - x


def mirror(x: np.ndarray) -> np.ndarray:
    """Reverse the space axis (axis 2)."""
    return x[:, :, ::-1].copy()


def coarse_grain(x: np.ndarray, block: int = 2, faithful: bool = False) -> np.ndarray:
    """Average over ``block x block`` windows and blow the result back up.

    The output has the input's shape, and every cell holds the mean of its
    block, so it stays in [0, 1].

    With ``faithful=True`` the notebook's bug is reproduced: the block values
    are additionally divided by ``timesteps**2`` (= 4096) before averaging, so
    the result never exceeds 1/4096 and the network sees a black image with
    the original label (``docs/provenance.md`` §3.1).
    """
    b, t, w, c = x.shape
    if t % block or w % block:
        raise ValueError(f"diagram size {t} x {w} is not divisible by the block size {block}")
    blocks = x.reshape(b, t // block, block, w // block, block, c)
    if faithful:
        # docs/provenance.md §3.1: CustomDataGenerator._coarse_grain divides by
        # width**2 with width = image.shape[0] (the 64 timesteps), although the
        # mean below already normalises. The fix is to delete this line.
        blocks = blocks / t**2
    means = blocks.mean(axis=(2, 4))
    return np.repeat(np.repeat(means, block, axis=1), block, axis=2).astype(x.dtype)


def salt_and_pepper(x: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Flip every cell independently with probability ``fraction``.

    The notebook draws a 0/1 mask with ``p = [1 - fraction, fraction]`` and
    takes ``|mask - x|``, which for binary ``x`` is the same flip.
    """
    flip = rng.random(x.shape) < fraction
    return np.where(flip, 1 - x, x).astype(x.dtype)


class Augmenter:
    """Apply one randomly chosen augmentation (or none) to every diagram of a batch.

    Parameters
    ----------
    augmentations : sequence of str
        The n selected augmentations, from ``invert``, ``mirror``,
        ``coarse_grain``, ``salt_and_pepper``.
    salt_and_pepper_fraction : float
        Probability of flipping a cell (0.01 in the chapter).
    coarse_grain_block : int
        Block size of coarse-graining (2 in the chapter).
    faithful : bool
        Reproduce the coarse-graining bug (§3.1).
    """

    def __init__(self, augmentations: Sequence[str], salt_and_pepper_fraction: float = 0.01,
                 coarse_grain_block: int = 2, faithful: bool = False):
        for a in augmentations:
            if a not in AUGMENTATIONS:
                raise ValueError(f"unknown augmentation {a!r}")
        self.augmentations = list(augmentations)
        self.salt_and_pepper_fraction = salt_and_pepper_fraction
        self.coarse_grain_block = coarse_grain_block
        self.faithful = faithful

    @classmethod
    def from_config(cls, cfg: Config) -> "Augmenter":
        a = cfg.augment
        return cls(a.augmentations, a.salt_and_pepper_fraction, a.coarse_grain_block, cfg.faithful)

    def apply(self, name: str, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply the augmentation called ``name`` to every diagram of ``x``."""
        if name == "invert":
            return invert(x)
        if name == "mirror":
            return mirror(x)
        if name == "coarse_grain":
            return coarse_grain(x, self.coarse_grain_block, self.faithful)
        if name == "salt_and_pepper":
            return salt_and_pepper(x, self.salt_and_pepper_fraction, rng)
        raise ValueError(f"unknown augmentation {name!r}")

    def choose(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """For each of ``n`` diagrams: 0 for "unchanged", ``k`` for augmentation ``k - 1``.

        Uniform over the n + 1 options, like ``np.random.choice(['none', *selected])``.
        """
        return rng.integers(0, len(self.augmentations) + 1, size=n)

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.augmentations:
            return x
        choice = self.choose(len(x), rng)
        out = x.copy()
        for k, name in enumerate(self.augmentations, start=1):
            sel = choice == k
            if sel.any():
                out[sel] = self.apply(name, x[sel], rng)
        return out
