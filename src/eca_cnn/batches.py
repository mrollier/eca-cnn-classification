"""Keras batch generator over bit-packed diagram sets.

Kept apart from :mod:`eca_cnn.data` so that generating data and running the
benchmark do not have to import TensorFlow.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import tensorflow as tf

from .data import DiagramSet, one_hot


class DiagramSequence(tf.keras.utils.Sequence):
    """Batches of ``(diagrams, one-hot labels)`` for ``model.fit`` and ``predict``.

    Diagrams are unpacked to float32 of shape ``(batch, timesteps, width, 1)``
    only when a batch is requested. The notebooks fed the whole uint8 array to
    Keras instead, which needs about 1 GB of memory for the training set and
    4 GB once cast to float32.

    Parameters
    ----------
    ds : DiagramSet
        The set to draw from.
    indices : array of int
        Which diagrams of ``ds`` belong to this sequence (e.g. the training part).
    labels : array of int, or None
        Integer targets aligned with ``indices``. ``None`` yields diagrams only
        (for ``predict``).
    n_classes : int
        Length of the one-hot vectors.
    batch_size : int
        64 throughout the chapter.
    shuffle : bool
        Reshuffle the order at the end of every epoch. ``model.fit`` is called
        with ``shuffle=False``, so this is the only shuffling.
    seed : int
        Fixes the shuffles and the augmentation draws.
    augmenter : callable, optional
        ``augmenter(batch, rng) -> batch``, e.g. :class:`eca_cnn.augment.Augmenter`.
    crop : (rows, columns), optional
        Keep only the top-left corner of each diagram (``MLP-verification_cropped``).

    Notes
    -----
    Randomness depends only on ``(seed, epoch, batch index)``. Shuffling and
    augmentation are therefore the same on every run, whatever order Keras
    requests the batches in. (Keras requests batch 0 once more before training,
    to infer the output shapes.)
    """

    def __init__(self, ds: DiagramSet, indices: np.ndarray, labels: Optional[np.ndarray],
                 n_classes: int, batch_size: int = 64, shuffle: bool = False, seed: int = 0,
                 augmenter=None, crop=None):
        super().__init__()
        self.ds = ds
        self.indices = np.asarray(indices)
        self.labels = None if labels is None else np.asarray(labels)
        if self.labels is not None and len(self.labels) != len(self.indices):
            raise ValueError("labels must be aligned with indices")
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.augmenter = augmenter
        self.crop = crop
        self.epoch = 0
        self._order = self._make_order()

    def _make_order(self) -> np.ndarray:
        if not self.shuffle:
            return np.arange(len(self.indices))
        return np.random.default_rng([self.seed, self.epoch, 0]).permutation(len(self.indices))

    def __len__(self) -> int:
        return math.ceil(len(self.indices) / self.batch_size)

    def batch_positions(self, i: int) -> np.ndarray:
        """Positions (into ``indices``) of the diagrams in batch ``i``."""
        return self._order[i * self.batch_size:(i + 1) * self.batch_size]

    def __getitem__(self, i: int):
        pos = self.batch_positions(i)
        x = self.ds.diagrams(self.indices[pos]).astype(np.float32)[..., None]
        if self.crop is not None:
            x = x[:, :self.crop[0], :self.crop[1]]
        if self.augmenter is not None:
            x = self.augmenter(x, np.random.default_rng([self.seed, self.epoch, i + 1]))
        if self.labels is None:
            return x
        return x, one_hot(self.labels[pos], self.n_classes)

    def on_epoch_end(self) -> None:
        self.epoch += 1
        self._order = self._make_order()
