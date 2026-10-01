"""Datasets of ECA spacetime diagrams: generation, storage, labels and splits.

The chapter (Sec. 1.3) uses two independent samples of 64 x 64 diagrams with
periodic boundaries, each starting from a uniformly random initial row:

- a training-and-validation set of 1024 diagrams per rule (262,144 in total),
  split 3/4 : 1/4 into training and validation;
- a held-out test set of 128 diagrams per rule (32,768 in total).

Diagrams are stored in rule order: diagram ``k`` of rule ``r`` is entry
``r * n_per_rule + k``, as in the original ``.npy`` files.

Storage
-------
A binary 64 x 64 diagram holds 4096 bits. The notebooks stored one byte per
cell (1.2 GB in total). Here each row is packed into 8 bytes with
``np.packbits``, so both sets take about 150 MB on disk and in memory. Batches
are unpacked on the fly by :class:`eca_cnn.batches.DiagramSequence`. The original Colab files
can still be used unchanged (``data.source: original``).

Seeding
-------
``create-dataset`` set no seed, so its datasets cannot be regenerated bit for
bit (§4). :func:`generate_set` draws all initial rows from
``np.random.default_rng(seed)``, so the same seed gives the same data on
every machine. ``data/results/dataset_checksums.json`` records the checksums.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .config import Config, DataConfig, SplitConfig
from .lookup import N_OUTPUTS, load_lookup
from .simulate import evolve, evolve_cellpylib

#: Bump when the generation procedure changes in a way that changes the data.
GENERATOR_VERSION = 1

#: File names of the original Colab datasets (``create-dataset`` cell 10).
ORIGINAL_FILES = {
    "trainval": ("ECAs_1024sims-per-rule_64x64_train_diagrams.npy",
                 "ECAs_1024sims-per-rule_64x64_train_labels.npy"),
    "test": ("ECAs_128sims-per-rule_64x64_test_diagrams.npy",
             "ECAs_128sims-per-rule_64x64_test_labels.npy"),
}


@dataclass
class DiagramSet:
    """A set of binary spacetime diagrams, bit-packed along the space axis.

    Attributes
    ----------
    packed : (n, timesteps, ceil(width / 8)) uint8
        ``np.packbits(diagrams, axis=-1)``.
    rules : (n,) uint8
        The Wolfram rule that generated each diagram.
    width, timesteps : int
        Unpacked diagram shape is ``(timesteps, width)``.
    meta : dict
        How the set was made (seed, sizes, source); written into the file.
    """

    packed: np.ndarray
    rules: np.ndarray
    width: int
    timesteps: int
    meta: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.rules.shape[0])

    def diagrams(self, index=None) -> np.ndarray:
        """Unpacked diagrams, shape ``(n, timesteps, width)``, dtype uint8."""
        packed = self.packed if index is None else self.packed[index]
        return np.unpackbits(packed, axis=-1, count=self.width)

    def checksum(self) -> str:
        """SHA-256 of the packed diagrams and the rules (for reproducibility checks)."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.packed).tobytes())
        h.update(np.ascontiguousarray(self.rules).tobytes())
        return h.hexdigest()


# -- generation --------------------------------------------------------------

def generate_set(n_per_rule: int, width: int, timesteps: int, seed: Optional[int],
                 simulator: str = "numpy", rules=range(256)) -> DiagramSet:
    """Simulate ``n_per_rule`` diagrams of every rule from random initial rows.

    All initial rows of a rule are drawn in one call, rule after rule, from a
    single ``np.random.default_rng(seed)``. Each cell of an initial row is 0
    or 1 with probability 1/2, as ``cellpylib.init_random`` does.
    ``seed=None`` reproduces the unseeded behaviour of the original notebook.
    """
    rules = list(rules)
    rng = np.random.default_rng(seed)
    run = evolve if simulator == "numpy" else evolve_cellpylib
    n_bytes = math.ceil(width / 8)
    packed = np.empty((len(rules) * n_per_rule, timesteps, n_bytes), dtype=np.uint8)
    labels = np.empty(len(rules) * n_per_rule, dtype=np.uint8)
    for i, rule in enumerate(rules):
        initial = rng.integers(0, 2, size=(n_per_rule, width), dtype=np.uint8)
        block = slice(i * n_per_rule, (i + 1) * n_per_rule)
        packed[block] = np.packbits(run(initial, rule, timesteps), axis=-1)
        labels[block] = rule
    meta = dict(source="generated", n_per_rule=n_per_rule, width=width, timesteps=timesteps,
                seed=_seed_meta(seed), generator_version=GENERATOR_VERSION)
    return DiagramSet(packed, labels, width, timesteps, meta)


def _seed_meta(seed: Optional[int]) -> int:
    """Seed as stored in a file's metadata; -1 marks an unseeded set."""
    return -1 if seed is None else int(seed)


def generate_from_config(data: DataConfig, which: str) -> DiagramSet:
    """Generate the ``trainval`` or ``test`` set described by a config."""
    if which == "trainval":
        n, seed = data.n_per_rule_trainval, data.seed_trainval
    elif which == "test":
        n, seed = data.n_per_rule_test, data.seed_test
    else:
        raise ValueError(f"which must be 'trainval' or 'test', got {which!r}")
    return generate_set(n, data.width, data.timesteps, seed, data.simulator)


# -- storage -----------------------------------------------------------------

def save_set(ds: DiagramSet, path) -> None:
    """Write a set as an uncompressed ``.npz`` (packed diagrams, rules, metadata)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, packed=ds.packed, rules=ds.rules,
             **{f"meta_{k}": np.asarray(v) for k, v in ds.meta.items()})


def load_set(path) -> DiagramSet:
    """Read a set written by :func:`save_set`."""
    with np.load(Path(path), allow_pickle=False) as z:
        meta = {k[5:]: z[k].item() for k in z.files if k.startswith("meta_")}
        return DiagramSet(z["packed"], z["rules"], int(meta["width"]), int(meta["timesteps"]), meta)


def load_original(diagrams_path, labels_path) -> DiagramSet:
    """Read an original Colab dataset: ``(n, 64, 64, 1)`` uint8 diagrams, ``(n, 1)`` rules.

    The array is packed in chunks, so the 1.1 GB training file never needs a
    second full-size copy in memory.
    """
    diagrams = np.load(diagrams_path, mmap_mode="r")
    rules = np.load(labels_path).reshape(-1).astype(np.uint8)
    n, timesteps, width = diagrams.shape[:3]
    packed = np.empty((n, timesteps, math.ceil(width / 8)), dtype=np.uint8)
    for start in range(0, n, 16384):
        stop = min(start + 16384, n)
        packed[start:stop] = np.packbits(np.asarray(diagrams[start:stop, ..., 0]), axis=-1)
    meta = dict(source="original", n_per_rule=n // 256, width=width, timesteps=timesteps,
                file=str(Path(diagrams_path).name))
    return DiagramSet(packed, rules, width, timesteps, meta)


def dataset_path(cfg: Config, which: str) -> Path:
    return cfg.path("data_dir") / f"{which}.npz"


def load_datasets(cfg: Config, which=("trainval", "test")) -> Dict[str, DiagramSet]:
    """Load the sets an experiment needs, checking they match its config.

    For generated data, a mismatch between the file's metadata and the config
    (e.g. a different seed or size) is an error rather than a silent mix-up;
    re-run ``scripts/make_dataset.py``.
    """
    out = {}
    for w in which:
        if cfg.data.source == "original":
            d = cfg.path("original_data_dir")
            out[w] = load_original(d / ORIGINAL_FILES[w][0], d / ORIGINAL_FILES[w][1])
            continue
        path = dataset_path(cfg, w)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run: python scripts/make_dataset.py")
        ds = load_set(path)
        expected = dict(
            n_per_rule=cfg.data.n_per_rule_trainval if w == "trainval" else cfg.data.n_per_rule_test,
            width=cfg.data.width, timesteps=cfg.data.timesteps,
            seed=_seed_meta(cfg.data.seed_trainval if w == "trainval" else cfg.data.seed_test),
            generator_version=GENERATOR_VERSION)
        wrong = {k: (ds.meta.get(k), v) for k, v in expected.items() if ds.meta.get(k) != v}
        if wrong:
            raise ValueError(f"{path} does not match the config ({{key: (file, config)}}: {wrong}); "
                             "regenerate it with scripts/make_dataset.py")
        out[w] = ds
    return out


# -- labels and splits -------------------------------------------------------

def labels_for(cfg: Config, rules: np.ndarray) -> np.ndarray:
    """The training target of every diagram, for the config's ``model.output``."""
    return load_lookup(cfg.path("lookup_dir")).labels(rules, cfg.model.output)


def split_indices(rules: np.ndarray, labels: np.ndarray, split: SplitConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of the training and validation parts, made the way the notebooks made them.

    The notebooks shuffle once with ``np.random.seed(2023)`` and
    ``np.random.permutation``, then call
    ``train_test_split(test_size=int(n / 4), random_state=0, shuffle=True,
    stratify=labels)``. Both steps are repeated here.

    One detail matters. The notebooks' label arrays have shape ``(n, 1)`` (as
    written by ``create-dataset``). For a 2-D ``stratify`` array, scikit-learn
    turns every row into a string, so the classes are ordered as strings
    ("0", "1", "10", "100", ...) when it draws the split. With 256 rules or 88
    equivalence classes this gives a different split than 1-D integer labels
    would (only about a quarter of the validation set in common). The labels
    are therefore passed with shape ``(n, 1)``, in both modes: this is the
    notebooks' procedure, not a bug (``docs/provenance.md`` §7.1).

    On the original data this reproduces the notebooks' split, provided no
    other draw from numpy's global generator happened between the seed and
    the shuffle in the Colab session. The saved notebooks cannot show that.
    """
    from sklearn.model_selection import train_test_split

    n = len(rules)
    perm = np.random.RandomState(split.shuffle_seed).permutation(n)
    strat = np.asarray(rules if split.stratify_by == "rule" else labels).reshape(-1, 1)
    train_idx, val_idx = train_test_split(perm, test_size=int(n * split.val_fraction),
                                          random_state=split.split_seed, shuffle=True,
                                          stratify=strat[perm])
    return train_idx, val_idx


def one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Float32 one-hot vectors (``keras.utils.to_categorical``)."""
    out = np.zeros((len(labels), n_classes), dtype=np.float32)
    out[np.arange(len(labels)), np.asarray(labels, dtype=np.int64)] = 1.0
    return out


def example_indices(n_per_rule: int, rules) -> np.ndarray:
    """Index of the first diagram of each rule in a rule-ordered set."""
    return np.asarray(rules, dtype=np.int64) * n_per_rule


def n_outputs(cfg: Config) -> int:
    return N_OUTPUTS[cfg.model.output]
