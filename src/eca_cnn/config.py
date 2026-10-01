"""Experiment configuration: one YAML file per experiment, validated on load.

Every experiment in ``configs/`` is a YAML file that may inherit from another
(``inherits: base.yaml``) and override any field. The file is turned into a
:class:`Config` dataclass, and unknown keys are an error, so a typo cannot
silently fall back to a default.

Faithful and corrected mode
---------------------------
The original Colab notebooks contain two bugs that change published numbers
(``docs/provenance.md`` §3). The top-level field ``mode`` decides which
behaviour runs:

``corrected`` (the default)
    The bugs are fixed.
``faithful``
    The notebooks are reproduced, bugs included, so that the published numbers
    can be traced.

``mode`` switches exactly these behaviours, and nothing else:

1. Coarse-graining augmentation (``augment.coarse_grain``): faithful mode divides
   the block mean by ``64**2`` once more, so every coarse-grained diagram is
   almost black (§3.1).
2. Periodic padding in the rule-table scan (``benchmark.periodic_pad``): faithful
   mode pads with the first column on the left and the last on the right, which
   is the wrong way round (§3.2).
3. Class-ambiguity check in the scan (``benchmark.alternative_rules``): with
   more than one missing neighbourhood, faithful mode checks a single
   alternative rule (all missing outputs flipped), as the notebook does; corrected
   mode checks every completion of the rule table (§7.4). On the chapter's data
   no diagram misses more than one neighbourhood, so this cannot change a
   published number.

Everything else that differs from the notebooks (seeded generation, vectorised
simulation, bit-packed storage, the batch generator) is either bit-identical in
outcome or has its own config field below, documented where it is defined.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

#: Repository root: ``src/eca_cnn/config.py`` -> three levels up. Relative paths
#: in a config are resolved against it, so scripts work from any directory.
ROOT = Path(__file__).resolve().parents[2]

MODES = ("corrected", "faithful")
OUTPUTS = ("rules", "independent", "classes")
ARCHITECTURES = ("default", "trimmed", "exact", "mlp")
AUGMENTATIONS = ("invert", "mirror", "coarse_grain", "salt_and_pepper")
SPLITS = ("train", "val", "trainval", "test")


@dataclass
class PathsConfig:
    """Where data, lookup tables and outputs live (relative to the repo root)."""

    #: The three Li-Packard / equivalence JSON files (committed).
    lookup_dir: str = "data/lookup"
    #: Generated, bit-packed datasets (``trainval.npz``, ``test.npz``).
    data_dir: str = "data/generated"
    #: Figures, trained models, run metrics.
    output_dir: str = "output"
    #: Folder with the original Colab ``.npy`` files, used when
    #: ``data.source == "original"``. The originals were generated without a
    #: seed, so they are the only exact copy of the chapter's data.
    original_data_dir: Optional[str] = "${ECA_CNN_ORIGINAL_DATA}"


@dataclass
class DataConfig:
    """The spacetime-diagram datasets (chapter Sec. 1.3, "Data and objectives")."""

    #: ``generated``: the seeded datasets written by ``scripts/make_dataset.py``.
    #: ``original``: the unseeded Colab files from ``paths.original_data_dir``.
    source: str = "generated"
    #: Number of cells; periodic boundaries.
    width: int = 64
    #: Number of rows of a diagram, *including* the random initial row
    #: (cellpylib's ``timesteps`` convention, as used by ``create-dataset``).
    timesteps: int = 64
    #: Diagrams per rule in the training-and-validation set (262,144 in total).
    n_per_rule_trainval: int = 1024
    #: Diagrams per rule in the held-out test set (32,768 in total).
    n_per_rule_test: int = 128
    #: Seeds of the two sets. The original notebook set no seed (§4); these make
    #: the regenerated data identical on every machine.
    seed_trainval: int = 20230001
    seed_test: int = 20230002
    #: ``numpy`` (vectorised, fast) or ``cellpylib`` (the notebook's simulator,
    #: about 1000x slower). Both give bit-identical diagrams from the same
    #: initial rows; the choice only affects run time.
    simulator: str = "numpy"


@dataclass
class SplitConfig:
    """The 3/4 : 1/4 train/validation split of the training-and-validation set.

    The notebooks first shuffle with ``np.random.seed(2023)`` and then call
    ``sklearn.model_selection.train_test_split(..., random_state=0,
    stratify=...)``. This repository repeats both steps.
    """

    val_fraction: float = 0.25
    #: Seed of the pre-shuffle (``np.random.seed(2023)`` in every notebook).
    shuffle_seed: int = 2023
    #: ``random_state`` of ``train_test_split``.
    split_seed: int = 0
    #: What the split is stratified on. ``rule`` (the ``_trimmed`` and MLP
    #: notebooks) or ``label``, i.e. the training target (the ``_default`` and
    #: ``data-augmentation`` notebooks stratify on whatever label file they load:
    #: class, equivalence class or rule). See ``docs/provenance.md`` §7.1.
    stratify_by: str = "rule"


@dataclass
class ModelConfig:
    """Which network to build (``models.build_model``)."""

    #: ``default`` (Sec. 2.2), ``exact`` (Sec. 3.1), ``trimmed`` (Sec. 3.2) or
    #: ``mlp`` (the fully-connected comparison in Sec. 3.2).
    architecture: str = "trimmed"
    #: ``rules`` (256 outputs), ``independent`` (88 equivalence classes) or
    #: ``classes`` (5 Li-Packard classes).
    output: str = "classes"
    #: Convolution kernel as Keras ``(rows, columns)`` = (time, space). Keras
    #: ``(2, 3)`` is the chapter's "3 x 2" T-tetromino window. ``None`` takes the
    #: architecture's own default: (2, 3) for ``default``, (2, 2) for ``trimmed``.
    kernel: Optional[List[int]] = None
    #: Hidden layer widths of the ``mlp`` architecture.
    hidden_units: Optional[List[int]] = None
    #: Crop for the ``mlp`` architecture, as ``[rows, columns]`` taken from the
    #: top-left corner of the diagram (``MLP-verification_cropped`` uses 28 x 28).
    crop: Optional[List[int]] = None


@dataclass
class AugmentConfig:
    """Online augmentation (chapter Sec. 3.3). Empty list: no augmentation."""

    #: Subset of ``invert``, ``mirror``, ``coarse_grain``, ``salt_and_pepper``.
    #: Each diagram receives exactly one of the n listed augmentations with
    #: probability n/(n+1), and none otherwise.
    augmentations: List[str] = field(default_factory=list)
    #: Fraction of pixels flipped by salt-and-pepper noise. The notebook
    #: hard-codes 0.01 (its constructor argument is ignored, §4), which is the
    #: chapter's "1%".
    salt_and_pepper_fraction: float = 0.01
    #: Block size of coarse-graining (the chapter's "2 x 2").
    coarse_grain_block: int = 2


@dataclass
class EarlyStoppingConfig:
    """``keras.callbacks.EarlyStopping`` as configured in the notebooks."""

    enabled: bool = True
    monitor: str = "val_accuracy"
    patience: int = 5
    min_delta: float = 1e-5
    restore_best_weights: bool = True


@dataclass
class TrainConfig:
    """Optimiser and schedule (chapter Sec. 2.2, "Training strategy")."""

    batch_size: int = 64
    #: Maximum number of epochs (early stopping may end training sooner).
    epochs: int = 50
    #: Adam learning rate.
    learning_rate: float = 1e-3
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    #: ``train``: fit on the 3/4 training part and monitor the 1/4 validation
    #: part. ``trainval``: fit on all 262,144 diagrams without validation (the
    #: chapter's final model).
    train_on: str = "train"
    #: Sets to report accuracies and save predictions for, after training.
    evaluate_on: List[str] = field(default_factory=lambda: ["train", "val"])
    #: Seeds weight initialisation, batch order and augmentation draws.
    seed: int = 2023
    #: ``tf.config.experimental.enable_op_determinism()``: identical results on
    #: re-runs on the same hardware, at some cost in speed.
    deterministic_ops: bool = True


@dataclass
class Config:
    """A complete experiment."""

    name: str
    description: str = ""
    #: ``corrected`` or ``faithful``; see the module docstring.
    mode: str = "corrected"
    #: The archived notebook this experiment reproduces, and the chapter section.
    source_notebook: str = ""
    chapter_reference: str = ""
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    #: Optional grid of overrides, ``{"dotted.key": [value, ...]}``. Each
    #: combination is one run ("cell"); see :func:`expand_sweep`.
    sweep: Optional[Dict[str, List[Any]]] = None

    # -- derived ------------------------------------------------------------
    @property
    def faithful(self) -> bool:
        """True when the notebooks' bugs are to be reproduced."""
        return self.mode == "faithful"

    @property
    def training_depends_on_mode(self) -> bool:
        """Whether faithful mode changes what this experiment trains.

        Of the three behaviours ``mode`` switches, only coarse-graining affects
        training; the other two live in the benchmark scan.
        """
        return "coarse_grain" in self.augment.augmentations

    @property
    def run_name(self) -> str:
        """Name of the output folder.

        Faithful runs get a ``__faithful`` suffix only when the mode changes
        the training. Otherwise both modes train the same network and share
        one folder.
        """
        return self.name + ("__faithful" if self.faithful and self.training_depends_on_mode else "")

    def path(self, which: str) -> Path:
        """Absolute path of a ``paths`` entry (relative ones resolve to the root).

        Environment variables are expanded, so ``${ECA_CNN_ORIGINAL_DATA}`` in a
        config points wherever that variable does.
        """
        value = getattr(self.paths, which)
        if value is None:
            raise ValueError(f"paths.{which} is not set in config {self.name!r}")
        expanded = os.path.expandvars(str(value))
        if "$" in expanded:
            raise ValueError(f"paths.{which} = {value!r}: set the environment variable it refers to")
        p = Path(expanded).expanduser()
        return p if p.is_absolute() else ROOT / p

    def run_dir(self) -> Path:
        return self.path("output_dir") / "runs" / self.run_name

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> "Config":
        """Reject values the code would otherwise misinterpret."""
        _check(self.mode in MODES, f"mode must be one of {MODES}, got {self.mode!r}")
        _check(self.data.source in ("generated", "original"), f"data.source: {self.data.source!r}")
        _check(self.data.simulator in ("numpy", "cellpylib"), f"data.simulator: {self.data.simulator!r}")
        _check(self.split.stratify_by in ("rule", "label"), f"split.stratify_by: {self.split.stratify_by!r}")
        _check(0 < self.split.val_fraction < 1, "split.val_fraction must lie in (0, 1)")
        _check(self.model.architecture in ARCHITECTURES, f"model.architecture: {self.model.architecture!r}")
        _check(self.model.output in OUTPUTS, f"model.output: {self.model.output!r}")
        for aug in self.augment.augmentations:
            _check(aug in AUGMENTATIONS, f"unknown augmentation {aug!r}; choose from {AUGMENTATIONS}")
        _check(len(set(self.augment.augmentations)) == len(self.augment.augmentations),
               "augment.augmentations lists an augmentation twice")
        _check(self.train.train_on in ("train", "trainval"), f"train.train_on: {self.train.train_on!r}")
        for s in self.train.evaluate_on:
            _check(s in SPLITS, f"train.evaluate_on: unknown set {s!r}")
        if self.train.train_on == "trainval":
            _check("val" not in self.train.evaluate_on,
                   "train_on: trainval leaves no validation set to evaluate on")
            _check(not self.train.early_stopping.enabled or
                   not self.train.early_stopping.monitor.startswith("val_"),
                   "early stopping on a validation metric needs train_on: train")
        if self.model.architecture == "mlp":
            _check(bool(self.model.hidden_units), "the mlp architecture needs model.hidden_units")
        return self


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# -- loading ----------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (lists are replaced)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml_with_inheritance(path: Path, _seen: tuple = ()) -> Dict[str, Any]:
    path = path.resolve()
    if path in _seen:
        raise ValueError(f"circular 'inherits' chain: {' -> '.join(map(str, _seen + (path,)))}")
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.pop("inherits", None)
    if parent is None:
        return raw
    base = _read_yaml_with_inheritance(path.parent / parent, _seen + (path,))
    # The name is the one thing a child must not inherit: two experiments with
    # the same name would write to the same output folder. Without its own
    # ``name``, a child is named after its file (see load_config).
    base.pop("name", None)
    return _deep_merge(base, raw)


def _build(cls, data: Dict[str, Any], where: str):
    """Instantiate a (nested) dataclass, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(known)}")
    kwargs = {}
    for name, value in data.items():
        default = known[name].default_factory if known[name].default_factory is not dataclasses.MISSING else None
        sub = default() if default is not None else None
        if dataclasses.is_dataclass(sub) and isinstance(value, dict):
            kwargs[name] = _build(type(sub), value, f"{where}.{name}")
        else:
            kwargs[name] = _coerce(value, known[name].type, f"{where}.{name}")
    return cls(**kwargs)


def _coerce(value: Any, annotation: str, where: str) -> Any:
    """Make numeric fields numeric.

    YAML reads ``1e-3`` (without a decimal point) as a string, which would only
    fail deep inside Keras. Fields annotated ``float`` or ``int`` are converted
    here, or rejected with a clear message.
    """
    if annotation not in ("float", "int") or isinstance(value, bool):
        return value
    try:
        return float(value) if annotation == "float" else int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{where} must be a number, got {value!r}") from None


def set_dotted(data: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``data["a"]["b"] = value`` for ``dotted == "a.b"`` (creating levels)."""
    keys = dotted.split(".")
    node = data
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value


def parse_overrides(pairs: Optional[List[str]]) -> Dict[str, Any]:
    """Turn ``["train.epochs=3", "mode=faithful"]`` into a nested dict.

    Values are parsed as YAML, so ``3`` is an int, ``[invert, mirror]`` a list
    and ``null`` is ``None``.
    """
    out: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"override {pair!r} is not of the form key=value")
        key, raw = pair.split("=", 1)
        set_dotted(out, key.strip(), yaml.safe_load(raw))
    return out


def load_config(path, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load, merge with ``overrides`` (a nested dict) and validate a config file."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    data = _read_yaml_with_inheritance(path)
    if overrides:
        data = _deep_merge(data, overrides)
    if "name" not in data:
        data["name"] = path.stem
    return _build(Config, data, path.name).validate()


def expand_sweep(cfg: Config) -> List[Config]:
    """One config per combination of the ``sweep`` grid, in a fixed order.

    The cell's name is the base name followed by a readable label of each swept
    value, e.g. ``augmentation_table__invert_classes``. Without a sweep, the
    config itself is the only cell.
    """
    if not cfg.sweep:
        return [cfg]
    import itertools

    keys = list(cfg.sweep)
    cells = []
    for combo in itertools.product(*(cfg.sweep[k] for k in keys)):
        data = cfg.to_dict()
        data["sweep"] = None
        labels = []
        for key, value in zip(keys, combo):
            set_dotted(data, key, value)
            labels.append(_label(value))
        data["name"] = f"{cfg.name}__{'_'.join(labels)}"
        cells.append(_build(Config, data, data["name"]).validate())
    return cells


def _label(value: Any) -> str:
    """Short name of a swept value: ``none``, ``all`` (every augmentation) or the items."""
    if isinstance(value, (list, tuple)):
        if not value:
            return "none"
        if sorted(value) == sorted(AUGMENTATIONS):
            return "all"
        return "+".join(map(str, value))
    return str(value)


def save_config(cfg: Config, path: Path) -> None:
    """Write the fully resolved config next to a run's outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False)
