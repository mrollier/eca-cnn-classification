"""Training with the chapter's hyperparameters (Sec. 2.2, 3.2 and 3.4).

All settings come from the experiment's config. The notebooks' recipe is:

- one-hot labels and categorical cross-entropy;
- Adam, batch size 64 (3072 updates per epoch on the 196,608 training diagrams);
- early stopping on validation accuracy with ``restore_best_weights=True``.
  As in Keras, the best weights are restored only if training stops early; a
  run that reaches the epoch limit keeps its last weights.

A run writes into ``output/runs/<run name>/``:

``config.yaml``       the fully resolved config
``model.keras``       the trained network (Keras 2 format)
``history.csv``       per-epoch loss and accuracy (training batches are augmented)
``metrics.json``      accuracies on the sets in ``train.evaluate_on``, run time, versions
``predictions_<set>.npz``  per-diagram predictions, read by the figures
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .cli import environment, write_json
from .config import Config, save_config
from .data import DiagramSet, load_datasets, split_indices
from .lookup import N_OUTPUTS, Lookup, load_lookup

#: In a smoke run: diagrams per rule kept from each part, and epochs.
SMOKE_PER_RULE = {"train": 6, "val": 2, "test": 2}


def configure_tensorflow(seed: int, deterministic: bool) -> None:
    """Seed Python, numpy and TensorFlow; optionally force deterministic kernels."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    if deterministic:
        tf.config.experimental.enable_op_determinism()


@dataclass
class Prepared:
    """Data, labels and index sets of one experiment."""

    lookup: Lookup
    sets: Dict[str, DiagramSet]
    labels: Dict[str, np.ndarray]   # per set name in ``sets``
    train_idx: np.ndarray
    val_idx: Optional[np.ndarray]


def _per_rule_subset(rules: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    """The first ``k`` entries of ``idx`` for every rule (keeps the class balance)."""
    keep, seen = [], np.zeros(256, dtype=np.int64)
    for i in idx:
        r = rules[i]
        if seen[r] < k:
            keep.append(i)
            seen[r] += 1
    return np.asarray(keep)


def prepare(cfg: Config, smoke: bool = False) -> Prepared:
    """Load the data and make the train/validation split the config asks for."""
    lookup = load_lookup(cfg.path("lookup_dir"))
    needed = ["trainval"] + (["test"] if "test" in cfg.train.evaluate_on else [])
    sets = load_datasets(cfg, needed)
    labels = {k: lookup.labels(ds.rules, cfg.model.output) for k, ds in sets.items()}
    tv = sets["trainval"]
    if cfg.train.train_on == "train":
        train_idx, val_idx = split_indices(tv.rules, labels["trainval"], cfg.split)
    else:
        train_idx, val_idx = np.arange(len(tv)), None
    if smoke:
        train_idx = _per_rule_subset(tv.rules, train_idx, SMOKE_PER_RULE["train"])
        if val_idx is not None:
            val_idx = _per_rule_subset(tv.rules, val_idx, SMOKE_PER_RULE["val"])
    return Prepared(lookup, sets, labels, train_idx, val_idx)


def _timing_callback():
    import tensorflow as tf

    class Timing(tf.keras.callbacks.Callback):
        """Wall time per epoch, and the median time of a training batch."""

        def on_train_begin(self, logs=None):
            self.epoch_seconds, self.batch_seconds = [], []

        def on_epoch_begin(self, epoch, logs=None):
            self._t_epoch = time.perf_counter()

        def on_epoch_end(self, epoch, logs=None):
            self.epoch_seconds.append(round(time.perf_counter() - self._t_epoch, 2))

        def on_train_batch_begin(self, batch, logs=None):
            self._t_batch = time.perf_counter()

        def on_train_batch_end(self, batch, logs=None):
            self.batch_seconds.append(time.perf_counter() - self._t_batch)

    return Timing()


def train(cfg: Config, smoke: bool = False, verbose: int = 2) -> Dict:
    """Train the network of ``cfg``, evaluate it and save everything; return the metrics.

    ``smoke=True`` trains for one epoch on a few diagrams per rule
    (``SMOKE_PER_RULE``) to check that a config runs end to end. It also
    estimates how long one full epoch would take on this machine. The run is
    written to ``<run name>__smoke``.
    """
    configure_tensorflow(cfg.train.seed, cfg.train.deterministic_ops)
    import tensorflow as tf

    from .augment import Augmenter
    from .batches import DiagramSequence
    from .evaluate import evaluate_model
    from .models import build_model, count_parameters

    if cfg.model.architecture == "exact":
        raise ValueError("the exact CNN is not trained; use scripts/run_exact_cnn.py")

    run_dir = cfg.run_dir()
    if smoke:
        run_dir = run_dir.with_name(run_dir.name + "__smoke")
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    prep = prepare(cfg, smoke)
    tv, y_tv = prep.sets["trainval"], prep.labels["trainval"]
    n_out = N_OUTPUTS[cfg.model.output]
    augmenter = Augmenter.from_config(cfg) if cfg.augment.augmentations else None
    train_seq = DiagramSequence(tv, prep.train_idx, y_tv[prep.train_idx], n_out, cfg.train.batch_size,
                                shuffle=True, seed=cfg.train.seed, augmenter=augmenter, crop=cfg.model.crop)
    val_seq = None
    if prep.val_idx is not None:
        val_seq = DiagramSequence(tv, prep.val_idx, y_tv[prep.val_idx], n_out, cfg.train.batch_size,
                                  crop=cfg.model.crop)

    model = build_model(cfg)
    model.compile(loss="categorical_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.train.learning_rate),
                  metrics=["accuracy"])

    timing = _timing_callback()
    callbacks = [timing, tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv"))]
    es_cfg, early = cfg.train.early_stopping, None
    if es_cfg.enabled:
        early = tf.keras.callbacks.EarlyStopping(monitor=es_cfg.monitor, patience=es_cfg.patience,
                                                 min_delta=es_cfg.min_delta, verbose=1,
                                                 restore_best_weights=es_cfg.restore_best_weights)
        callbacks.append(early)

    epochs = 1 if smoke else cfg.train.epochs
    t0 = time.perf_counter()
    history = model.fit(train_seq, validation_data=val_seq, epochs=epochs, callbacks=callbacks,
                        shuffle=False, verbose=verbose)
    train_seconds = time.perf_counter() - t0
    model.save(run_dir / "model.keras")

    which = {}
    for name in cfg.train.evaluate_on:
        if name == "train":
            which[name] = (tv, prep.train_idx, y_tv[prep.train_idx])
        elif name == "val":
            which[name] = (tv, prep.val_idx, y_tv[prep.val_idx])
        elif name == "trainval":
            idx = prep.train_idx if smoke else np.arange(len(tv))
            which[name] = (tv, idx, y_tv[idx])
        elif name == "test":
            ts = prep.sets["test"]
            idx = np.arange(len(ts))
            if smoke:
                idx = _per_rule_subset(ts.rules, idx, SMOKE_PER_RULE["test"])
            which[name] = (ts, idx, prep.labels["test"][idx])
    results = evaluate_model(model, cfg, prep.lookup, which, run_dir)

    epochs_run = len(history.epoch)
    metrics = {
        "run": run_dir.name, "smoke": smoke, "mode": cfg.mode,
        "architecture": cfg.model.architecture, "output": cfg.model.output,
        "augmentations": cfg.augment.augmentations,
        "n_parameters": count_parameters(model),
        "n_train": int(len(prep.train_idx)), "n_val": None if prep.val_idx is None else int(len(prep.val_idx)),
        "epochs_run": epochs_run,
        "stopped_early": bool(early is not None and early.stopped_epoch > 0),
        "best_epoch": (int(getattr(early, "best_epoch", -1)) + 1) if early is not None else None,
        "best_val_accuracy": (round(float(max(history.history["val_accuracy"])), 6)
                              if "val_accuracy" in history.history else None),
        "train_seconds": round(train_seconds, 1),
        "epoch_seconds": timing.epoch_seconds,
        "results": results,
        "environment": environment(),
    }
    if smoke and timing.batch_seconds:
        per_batch = float(np.median(timing.batch_seconds[1:] or timing.batch_seconds))
        full_batches = int(np.ceil((len(tv) * (1 - cfg.split.val_fraction) if cfg.train.train_on == "train"
                                    else len(tv)) / cfg.train.batch_size))
        metrics["estimated_full_epoch_seconds"] = round(per_batch * full_batches, 1)
    write_json(metrics, run_dir / "metrics.json")
    return metrics
