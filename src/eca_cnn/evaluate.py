"""Accuracies, misclassifications and confusion matrices.

Used after training (:func:`evaluate_model`), for models saved by the original
notebooks (``scripts/evaluate_model.py``), and by the figures, which read the
predictions a run saved (:func:`load_run`) instead of re-running a network.

Accuracy and loss come from one ``predict`` pass rather than from
``model.evaluate``, and equal what ``model.evaluate`` reports. The accuracy is
Keras's ``accuracy`` (arg-max equals label). The loss is categorical
cross-entropy computed, as Keras does for a softmax output, from the logits:
:func:`log_probabilities` recomputes them from the last Dense layer, so a
confident wrong prediction is not clipped at -log(1e-7).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import Config
from .data import DiagramSet
from .lookup import Lookup

EPSILON = 1e-7  # tf.keras.backend.epsilon(); clipping only for outputs that are not a softmax Dense layer


def log_probabilities(model, seq) -> np.ndarray:
    """Log-probabilities of every output node, ``(n, n_outputs)`` float64.

    For a network ending in ``Dense(..., activation="softmax")`` (every network
    here and in the notebooks), the logits are recomputed from the layer's
    input and weights and normalised with log-sum-exp. Otherwise the predicted
    probabilities are clipped and logged.
    """
    import tensorflow as tf

    last = model.layers[-1]
    if isinstance(last, tf.keras.layers.Dense) and last.get_config().get("activation") == "softmax":
        features = tf.keras.Model(model.inputs, last.input).predict(seq, verbose=0).astype(np.float64)
        w, b = (a.astype(np.float64) for a in last.get_weights())
        logits = features @ w + b
        top = logits.max(axis=1, keepdims=True)
        return logits - (top + np.log(np.exp(logits - top).sum(axis=1, keepdims=True)))
    return np.log(np.clip(model.predict(seq, verbose=0).astype(np.float64), EPSILON, 1.0))


def predict(model, ds: DiagramSet, indices: np.ndarray, labels: np.ndarray, crop=None,
            batch_size: int = 1024) -> Dict[str, np.ndarray]:
    """Run ``model`` on ``ds[indices]`` and keep what the analyses need.

    Returns arrays aligned with ``indices``: ``index``, ``rule``, ``label``,
    ``pred`` (arg-max), ``p_label`` (probability given to the true label),
    ``p_pred`` (probability of the prediction) and ``nll`` (cross-entropy of
    each diagram, the per-diagram loss).
    """
    from .batches import DiagramSequence

    labels = np.asarray(labels)
    n_out = int(model.output_shape[-1])
    seq = DiagramSequence(ds, indices, None, n_out, batch_size=batch_size, crop=crop)
    logp = log_probabilities(model, seq)
    rows = np.arange(len(indices))
    pred = logp.argmax(axis=1)
    return {"index": np.asarray(indices), "rule": ds.rules[indices], "label": labels,
            "pred": pred.astype(np.int16), "p_label": np.exp(logp[rows, labels]).astype(np.float32),
            "p_pred": np.exp(logp[rows, pred]).astype(np.float32),
            "nll": (-logp[rows, labels]).astype(np.float32)}


def summarise(p: Dict[str, np.ndarray]) -> Dict:
    """Accuracy, loss and number of errors of one set of predictions."""
    wrong = p["pred"] != p["label"]
    loss = float(np.mean(p["nll"].astype(np.float64)))
    return {"n": int(len(wrong)), "n_wrong": int(wrong.sum()),
            "accuracy": round(float(1 - wrong.mean()), 6), "loss": round(loss, 6)}


def implied_class_accuracy(p: Dict[str, np.ndarray], output: str, lookup: Lookup) -> Optional[float]:
    """For a rule or equivalence-class model: how often the *class* comes out right.

    A wrongly identified rule may still lie in the right Li-Packard class. For
    ``classes`` models this is just the accuracy and ``None`` is returned.
    """
    if output == "rules":
        pred_rule = p["pred"].astype(np.int64)
    elif output == "independent":
        pred_rule = lookup.representatives[p["pred"].astype(np.int64)]
    else:
        return None
    true_cls = lookup.rule_to_class[p["rule"].astype(np.int64)]
    return round(float(np.mean(lookup.rule_to_class[pred_rule] == true_cls)), 6)


def confusion(labels: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Absolute confusion matrix: rows are actual labels, columns predictions."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(labels, np.int64), np.asarray(pred, np.int64)), 1)
    return cm


def normalise(cm: np.ndarray, over: str = "pred") -> np.ndarray:
    """Normalise a confusion matrix over predictions (columns) or actual labels (rows).

    The chapter's confusion matrix is normalised over the predictions, as
    ``sklearn.metrics.confusion_matrix(..., normalize='pred')`` does: each
    column sums to 1. Empty columns or rows give 0 rather than NaN.
    """
    axis = 0 if over == "pred" else 1
    totals = cm.sum(axis=axis, keepdims=True)
    return np.divide(cm, totals, out=np.zeros(cm.shape, dtype=float), where=totals > 0)


def errors_per_rule(p: Dict[str, np.ndarray]) -> np.ndarray:
    """Number of misidentified diagrams for each true rule (256 entries)."""
    wrong = p["pred"] != p["label"]
    return np.bincount(p["rule"][wrong].astype(np.int64), minlength=256)


def evaluate_model(model, cfg: Config, lookup: Lookup, which: Dict[str, tuple],
                   run_dir: Optional[Path] = None) -> Dict:
    """Predict on several sets, save the predictions, and return the summaries.

    ``which`` maps a set name to ``(DiagramSet, indices, labels)``.
    """
    out = {}
    for name, (ds, idx, labels) in which.items():
        p = predict(model, ds, idx, labels, crop=cfg.model.crop)
        s = summarise(p)
        implied = implied_class_accuracy(p, cfg.model.output, lookup)
        if implied is not None:
            s["implied_class_accuracy"] = implied
        if cfg.model.output == "classes":
            s["confusion"] = confusion(p["label"], p["pred"], 5).tolist()
        out[name] = s
        if run_dir is not None:
            np.savez_compressed(Path(run_dir) / f"predictions_{name}.npz", **p)
    return out


def load_run(run_dir) -> Dict:
    """``metrics.json`` of a run plus its saved predictions, keyed by set name."""
    run_dir = Path(run_dir)
    with open(run_dir / "metrics.json", encoding="utf-8") as fh:
        run = json.load(fh)
    run["predictions"] = {}
    for f in sorted(run_dir.glob("predictions_*.npz")):
        with np.load(f) as z:
            run["predictions"][f.stem[len("predictions_"):]] = {k: z[k] for k in z.files}
    return run
