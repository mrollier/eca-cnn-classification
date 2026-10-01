"""The chapter's figures, one function per figure, named after the thesis file.

Each function takes the data it plots and returns a matplotlib ``Figure``.
``scripts/make_figures.py`` gathers the data (datasets, benchmark scans,
predictions saved by training runs) and saves every figure to
``output/figures/<thesis file name>.pdf``.

=============================================  =====================================  ==========================
Thesis figure file                             Function                               Needs
=============================================  =====================================  ==========================
LP_classes-examples.pdf                        :func:`lp_classes_examples`            data
benchmark-undetermined_diagrams.pdf            :func:`benchmark_undetermined`         data
benchmark-undetermined_diagrams-examples2.pdf  :func:`benchmark_undetermined_examples` data
example-convolution_and_relu.pdf               :func:`example_convolution_and_relu`   data
convolutional_decomposition-example_120.pdf    :func:`convolutional_decomposition`    data
augmentation-examples.pdf                      :func:`augmentation_examples`          data
default_model-misidentified_val_set.pdf        :func:`default_model_misidentified`    run ``default_rules``
default_model-all_wrong_LP_classes.pdf         :func:`default_model_all_wrong`        run ``default_classes``
confusion_matrix-final.pdf                     :func:`confusion_matrix_final`         run ``final_classes``
default_CNN-architecture.jpg,                  (not reproduced: drawn with NN-SVG,
trimmed_CNN-architecture.jpg                   not by the notebooks)
=============================================  =====================================  ==========================

Colours. By default the figures use the notebooks' colours (dark green for
the training set, maroon for the test set, dodger blue for class sizes,
``YlGn`` for the confusion matrix), so they can be compared with the published
ones. Dark green and maroon are hard to tell apart with deuteranopia (CVD
Delta E 4.8, below the usual floor of 8). ``palette="accessible"`` swaps in a
blue / orange / aqua set that passes that check.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from .augment import coarse_grain, invert, mirror, salt_and_pepper
from .benchmark import NEIGHBOURHOODS, ScanResult
from .lookup import CLASS_NAMES, CLASS_TICKLABELS, Lookup

PALETTES = {
    # The notebooks' colours (benchmark-classification-by-scanning cell 15).
    "published": {"train": "darkgreen", "test": "maroon", "size": "dodgerblue", "alpha": 0.5,
                  "cmap": "YlGn"},
    # Passes the colour-vision-deficiency check (validated with the dataviz
    # skill's validate_palette.js: worst adjacent deutan Delta E 9.2).
    "accessible": {"train": "#2a78d6", "test": "#eb6834", "size": "#1baf7a", "alpha": 0.75,
                   "cmap": "YlGn"},
}

#: Rules whose diagrams illustrate the five classes: the last rule of each
#: class in Li and Packard's table (notebooks: ``rules_example``).
CLASS_EXAMPLE_RULES = (168, 232, 178, 154, 161)


def _blank(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def _percent_axis(ax, top: float, max_ticks: int = 9) -> None:
    """Y axis from 0 to ``top`` (a fraction), ticked at round percentages."""
    step = next(s for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5) if top / s <= max_ticks - 1)
    top = float(np.ceil(top / step - 1e-9) * step)
    ticks = np.arange(0, top + step / 2, step)
    ax.set_ylim(0, top)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{100 * t:.0f}%" for t in ticks])


def _nice_top(x: float) -> float:
    """Round a fraction up to the next 1% (or 10% above 20%)."""
    step = 0.1 if x > 0.2 else 0.01
    return max(step, float(np.ceil(x / step - 1e-9) * step))


# -- figures that need only data ----------------------------------------------

def lp_classes_examples(diagrams: Sequence[np.ndarray], rules: Sequence[int]):
    """``LP_classes-examples``: one diagram per Li-Packard class (Fig. input-examples).

    From the ``_default`` notebook, cell 8: the first diagram of rules 168,
    232, 178, 154 and 161.
    """
    names = ["Null", "Fixed", "Periodic", "Locally Chaotic", "Chaotic"]
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.9))
    for ax, d, r, name in zip(axs, diagrams, rules, names):
        ax.imshow(d, cmap="Greys", interpolation="none")
        ax.set_title(f"Class: {name}\nRule: {r}", size=16)
        _blank(ax)
    return fig


def benchmark_undetermined(train: ScanResult, test: ScanResult, lookup: Lookup, palette: str = "published"):
    """``benchmark-undetermined_diagrams``: where the rule-table scan is incomplete.

    Left: density histogram over the 256 rules of the undetermined diagrams in
    the training-and-validation set and the test set. Right: the same
    aggregated per Li-Packard class, next to each class's share of the rules.
    From ``benchmark-classification-by-scanning`` cell 15.
    """
    c = PALETTES[palette]
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), gridspec_kw={"width_ratios": (3, 1.3)})
    plt.subplots_adjust(wspace=0.15)
    bins = np.arange(257) - 0.5
    tops = []
    for res, key, label in ((train, "train", "Training set"), (test, "test", "Test set")):
        if res.n_undetermined:
            h, _, _ = axs[0].hist(res.rules, bins=bins, density=True, color=c[key], alpha=c["alpha"],
                                  label=f"{label} ({res.n_undetermined} diagrams)")
            tops.append(h.max())
    axs[0].set_xlim(-1, 256)
    xt = np.append(np.arange(0, 256, 32), 255)
    axs[0].set_xticks(xt)
    axs[0].set_xticklabels(xt, size=12)
    _percent_axis(axs[0], _nice_top(max(tops, default=0.01) * 1.25))
    axs[0].legend(loc="upper left", ncols=2, prop={"size": 12})
    axs[0].set_title("Spacetime diagrams with incomplete information. Distribution of local update rules",
                     fontsize=14)

    w = 0.2
    x = np.arange(5)
    shares = []
    for res, key, off, label in ((train, "train", -w, "Training set"), (test, "test", 0, "Test set")):
        per_class = res.per_class(lookup)
        share = per_class / max(per_class.sum(), 1)
        shares.append(share.max())
        axs[1].bar(x + off, share, width=w, color=c[key], alpha=c["alpha"], label=label)
    sizes = lookup.class_sizes() / 256
    axs[1].bar(x + w, sizes, width=w, color=c["size"], alpha=c["alpha"], label="Class size")
    axs[1].set_xlim(-0.5, 4.5)
    _percent_axis(axs[1], _nice_top(max(max(shares), sizes.max()) * 1.05), 5)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(CLASS_TICKLABELS, size=12)
    axs[1].legend(prop={"size": 12})
    axs[1].set_title("Distribution of Li-Packard classes", fontsize=14)
    return fig


def benchmark_undetermined_examples(examples: List[Dict]):
    """``benchmark-undetermined_diagrams-examples2``: two diagrams with a missing entry.

    Each example is a dict with ``diagram``, ``rule``, ``class`` and
    ``missing`` (the absent neighbourhoods). In the chapter: rule 59
    (periodic, missing 000), the only diagram whose class the scan could get
    wrong, and a chaotic diagram missing 111. From
    ``benchmark-classification-by-scanning`` cell 17.
    """
    fig, axs = plt.subplots(1, len(examples), figsize=(4.2 * len(examples), 4), squeeze=False)
    for ax, ex in zip(axs[0], examples):
        ax.imshow(ex["diagram"], cmap="Greys", interpolation="none")
        missing = ", ".join(ex["missing"])
        ax.set_title(f"Rule {ex['rule']} ({ex['class']})\nMissing pattern: {missing}", fontsize=18)
        _blank(ax)
    return fig


def _convolve(diagram: np.ndarray, kernel: np.ndarray, bias: float, relu: bool = False) -> np.ndarray:
    """Cross-correlation as a Keras Conv2D computes it (``convolve2d`` with a flipped kernel)."""
    from scipy.signal import convolve2d

    out = convolve2d(diagram, np.flip(kernel), mode="valid") + bias
    return np.maximum(out, 0) if relu else out


def example_convolution_and_relu(diagram: np.ndarray):
    """``example-convolution_and_relu``: one kernel that detects ``(1, 1, 0) -> 1``.

    The central 32 x 32 part of a diagram, convolved with the kernel
    ``[[1, 1, -1], [0, 1, 0]]`` and bias -2, before and after a ReLU. From
    ``create-illustration-of-convolution`` cell 7.
    """
    crop = diagram[16:-16, 16:-16]
    kernel, bias = np.array([[1, 1, -1], [0, 1, 0]]), -2
    conv = _convolve(crop, kernel, bias)
    fs = 22
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    axs[0].imshow(crop, cmap="Greys", interpolation="none")
    axs[0].set_title("Input\ndiagram", size=fs)
    axs[1].imshow(kernel, cmap="Greys")
    axs[1].set_title(f"Convolution kernel\n(bias: {bias})", size=fs - 4)
    for (j, i), v in np.ndenumerate(kernel):
        t = axs[1].text(i, j, v, ha="center", va="center", size=fs + 2)
        t.set_bbox(dict(facecolor="white", alpha=0.5, linewidth=0))
    axs[2].imshow(conv, cmap="Greys", interpolation="none")
    axs[2].set_title("Diagram after\nconvolution", size=fs)
    xs = np.linspace(-1, 1, 101)
    axs[3].plot(xs, np.maximum(xs, 0) - 0.6, "k")
    axs[3].set_ylim(-1, 1)
    axs[3].set_title("ReLU", size=fs - 4, y=0.8)
    axs[3].axis("off")
    axs[4].imshow(np.maximum(conv, 0), cmap="Greys", interpolation="none")
    axs[4].set_title("Diagram after\nconv. and ReLU", size=fs)
    for ax in axs:
        _blank(ax)
    return fig


def convolutional_decomposition(diagram: np.ndarray, rule: int):
    """``convolutional_decomposition-example_120``: the exact CNN's eight channels.

    The diagram (periodically padded) passes through the exact CNN's
    convolution and ReLU. Each channel lights up where its neighbourhood
    occurs and produces a 1; its maximum (boxed) is one bit of the rule. From
    ``custom-made-cnn`` cell 9. The chapter uses a diagram of rule 120.
    """
    from .models import exact_weights

    kernel, bias, _, _ = exact_weights()
    padded = np.pad(diagram, ((0, 0), (1, 1)), mode="wrap")
    fig, axs = plt.subplots(1, 9, figsize=(20, 4))
    axs[0].imshow(diagram, cmap="Greys", interpolation="none")
    axs[0].set_title(f"Rule {rule}")
    _blank(axs[0])
    for i, k in enumerate(range(7, -1, -1)):  # 111 first, as in the notebook
        ch = _convolve(padded, kernel[:, :, 0, k], bias[k], relu=True)
        ax = axs[i + 1]
        ax.imshow(ch, cmap="Greys", interpolation="none", vmin=0, vmax=1)
        ax.set_title(f"({', '.join(NEIGHBOURHOODS[k])}) ↦ 1")
        t = ax.text(ch.shape[1] / 2, ch.shape[0] / 2, f" {int(ch.max())} ", ha="center", va="center",
                    size=42, weight="bold")
        t.set_bbox(dict(facecolor="grey", alpha=0.3, linewidth=3))
        _blank(ax)
    return fig


def augmentation_examples(diagram: np.ndarray, salt_and_pepper_fraction: float = 0.05, seed: int = 2023):
    """``augmentation-examples``: a chaotic diagram and its four augmentations.

    From ``data-augmentation`` cell 11 (first diagram of rule 161). The
    caption says 5% salt-and-pepper noise; the notebook's saved state uses
    10% (docs/provenance.md §7.3). The fraction is a parameter.
    """
    x = diagram.astype(np.float32)[None, ..., None]
    rng = np.random.default_rng(seed)
    panels = [("Original", x), ("Inverted", invert(x)), ("Mirrored", mirror(x)),
              ("Coarse-grained\n(level 2)", coarse_grain(x)),
              (f"Salt and pepper\n(level {salt_and_pepper_fraction:g})",
               salt_and_pepper(x, salt_and_pepper_fraction, rng))]
    fig, axs = plt.subplots(1, 5, figsize=(16, 3.9))
    for ax, (name, img) in zip(axs, panels):
        ax.imshow(img[0, ..., 0], cmap="Greys", interpolation="none")
        ax.set_title(name, size=16)
        _blank(ax)
    return fig


# -- figures that need a trained model's predictions ---------------------------

def default_model_misidentified(pred: Dict[str, np.ndarray], lookup: Lookup, palette: str = "published"):
    """``default_model-misidentified_val_set``: rules the default CNN gets wrong.

    ``pred`` holds the validation predictions of the ``default_rules`` run.
    Left: share of the misidentified diagrams per true rule. Right: per
    Li-Packard class of the true rule, next to the class sizes. From the
    ``_default`` notebook, cell 16.
    """
    c = PALETTES[palette]
    wrong = pred["pred"] != pred["label"]
    rules = pred["rule"][wrong].astype(np.int64)
    n_wrong = int(wrong.sum())
    per_rule = np.bincount(rules, minlength=256) / max(n_wrong, 1)
    per_class = np.bincount(lookup.rule_to_class[rules], minlength=5) / max(n_wrong, 1)
    sizes = lookup.class_sizes() / 256

    fig, axs = plt.subplots(1, 2, figsize=(16, 3.5), gridspec_kw={"width_ratios": (3, 1)})
    plt.subplots_adjust(wspace=0.15)
    axs[0].bar(np.arange(256), per_rule, width=1, color=c["train"], alpha=c["alpha"],
               label="Real rule of misidentified diagram")
    axs[0].set_xlim(-1, 256)
    _percent_axis(axs[0], _nice_top(per_rule.max() * 1.1))
    w = 0.3
    axs[1].bar(np.arange(5) - w / 2, per_class, width=w, color=c["train"], alpha=c["alpha"],
               label="Real class of\nmisidentified rule")
    axs[1].bar(np.arange(5) + w / 2, sizes, width=w, color=c["size"], alpha=c["alpha"], label="Class size")
    axs[1].set_xlim(-0.5, 4.5)
    _percent_axis(axs[1], _nice_top(max(per_class.max(), sizes.max()) * 1.05), 5)
    axs[1].set_xticks(range(5))
    axs[1].set_xticklabels(CLASS_TICKLABELS)
    axs[0].set_title(f"All {n_wrong} misidentified diagrams from validation set (size {len(wrong)}), per rule",
                     fontsize=14)
    axs[1].set_title("... belonging to LP class", fontsize=14)
    axs[1].legend(prop={"size": 10})
    return fig


def default_model_all_wrong(diagrams: np.ndarray, labels: np.ndarray, preds: np.ndarray, max_panels: int = 12):
    """``default_model-all_wrong_LP_classes``: every misclassified validation diagram.

    The chapter shows all six diagrams that the default CNN puts in the wrong
    class. From the ``_default`` notebook, cell 19. Shows at most
    ``max_panels`` diagrams, six per row.
    """
    n = min(len(diagrams), max_panels)
    if n == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No misclassified diagrams", ha="center", va="center", size=16)
        ax.axis("off")
        return fig
    cols = min(n, 6)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(28 * cols / 6, 4.6 * rows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for i in range(n):
        ax = axs.ravel()[i]
        ax.axis("on")
        ax.imshow(diagrams[i], cmap="Greys", interpolation="none")
        ax.set_title(f"Label: {CLASS_NAMES[labels[i]]},\nprediction: {CLASS_NAMES[preds[i]]}", size=20)
        _blank(ax)
    return fig


def confusion_matrix_final(cm_abs: np.ndarray, cm_independent: Optional[np.ndarray] = None,
                           palette: str = "published"):
    """``confusion_matrix-final``: the final model on the test set.

    Rows are the actual class, columns the prediction. The colours and the
    percentages are normalised over the predictions (each column sums to
    100%), as in the chapter; the absolute counts are in parentheses. From
    ``data-augmentation`` cell 26. With ``cm_independent`` (88 x 88) a second
    panel shows the equivalence-class model, as the notebook does; the
    chapter prints only the first.
    """
    from .evaluate import normalise

    cmap = PALETTES[palette]["cmap"]
    two = cm_independent is not None
    fig, axs = plt.subplots(1, 2 if two else 1, figsize=(16, 7) if two else (8, 7), squeeze=False)
    ax = axs[0, 0]
    norm5 = 100 * normalise(cm_abs, "pred")
    shown = np.where(norm5 > 0, norm5, np.nan)  # empty cells stay white under the log scale
    ax.imshow(shown, cmap=cmap, norm=LogNorm(vmin=max(np.nanmin(shown), 1e-2), vmax=100))
    ticks = np.arange(5)
    labels = ["Null", "Fixed\npoint", "Periodic", "Locally\nChaotic", "Chaotic"]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_yticklabels(labels, fontsize=14)
    for (j, i), v in np.ndenumerate(norm5):
        t = ax.text(i, j, f"{np.round(v, 2)}%\n({cm_abs[j, i]})", ha="center", va="center", size=13)
        t.set_bbox(dict(facecolor="white", alpha=0.5, edgecolor="white", boxstyle="round"))
    ax.set_title("LP classification", fontsize=20)
    ax.set_xlabel("Predicted", fontsize=20)
    ax.set_ylabel("Actual", fontsize=20)
    if two:
        ax2 = axs[0, 1]
        n88 = 100 * normalise(cm_independent, "pred")
        ax2.imshow(np.where(n88 > 0, n88, np.nan), cmap=cmap, norm=LogNorm(vmin=1e-2, vmax=100))
        _blank(ax2)
        ax2.set_title("Independent rules", fontsize=20)
        ax2.set_xlabel("Predicted", fontsize=20)
        ax2.set_ylabel("Actual", fontsize=20)
        fig.suptitle("Confusion matrices (final model)", size=24)
    return fig


def accuracy_history(history: Dict[str, Sequence[float]], title: str = ""):
    """Training and validation accuracy per epoch of one run (not a chapter figure)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(1, len(history["accuracy"]) + 1)
    ax.plot(epochs, history["accuracy"], "-", color="darkgreen", label="Train accuracy (augmented batches)")
    if "val_accuracy" in history:
        ax.plot(epochs, history["val_accuracy"], "-", color="maroon", label="Validation accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, size=12)
    ax.legend()
    return fig
