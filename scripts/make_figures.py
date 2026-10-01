"""Make the chapter's figures (``output/figures/<thesis file name>.pdf`` and ``.png``).

    python scripts/make_figures.py                       # every figure whose inputs exist
    python scripts/make_figures.py --only confusion_matrix-final
    python scripts/make_figures.py --set mode=faithful   # benchmark with the notebook's padding,
                                                         # and the runs trained in faithful mode
    python scripts/make_figures.py --palette accessible  # colour-vision-deficiency-safe colours
    python scripts/make_figures.py --smoke               # from the --smoke training runs (pipeline test)

Six figures need only the generated data. Three need the predictions of a
training run (``default_rules``, ``default_classes``, ``final_classes``); when
that run does not exist the figure is skipped with the command that makes it.
Mode-dependent figures get a ``__faithful`` suffix in faithful mode.
See ``src/eca_cnn/figures.py`` for the list.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only; no window
import numpy as np  # noqa: E402

from eca_cnn import figures as F  # noqa: E402
from eca_cnn.benchmark import NEIGHBOURHOODS, scan_set  # noqa: E402
from eca_cnn.cli import add_config_args, config_from_args  # noqa: E402
from eca_cnn.data import load_datasets  # noqa: E402
from eca_cnn.evaluate import load_run  # noqa: E402
from eca_cnn.lookup import CLASS_NAMES, load_lookup  # noqa: E402


class Skip(Exception):
    """A figure's inputs are missing."""


class Context:
    """Lazily loaded inputs shared by the figures."""

    def __init__(self, cfg, palette: str, smoke: bool):
        self.cfg, self.palette, self.smoke = cfg, palette, smoke
        self.lookup = load_lookup(cfg.path("lookup_dir"))
        self._sets, self._scans = None, None

    @property
    def sets(self):
        if self._sets is None:
            self._sets = load_datasets(self.cfg)
        return self._sets

    @property
    def scans(self):
        if self._scans is None:
            self._scans = {w: scan_set(ds, self.lookup, self.cfg.faithful) for w, ds in self.sets.items()}
        return self._scans

    def first_of_rule(self, rule: int) -> np.ndarray:
        """The first diagram of ``rule`` (as the notebooks' ``rules_example``)."""
        tv = self.sets["trainval"]
        return tv.diagrams([rule * int(tv.meta["n_per_rule"])])[0]

    def lively_of_rule(self, rule: int) -> np.ndarray:
        """The first diagram of ``rule`` whose last row is not uniform.

        The notebooks drew their illustrations at random. For a chaotic rule a
        diagram can still freeze into a uniform block, which makes a poor
        illustration; this picks deterministically one that does not.
        """
        tv = self.sets["trainval"]
        start = rule * int(tv.meta["n_per_rule"])
        block = tv.diagrams(slice(start, start + int(tv.meta["n_per_rule"])))
        lively = np.flatnonzero(block[:, -1].min(axis=1) != block[:, -1].max(axis=1))
        return block[lively[0] if len(lively) else 0]

    def run(self, name: str):
        """Metrics and predictions of the run of ``configs/<name>.yaml`` in this mode."""
        from eca_cnn.config import load_config

        run_cfg = load_config(f"configs/{name}.yaml", {"mode": self.cfg.mode})
        run_dir = run_cfg.run_dir().with_name(run_cfg.run_name + ("__smoke" if self.smoke else ""))
        if not (run_dir / "metrics.json").exists():
            mode = " --set mode=faithful" if run_cfg.faithful and run_cfg.training_depends_on_mode else ""
            smoke = " --smoke" if self.smoke else ""
            raise Skip(f"run {run_dir.name} not found; train it with: "
                       f"python scripts/train.py --config configs/{name}.yaml{mode}{smoke}")
        return load_run(run_dir)


def fig_lp_classes(ctx):
    rules = F.CLASS_EXAMPLE_RULES
    return F.lp_classes_examples([ctx.first_of_rule(r) for r in rules], rules)


def fig_benchmark(ctx):
    return F.benchmark_undetermined(ctx.scans["trainval"], ctx.scans["test"], ctx.lookup, ctx.palette)


def fig_benchmark_examples(ctx):
    scan, tv = ctx.scans["trainval"], ctx.sets["trainval"]
    examples = []
    if scan.class_ambiguous:
        # Prefer a periodic rule, like the chapter's rule 59.
        amb = sorted(scan.class_ambiguous, key=lambda a: a["class"] != "periodic")[0]
        examples.append({"diagram": tv.diagrams([amb["index"]])[0], "rule": amb["rule"],
                         "class": amb["class"], "missing": amb["missing"]})
    chaotic = [k for k, r in enumerate(scan.rules) if CLASS_NAMES[ctx.lookup.rule_to_class[r]] == "chaotic"]
    if chaotic:
        k = chaotic[0]
        examples.append({"diagram": tv.diagrams([scan.undetermined[k]])[0], "rule": int(scan.rules[k]),
                         "class": "chaotic", "missing": [NEIGHBOURHOODS[j] for j in np.flatnonzero(scan.missing[k])]})
    if not examples:
        raise Skip("no class-ambiguous or chaotic undetermined diagram in this dataset")
    return F.benchmark_undetermined_examples(examples)


def fig_convolution(ctx):
    tv = ctx.sets["trainval"]
    i = int(np.random.default_rng(2023).integers(len(tv)))
    return F.example_convolution_and_relu(tv.diagrams([i])[0])


def fig_decomposition(ctx):
    return F.convolutional_decomposition(ctx.lively_of_rule(120), 120)


def fig_augmentation(ctx):
    return F.augmentation_examples(ctx.lively_of_rule(161), salt_and_pepper_fraction=0.05)


def fig_default_misidentified(ctx):
    pred = ctx.run("default_rules")["predictions"].get("val")
    if pred is None:
        raise Skip("run default_rules has no validation predictions")
    return F.default_model_misidentified(pred, ctx.lookup, ctx.palette)


def fig_default_all_wrong(ctx):
    pred = ctx.run("default_classes")["predictions"].get("val")
    if pred is None:
        raise Skip("run default_classes has no validation predictions")
    wrong = np.flatnonzero(pred["pred"] != pred["label"])
    diagrams = ctx.sets["trainval"].diagrams(pred["index"][wrong]) if len(wrong) else np.zeros((0, 64, 64))
    return F.default_model_all_wrong(diagrams, pred["label"][wrong], pred["pred"][wrong])


def fig_confusion(ctx):
    from eca_cnn.evaluate import confusion

    p = ctx.run("final_classes")["predictions"].get("test")
    if p is None:
        raise Skip("run final_classes has no test predictions")
    return F.confusion_matrix_final(confusion(p["label"], p["pred"], 5), palette=ctx.palette)


#: (thesis file name, builder, depends on mode)
FIGURES = [
    ("LP_classes-examples", fig_lp_classes, False),
    ("benchmark-undetermined_diagrams", fig_benchmark, True),
    ("benchmark-undetermined_diagrams-examples2", fig_benchmark_examples, True),
    ("example-convolution_and_relu", fig_convolution, False),
    ("convolutional_decomposition-example_120", fig_decomposition, False),
    ("augmentation-examples", fig_augmentation, False),
    ("default_model-misidentified_val_set", fig_default_misidentified, False),
    ("default_model-all_wrong_LP_classes", fig_default_all_wrong, False),
    ("confusion_matrix-final", fig_confusion, True),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--only", nargs="+", choices=[f[0] for f in FIGURES], help="make only these figures")
    parser.add_argument("--palette", default="published", choices=sorted(F.PALETTES))
    parser.add_argument("--smoke", action="store_true", help="use the --smoke training runs")
    parser.add_argument("--strict", action="store_true", help="fail when a figure has to be skipped")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    ctx = Context(cfg, args.palette, args.smoke)
    out_dir = cfg.path("output_dir") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    for name, build, mode_dependent in FIGURES:
        if args.only and name not in args.only:
            continue
        suffix = ("__faithful" if cfg.faithful and mode_dependent else "") + \
                 ("__accessible" if args.palette != "published" else "") + ("__smoke" if args.smoke else "")
        try:
            fig = build(ctx)
        except Skip as e:
            print(f"skipped {name}: {e}")
            skipped += 1
            continue
        for ext, kw in (("pdf", {}), ("png", {"dpi": 110})):
            fig.savefig(out_dir / f"{name}{suffix}.{ext}", bbox_inches="tight", **kw)
        F.plt.close(fig)
        print(f"wrote {Path('output/figures') / (name + suffix)}.pdf")
    return 1 if (args.strict and skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
