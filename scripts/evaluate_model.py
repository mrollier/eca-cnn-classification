"""Evaluate a saved model: one trained here, or one saved by the original notebooks.

    python scripts/evaluate_model.py --model output/runs/final_classes/model.keras \
        --config configs/final_classes.yaml --sets test

    # The published final model (Keras 2 SavedModel folder from Colab) on the
    # original test set:
    python scripts/evaluate_model.py \
        --model "<Drive>/Models/reproducing_silverman2019_trimmed_classes_final_2x2-4-4-MP-4-4-16_bool-invert_mirror_coarse-grain" \
        --config configs/final_classes.yaml --sets test \
        --set data.source=original --set paths.original_data_dir="<Drive>/Files"

The config supplies the data, the split and the label type (``model.output``),
which must match the model's output layer. Results go to
``output/evaluations/<model name>/metrics.json``, with the per-diagram
predictions the figures use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eca_cnn.cli import add_config_args, config_from_args, environment, quiet_tensorflow, write_json

quiet_tensorflow()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, default=None)
    parser.add_argument("--model", required=True, help="a .keras file or a SavedModel folder")
    parser.add_argument("--sets", nargs="+", default=None, choices=["train", "val", "trainval", "test"],
                        help="sets to evaluate (default: the config's train.evaluate_on)")
    args = parser.parse_args(argv)
    if args.config is None:
        parser.error("--config is required")
    cfg = config_from_args(args)

    import numpy as np
    import tensorflow as tf

    from eca_cnn.evaluate import evaluate_model
    from eca_cnn.lookup import N_OUTPUTS
    from eca_cnn.models import PeriodicPadding  # noqa: F401  (registers the custom layer)
    from eca_cnn.train import prepare

    model = tf.keras.models.load_model(args.model, compile=False)
    if int(model.output_shape[-1]) != N_OUTPUTS[cfg.model.output]:
        parser.error(f"the model has {model.output_shape[-1]} outputs but model.output is "
                     f"{cfg.model.output!r} ({N_OUTPUTS[cfg.model.output]} outputs)")
    sets = args.sets or cfg.train.evaluate_on
    cfg.train.evaluate_on = list(sets)
    prep = prepare(cfg)
    tv, y = prep.sets["trainval"], prep.labels["trainval"]
    which = {}
    for name in sets:
        if name == "train":
            which[name] = (tv, prep.train_idx, y[prep.train_idx])
        elif name == "val":
            if prep.val_idx is None:
                parser.error("this config has no validation split (train_on: trainval)")
            which[name] = (tv, prep.val_idx, y[prep.val_idx])
        elif name == "trainval":
            which[name] = (tv, np.arange(len(tv)), y)
        else:
            ts = prep.sets["test"]
            which[name] = (ts, np.arange(len(ts)), prep.labels["test"])

    # A SavedModel is a folder named after the model; a .keras file sits in its run folder.
    model_path = Path(args.model.rstrip("/\\"))
    name = model_path.parent.name if model_path.is_file() else model_path.name
    out_dir = cfg.path("output_dir") / "evaluations" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = evaluate_model(model, cfg, prep.lookup, which, out_dir)
    for k, v in results.items():
        print(f"{k}: accuracy {v['accuracy']:.4%} ({v['n_wrong']} of {v['n']} wrong)")
    write_json({"model": str(args.model), "config": str(args.config), "data_source": cfg.data.source,
                "mode": cfg.mode, "results": results, "environment": environment()},
               out_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
