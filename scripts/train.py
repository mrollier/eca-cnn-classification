"""Train the network of an experiment config.

    python scripts/train.py --config configs/default_rules.yaml
    python scripts/train.py --config configs/final_classes.yaml --set mode=faithful
    python scripts/train.py --config configs/augmentation_table.yaml --list
    python scripts/train.py --config configs/augmentation_table.yaml --cell 3
    python scripts/train.py --config configs/trimmed_classes.yaml --smoke

``--smoke`` trains for one epoch on a few diagrams per rule. It checks that a
config runs end to end and estimates the full epoch time on this machine.

A run writes into ``output/runs/<name>/`` (see ``src/eca_cnn/train.py``). An
existing finished run (with a ``metrics.json``) is skipped unless ``--force``
is given, so an interrupted sweep can simply be restarted.
"""

from __future__ import annotations

import argparse
import time

from eca_cnn.cli import add_config_args, config_from_args, quiet_tensorflow
from eca_cnn.config import expand_sweep

quiet_tensorflow()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser, default=None)
    parser.add_argument("--cell", type=int, action="append",
                        help="run only this cell of a sweep (0-based; repeatable)")
    parser.add_argument("--list", action="store_true", help="list the runs this config defines and exit")
    parser.add_argument("--smoke", action="store_true", help="one epoch on a few diagrams per rule")
    parser.add_argument("--force", action="store_true", help="re-run finished runs")
    parser.add_argument("--verbose", type=int, default=2, choices=[0, 1, 2], help="Keras fit verbosity")
    args = parser.parse_args(argv)
    if args.config is None:
        parser.error("--config is required")

    cells = expand_sweep(config_from_args(args))
    if args.list:
        for i, c in enumerate(cells):
            print(f"{i:3d}  {c.run_name}")
        return 0
    if args.cell:
        bad = [i for i in args.cell if not 0 <= i < len(cells)]
        if bad:
            parser.error(f"--cell {bad}: this config has {len(cells)} cell(s) (see --list)")
        cells = [cells[i] for i in args.cell]

    from eca_cnn.train import train

    for i, cfg in enumerate(cells):
        done = cfg.run_dir().with_name(cfg.run_dir().name + ("__smoke" if args.smoke else "")) / "metrics.json"
        if done.exists() and not args.force:
            print(f"[{i + 1}/{len(cells)}] {cfg.run_name}: finished run exists, skipping (--force to redo)")
            continue
        print(f"[{i + 1}/{len(cells)}] {cfg.run_name}" + (" (smoke)" if args.smoke else ""))
        t0 = time.perf_counter()
        m = train(cfg, smoke=args.smoke, verbose=args.verbose)
        accs = ", ".join(f"{k} {v['accuracy']:.4%}" for k, v in m["results"].items())
        print(f"    {m['n_parameters']} parameters, {m['epochs_run']} epoch(s), {accs} "
              f"({time.perf_counter() - t0:.0f} s)")
        if "estimated_full_epoch_seconds" in m:
            print(f"    estimated full epoch on this machine: {m['estimated_full_epoch_seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
