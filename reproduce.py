#!/usr/bin/env python
"""Single entry point to reproduce the chapter's numbers and figures.

Usage
-----
    python reproduce.py quick    # data, benchmark, exact CNN, parameter counts, fast tests (~3 min; CI)
    python reproduce.py all      # quick + slow tests + figures + the walkthrough notebook (~6 min)
    python reproduce.py train    # every training run of the chapter (about a day on a laptop CPU;
                                 # see docs/runbook.md), then the results tables and figures
    python reproduce.py train --only default_rules final_classes

``quick`` regenerates every committed file in ``data/results/`` that does not
need training, so a ``git diff`` afterwards shows whether the code still
produces the committed numbers (CI does exactly that). ``all`` adds the
figures that need no training, plus those whose training runs exist. ``train``
skips runs that have already finished, so it can be restarted after an
interruption.

Every step runs as a subprocess, so one failure is reported without hiding
the rest. The exit code is non-zero if any step fails.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable

QUICK = [
    ("datasets (Sec. 1.3)", [PY, "scripts/make_dataset.py"]),
    ("parameter counts (Sec. 2.2, 3.1-3.3)", [PY, "scripts/count_parameters.py"]),
    ("benchmark scan, both modes (Sec. 2.1)", [PY, "scripts/run_benchmark.py"]),
    ("exact CNN (Sec. 3.1)", [PY, "scripts/run_exact_cnn.py"]),
    ("verification suite, fast part", [PY, "-m", "pytest", "-m", "not slow"]),
]
ALL = QUICK[:-1] + [
    ("verification suite, all", [PY, "-m", "pytest"]),
    ("figures (corrected mode)", [PY, "scripts/make_figures.py"]),
    ("figures (faithful mode)", [PY, "scripts/make_figures.py", "--set", "mode=faithful"]),
    ("walkthrough notebook", [PY, "notebooks/execute.py"]),
]

#: Every training run behind a number or figure of the chapter, cheapest first.
#: Faithful runs are only needed where the coarse-graining bug changes the
#: result (the coarse-grained and "all" cells of the table, the final models);
#: every other run is the same in both modes and is shared (Config.run_name).
TRAINING = [
    ("default_rules", ["--config", "configs/default_rules.yaml"]),
    ("default_classes", ["--config", "configs/default_classes.yaml"]),
    ("trimmed_classes", ["--config", "configs/trimmed_classes.yaml"]),
    ("trimmed_rules", ["--config", "configs/trimmed_rules.yaml"]),
    ("mlp_single_layer_classes", ["--config", "configs/mlp_single_layer_classes.yaml"]),
    ("mlp_cropped_classes", ["--config", "configs/mlp_cropped_classes.yaml"]),
    ("augmentation_table", ["--config", "configs/augmentation_table.yaml"]),
    ("augmentation_table__faithful", ["--config", "configs/augmentation_table.yaml", "--set", "mode=faithful",
                                      *[a for c in (9, 10, 11, 15, 16, 17) for a in ("--cell", str(c))]]),
    ("final_classes", ["--config", "configs/final_classes.yaml"]),
    ("final_independent", ["--config", "configs/final_independent.yaml"]),
    ("final_classes__faithful", ["--config", "configs/final_classes.yaml", "--set", "mode=faithful"]),
    ("final_independent__faithful", ["--config", "configs/final_independent.yaml", "--set", "mode=faithful"]),
]
AFTER_TRAINING = [
    ("results tables (corrected)", [PY, "scripts/collect_results.py"]),
    ("results tables (faithful)", [PY, "scripts/collect_results.py", "--set", "mode=faithful"]),
    ("figures (corrected mode)", [PY, "scripts/make_figures.py"]),
    ("figures (faithful mode)", [PY, "scripts/make_figures.py", "--set", "mode=faithful"]),
]


def run(label: str, argv) -> tuple:
    print(f"\n=== {label} ===", flush=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "TF_CPP_MIN_LOG_LEVEL": "2"}
    t0 = time.perf_counter()
    ok = subprocess.run(argv, cwd=ROOT, env=env).returncode == 0
    return label, ok, time.perf_counter() - t0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["quick", "all", "train"])
    p.add_argument("--only", nargs="+", choices=[t[0] for t in TRAINING], help="train mode: only these runs")
    args = p.parse_args(argv)

    if args.mode == "quick":
        steps = QUICK
    elif args.mode == "all":
        steps = ALL
    else:
        chosen = [t for t in TRAINING if not args.only or t[0] in args.only]
        steps = [("datasets", [PY, "scripts/make_dataset.py"])]
        steps += [(f"train {name}", [PY, "scripts/train.py", *a]) for name, a in chosen]
        steps += AFTER_TRAINING

    results = [run(label, a) for label, a in steps]
    print("\n" + "=" * 72)
    for label, ok, sec in results:
        print(f"{'OK  ' if ok else 'FAIL'}  {sec:8.1f} s  {label}")
    failed = [r for r in results if not r[1]]
    print("=" * 72 + f"\n{len(results) - len(failed)}/{len(results)} steps succeeded")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
