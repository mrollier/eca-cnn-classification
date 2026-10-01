"""The exact CNN of chapter Sec. 3.1, applied to every diagram.

    python scripts/run_exact_cnn.py

The chapter reports that the exact CNN (2360 hand-set parameters) misidentifies
331 of the 262,144 training-and-validation diagrams (99.87%). It also states
that these errors are exactly the undetermined diagrams whose missing
rule-table entry is a 1. Neither number is in a saved notebook output, and the
notebook's comments give 175 + 9 + 12 + 188 = 384 errors instead
(``docs/provenance.md`` §2, §3.2).

This script runs the network on both sets and checks, diagram by diagram,
that its answer equals the rule table read off by the corrected scan with
missing entries set to 0. It also counts how many of its errors fall among
the diagrams that the notebook's (faithful) scan flags, the comparison the
chapter makes. The numbers come with a breakdown by missing bit,
like the notebook's comments.

Writes ``data/results/exact_cnn.json`` (committed).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from eca_cnn.cli import RESULTS_DIR, add_config_args, config_from_args, quiet_tensorflow, write_json

quiet_tensorflow()

from eca_cnn.batches import DiagramSequence  # noqa: E402
from eca_cnn.benchmark import read_set, scan_set, table_to_rule  # noqa: E402
from eca_cnn.data import load_datasets  # noqa: E402
from eca_cnn.lookup import CLASS_NAMES, load_lookup  # noqa: E402
from eca_cnn.models import count_parameters, exact_cnn  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    lookup = load_lookup(cfg.path("lookup_dir"))
    sets = load_datasets(cfg)
    model = exact_cnn((cfg.data.timesteps, cfg.data.width, 1))

    out = {"n_parameters": count_parameters(model), "data_source": cfg.data.source,
           "chapter": {"trainval": {"n_misidentified": 331, "accuracy": 0.9987},
                       "notebook_comments": {"trainval": {"bit_0": 175, "bit_2": 9, "bit_5": 12, "bit_7": 188,
                                                          "total": 384}}}}
    for which, ds in sets.items():
        t0 = time.perf_counter()
        seq = DiagramSequence(ds, np.arange(len(ds)), None, 256, batch_size=1024)
        pred = model.predict(seq, verbose=0).argmax(axis=1)
        tables = read_set(ds)
        scan_guess = table_to_rule(tables, missing_as=0)
        undetermined = (tables == -1).any(axis=1)
        wrong = pred != ds.rules
        # The chapter compares the network's errors with the undetermined
        # diagrams of the notebook's scan, which pads the wrong way round (§3.2).
        flagged = np.zeros(len(ds), dtype=bool)
        flagged[scan_set(ds, lookup, faithful=True).undetermined] = True
        diff = np.bitwise_xor(pred, ds.rules.astype(np.int64))[wrong]
        by_bit = {f"bit_{k}": int(np.sum(diff == (1 << k))) for k in range(8)}
        out[which] = {
            "n_diagrams": len(ds),
            "n_misidentified": int(wrong.sum()),
            "accuracy": round(1 - wrong.mean(), 6),
            "equals_scan_with_missing_as_zero": bool(np.array_equal(pred, scan_guess)),
            "errors_only_on_undetermined": bool(not (wrong & ~undetermined).any()),
            "n_undetermined": int(undetermined.sum()),
            "n_undetermined_guessed_correctly": int((undetermined & ~wrong).sum()),
            "n_undetermined_faithful_scan": int(flagged.sum()),
            "n_misidentified_among_faithful_undetermined": int((wrong & flagged).sum()),
            "misidentified_by_missing_bit": {k: v for k, v in by_bit.items() if v},
            "misidentified_with_several_wrong_bits": int(len(diff) - sum(by_bit.values())),
            "misidentified_per_class": dict(zip(CLASS_NAMES, map(int, np.bincount(
                lookup.rule_to_class[ds.rules[wrong]], minlength=5)))),
        }
        print(f"{which}: {out[which]['n_misidentified']} misidentified of {len(ds)} "
              f"(accuracy {out[which]['accuracy']:.4%}); equals scan: {out[which]['equals_scan_with_missing_as_zero']} "
              f"({time.perf_counter() - t0:.0f} s)")
    write_json(out, RESULTS_DIR / ("exact_cnn.json" if cfg.data.source == "generated"
                                   else f"exact_cnn_{cfg.data.source}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
