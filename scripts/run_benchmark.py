"""The rule-table scan of chapter Sec. 2.1, in corrected and faithful mode.

    python scripts/run_benchmark.py                 # both modes
    python scripts/run_benchmark.py --modes corrected

The chapter reports 370 undetermined diagrams (training and validation) and
33 (test), and a single class-ambiguous diagram (rule 59). Those numbers come
from the notebook's scan, whose periodic padding is reversed
(``docs/provenance.md`` §3.2), on the unseeded original data. This script
reports both paddings on the data in ``data/`` (or on the originals, with
``--set data.source=original``).

Writes ``data/results/benchmark_<mode>.json`` (committed; CI checks that it is
regenerated unchanged) and ``output/benchmark/undetermined_<mode>_<set>.csv``.
"""

from __future__ import annotations

import argparse
import csv
import time

from eca_cnn.benchmark import NEIGHBOURHOODS, scan_set
from eca_cnn.cli import RESULTS_DIR, add_config_args, config_from_args, write_json
from eca_cnn.data import load_datasets
from eca_cnn.lookup import CLASS_NAMES, load_lookup

CHAPTER = {"trainval": {"n_undetermined": 370, "n_class_ambiguous": 1},
           "test": {"n_undetermined": 33, "n_class_ambiguous": 0},
           "max_missing_neighbourhoods": 1}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--modes", nargs="+", default=["corrected", "faithful"], choices=["corrected", "faithful"])
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    lookup = load_lookup(cfg.path("lookup_dir"))
    sets = load_datasets(cfg)
    suffix = "" if cfg.data.source == "generated" else f"_{cfg.data.source}"

    for mode in args.modes:
        t0 = time.perf_counter()
        out = {"mode": mode, "data_source": cfg.data.source, "chapter": CHAPTER}
        for which, ds in sets.items():
            res = scan_set(ds, lookup, faithful=(mode == "faithful"))
            out[which] = res.summary(lookup)
            csv_path = cfg.path("output_dir") / "benchmark" / f"undetermined_{mode}{suffix}_{which}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["index", "rule", "class", "missing"])
                for i, r, m in zip(res.undetermined, res.rules, res.missing):
                    w.writerow([int(i), int(r), CLASS_NAMES[lookup.rule_to_class[r]],
                                " ".join(NEIGHBOURHOODS[k] for k in range(8) if m[k])])
            print(f"[{mode}] {which}: {res.n_undetermined} undetermined of {res.n_diagrams} "
                  f"(chapter: {CHAPTER[which]['n_undetermined']}); "
                  f"{len(res.class_ambiguous)} class-ambiguous (chapter: {CHAPTER[which]['n_class_ambiguous']})")
        # No timing in the file: it is committed and must regenerate byte for byte.
        print(f"[{mode}] {time.perf_counter() - t0:.0f} s")
        write_json(out, RESULTS_DIR / f"benchmark_{mode}{suffix}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
