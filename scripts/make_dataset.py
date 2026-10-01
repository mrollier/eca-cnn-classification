"""Generate the training-and-validation set and the test set (chapter Sec. 1.3).

    python scripts/make_dataset.py              # both sets, as in configs/base.yaml
    python scripts/make_dataset.py --force      # regenerate even if the files exist

Writes ``data/generated/{trainval,test}.npz`` (bit-packed, about 150 MB in
total) and ``data/results/dataset_checksums.json``. The checksums are committed:
the same seed must give the same data on any machine, and CI checks that it
does. Takes about 12 s with the numpy simulator.
"""

from __future__ import annotations

import argparse
import time

from eca_cnn.cli import RESULTS_DIR, add_config_args, config_from_args, write_json
from eca_cnn.data import dataset_path, generate_from_config, load_set, save_set


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(parser)
    parser.add_argument("--force", action="store_true", help="regenerate existing files")
    args = parser.parse_args(argv)
    cfg = config_from_args(args)
    if cfg.data.source != "generated":
        parser.error("data.source is not 'generated'; nothing to generate")

    checksums = {}
    for which in ("trainval", "test"):
        path = dataset_path(cfg, which)
        n_per_rule = cfg.data.n_per_rule_trainval if which == "trainval" else cfg.data.n_per_rule_test
        seed = cfg.data.seed_trainval if which == "trainval" else cfg.data.seed_test
        ds = load_set(path) if path.exists() and not args.force else None
        if ds is not None and (ds.meta.get("seed"), ds.meta.get("n_per_rule")) != (seed, n_per_rule):
            print(f"{path.name} was made with other settings; regenerating")
            ds = None
        if ds is None:
            t0 = time.perf_counter()
            ds = generate_from_config(cfg.data, which)
            save_set(ds, path)
            print(f"{which}: {len(ds)} diagrams generated in {time.perf_counter() - t0:.1f} s -> {path}")
        else:
            print(f"{which}: {path} exists ({len(ds)} diagrams); use --force to regenerate")
        checksums[which] = {"n_diagrams": len(ds), **{k: ds.meta[k] for k in sorted(ds.meta)},
                            "sha256": ds.checksum()}
    write_json({"simulator_note": "numpy and cellpylib give identical diagrams", **checksums},
               RESULTS_DIR / "dataset_checksums.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
