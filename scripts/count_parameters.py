"""Parameter counts of every network in the chapter.

    python scripts/count_parameters.py

Writes ``data/results/parameter_counts.json`` (committed). The chapter's
values: default 6016 (rules) / 7301 (classes), exact 2360, trimmed 389
(classes) / 1504 (rules) / 664 (88 equivalence classes), and "well over a
million" for the fully-connected network.
"""

from __future__ import annotations

from eca_cnn.cli import RESULTS_DIR, quiet_tensorflow, write_json

quiet_tensorflow()

from eca_cnn.models import count_parameters, default_cnn, exact_cnn, mlp, trimmed_cnn  # noqa: E402

CHAPTER = {"default_rules": 6016, "default_classes": 7301, "exact_rules": 2360, "trimmed_classes": 389,
           "trimmed_rules": 1504, "trimmed_independent": 664}


def main() -> int:
    counts = {
        "default_rules": count_parameters(default_cnn("rules")),
        "default_classes": count_parameters(default_cnn("classes")),
        "exact_rules": count_parameters(exact_cnn()),
        "trimmed_classes": count_parameters(trimmed_cnn("classes")),
        "trimmed_rules": count_parameters(trimmed_cnn("rules")),
        "trimmed_independent": count_parameters(trimmed_cnn("independent")),
        "mlp_single_layer_classes": count_parameters(mlp("classes", (64, 64, 1), [256])),
        "mlp_single_layer_rules": count_parameters(mlp("rules", (64, 64, 1), [256])),
        "mlp_cropped_classes": count_parameters(mlp("classes", (28, 28, 1), [1024, 512, 256])),
    }
    for k, v in counts.items():
        ref = CHAPTER.get(k)
        print(f"{k:28s} {v:>9,d}" + (f"   (chapter: {ref:,d})" if ref else ""))
    write_json({"counts": counts, "chapter": CHAPTER}, RESULTS_DIR / "parameter_counts.json")
    return 0 if all(counts[k] == v for k, v in CHAPTER.items()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
