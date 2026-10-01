"""C9: the committed results in ``data/results/`` follow from the code.

On the full generated datasets (skipped when absent):

- the datasets have the committed checksums (the generator is deterministic
  across machines);
- the scan in both modes gives the committed undetermined counts;
- the exact CNN gives the committed error counts, it equals the scan with
  missing entries read as 0, and it fails only on undetermined diagrams.
"""

import json

import numpy as np
import pytest

from eca_cnn.benchmark import read_set, scan_set, table_to_rule
from eca_cnn.config import ROOT

RESULTS = ROOT / "data" / "results"


def _committed(name):
    with open(RESULTS / name, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.slow
def test_dataset_checksums(full_data):
    committed = _committed("dataset_checksums.json")
    for which, ds in full_data.items():
        assert ds.checksum() == committed[which]["sha256"], which


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["corrected", "faithful"])
def test_benchmark_counts(full_data, lookup, mode):
    committed = _committed(f"benchmark_{mode}.json")
    for which, ds in full_data.items():
        res = scan_set(ds, lookup, faithful=(mode == "faithful"))
        # Compare as JSON (integer keys become strings when written).
        assert json.loads(json.dumps(res.summary(lookup))) == committed[which]


@pytest.mark.slow
def test_exact_cnn_counts(full_data, lookup):
    from eca_cnn.batches import DiagramSequence
    from eca_cnn.models import exact_cnn

    committed = _committed("exact_cnn.json")
    model = exact_cnn()
    for which, ds in full_data.items():
        pred = model.predict(DiagramSequence(ds, np.arange(len(ds)), None, 256, batch_size=1024),
                             verbose=0).argmax(axis=1)
        tables = read_set(ds)
        assert np.array_equal(pred, table_to_rule(tables, missing_as=0))
        wrong = pred != ds.rules
        assert not (wrong & ~(tables == -1).any(axis=1)).any()
        assert int(wrong.sum()) == committed[which]["n_misidentified"]
        flagged = np.zeros(len(ds), dtype=bool)
        flagged[scan_set(ds, lookup, faithful=True).undetermined] = True
        assert int((wrong & flagged).sum()) == committed[which]["n_misidentified_among_faithful_undetermined"]
