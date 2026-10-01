"""The rule-table scan: chapter Sec. 2.1, "Benchmark: a simple grid search".

An ECA diagram shows, for every cell of every row except the last, one
neighbourhood ``(left, centre, right)`` and the state it produces in the next
row (a "T-tetromino"). Scanning the diagram therefore fills in the rule table,
and with it the Li-Packard class. The scan fails only when some neighbourhood
never occurs: the diagram is then *undetermined*. Its missing rule-table
entries can be guessed, and the class may or may not survive a wrong guess.

The chapter reports 370 undetermined diagrams in the training-and-validation
set and 33 in the test set, and exactly one diagram (of rule 59) whose class
could be guessed wrongly. Those numbers come from
``benchmark-classification-by-scanning``, whose periodic padding is the wrong
way round (``docs/provenance.md`` §3.2). ``mode: faithful`` repeats that
padding; ``mode: corrected`` (the default) fixes it.

Neighbourhood ``i`` is ``(left, centre, right) = format(i, "03b")``, so
neighbourhood 0 is ``000`` and 7 is ``111``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .data import DiagramSet
from .lookup import CLASS_NAMES, Lookup

NEIGHBOURHOODS = tuple(format(i, "03b") for i in range(8))

#: Diagrams unpacked at a time; bounds the memory use of a scan.
CHUNK = 8192


def periodic_pad(diagrams: np.ndarray, faithful: bool = False) -> np.ndarray:
    """Add one column on each side so that every cell has two neighbours.

    With periodic boundaries the left neighbour of the first cell is the *last*
    cell, and the right neighbour of the last cell is the *first*.
    """
    first, last = diagrams[..., :1], diagrams[..., -1:]
    if faithful:
        # docs/provenance.md §3.2: the notebook's _return_periodic_diagram pads
        # with the first column on the left and the last column on the right.
        # This invents triplets that do not occur and misses some that do.
        return np.concatenate([first, diagrams, last], axis=-1)
    return np.concatenate([last, diagrams, first], axis=-1)


def neighbourhood_indices(diagrams: np.ndarray, faithful: bool = False) -> np.ndarray:
    """Neighbourhood index of every cell in every row except the last.

    The last row is skipped because its neighbourhoods have no observed
    successor, so they reveal nothing about the rule table (as in the notebook,
    which scans ``range(timesteps - 1)``).

    Returns an array of shape ``(..., timesteps - 1, width)`` with values 0-7.
    """
    p = periodic_pad(diagrams[..., :-1, :], faithful).astype(np.uint8)
    return (p[..., :-2] << 2) | (p[..., 1:-1] << 1) | p[..., 2:]


def observed_neighbourhoods(diagrams: np.ndarray, faithful: bool = False) -> np.ndarray:
    """Boolean ``(n, 8)``: which neighbourhoods occur in each diagram (``(n, T, W)``)."""
    idx = neighbourhood_indices(diagrams, faithful).reshape(len(diagrams), -1).astype(np.int64)
    idx += 8 * np.arange(len(diagrams))[:, None]
    return (np.bincount(idx.ravel(), minlength=8 * len(diagrams)) > 0).reshape(-1, 8)


def read_rule_tables(diagrams: np.ndarray) -> np.ndarray:
    """Read the rule table off each diagram (correct periodic padding).

    Returns an int8 array ``(n, 8)`` whose entry ``k`` is the state that
    neighbourhood ``k`` produces in the diagram: 0 or 1, -1 if the
    neighbourhood never occurs, and -2 if it produces both states (impossible
    for a diagram of an ECA).
    """
    n = len(diagrams)
    idx = neighbourhood_indices(diagrams).reshape(n, -1).astype(np.int64)
    succ = diagrams[:, 1:, :].reshape(n, -1).astype(np.int64)
    code = 2 * idx + succ + 16 * np.arange(n)[:, None]
    counts = np.bincount(code.ravel(), minlength=16 * n).reshape(n, 8, 2)
    zeros, ones = counts[..., 0] > 0, counts[..., 1] > 0
    table = np.where(ones, 1, np.where(zeros, 0, -1)).astype(np.int8)
    table[zeros & ones] = -2
    return table


def table_to_rule(table: np.ndarray, missing_as: int = 0) -> np.ndarray:
    """Rule numbers from ``(n, 8)`` tables, with missing entries set to ``missing_as``.

    With ``missing_as=0`` this is exactly the guess of the exact CNN
    (Sec. 3.1): a neighbourhood that never occurs gives an all-zero channel,
    which the output layer reads as a 0 in the rule table.
    """
    bits = np.where(table < 0, missing_as, table).astype(np.int64)
    return (bits << np.arange(8)).sum(axis=1)


def alternative_rules(rule: int, missing: List[int], faithful: bool = False) -> List[int]:
    """The other rules consistent with a diagram that lacks ``missing`` neighbourhoods.

    Corrected mode returns every completion of the rule table except the true
    one: ``2**len(missing) - 1`` rules. Faithful mode returns the single rule
    the notebook checks, with all missing outputs flipped. The notebook's own
    comment warns that this only works when one neighbourhood is missing. The
    two modes agree in that case.
    """
    if not missing:
        return []
    if faithful:
        # docs/provenance.md §7.4: suggest_alternative_rule flips every missing bit.
        return [int(rule) ^ sum(1 << k for k in missing)]
    alts = []
    for values in itertools.product((0, 1), repeat=len(missing)):
        r = int(rule)
        for k, v in zip(missing, values):
            r = (r & ~(1 << k)) | (v << k)
        if r != rule:
            alts.append(r)
    return alts


@dataclass
class ScanResult:
    """Outcome of scanning one set of diagrams."""

    n_diagrams: int
    faithful: bool
    #: Indices (into the set) of the undetermined diagrams.
    undetermined: np.ndarray
    #: Their rules.
    rules: np.ndarray
    #: Their missing neighbourhoods, boolean ``(n_undetermined, 8)``.
    missing: np.ndarray
    #: Undetermined diagrams whose class could be guessed wrongly.
    class_ambiguous: List[Dict] = field(default_factory=list)

    @property
    def n_undetermined(self) -> int:
        return int(len(self.undetermined))

    def per_rule(self) -> np.ndarray:
        """Number of undetermined diagrams for each of the 256 rules."""
        return np.bincount(self.rules.astype(np.int64), minlength=256)

    def per_class(self, lookup: Lookup) -> np.ndarray:
        """Number of undetermined diagrams in each Li-Packard class."""
        return np.bincount(lookup.rule_to_class[self.rules.astype(np.int64)], minlength=5)

    def summary(self, lookup: Lookup) -> Dict:
        n_missing = self.missing.sum(axis=1)
        return {
            "n_diagrams": self.n_diagrams,
            "n_undetermined": self.n_undetermined,
            "fraction_undetermined": round(self.n_undetermined / self.n_diagrams, 6),
            "max_missing_neighbourhoods": int(n_missing.max()) if len(n_missing) else 0,
            "n_missing_histogram": {int(k): int(v) for k, v in zip(*np.unique(n_missing, return_counts=True))},
            "undetermined_per_class": dict(zip(CLASS_NAMES, map(int, self.per_class(lookup)))),
            "missing_neighbourhood_counts": dict(zip(NEIGHBOURHOODS, map(int, self.missing.sum(axis=0)))),
            "n_class_ambiguous": len(self.class_ambiguous),
            "class_ambiguous": self.class_ambiguous,
        }


def scan_set(ds: DiagramSet, lookup: Lookup, faithful: bool = False) -> ScanResult:
    """Scan every diagram of a set for missing neighbourhoods.

    A diagram is *class-ambiguous* when at least one alternative rule
    consistent with it (see :func:`alternative_rules`) lies in a different
    Li-Packard class than the true rule.
    """
    und, rules, missing = [], [], []
    for start in range(0, len(ds), CHUNK):
        stop = min(start + CHUNK, len(ds))
        obs = observed_neighbourhoods(ds.diagrams(slice(start, stop)), faithful)
        bad = np.flatnonzero(~obs.all(axis=1))
        und.append(start + bad)
        rules.append(ds.rules[start + bad])
        missing.append(~obs[bad])
    und = np.concatenate(und)
    rules = np.concatenate(rules)
    missing = np.concatenate(missing) if missing else np.zeros((0, 8), bool)

    ambiguous = []
    for i, rule, miss in zip(und, rules, missing):
        true_class = int(lookup.rule_to_class[rule])
        alts = alternative_rules(int(rule), list(np.flatnonzero(miss)), faithful)
        wrong = [a for a in alts if lookup.rule_to_class[a] != true_class]
        if wrong:
            ambiguous.append({
                "index": int(i), "rule": int(rule), "class": CLASS_NAMES[true_class],
                "missing": [NEIGHBOURHOODS[k] for k in np.flatnonzero(miss)],
                "alternatives_in_other_class": [
                    {"rule": int(a), "class": CLASS_NAMES[int(lookup.rule_to_class[a])]} for a in wrong],
                "probability_class_correct": round(1 - len(wrong) / (len(alts) + 1), 6),
            })
    return ScanResult(len(ds), faithful, und, rules, missing, ambiguous)


def read_set(ds: DiagramSet) -> np.ndarray:
    """:func:`read_rule_tables` for a whole set, in chunks; ``(n, 8)`` int8."""
    out = np.empty((len(ds), 8), dtype=np.int8)
    for start in range(0, len(ds), CHUNK):
        stop = min(start + CHUNK, len(ds))
        out[start:stop] = read_rule_tables(ds.diagrams(slice(start, stop)))
    return out
