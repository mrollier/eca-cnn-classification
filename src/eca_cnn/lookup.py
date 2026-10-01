"""Li-Packard classes and equivalence classes of the 256 elementary rules.

Two tables from Li and Packard (1990), stored in ``data/lookup/`` exactly as
the ``create-dataset`` notebook wrote them (cells 5-7):

``equiv_ECA_rules.json``
    The 88 equivalence classes. Each key is the representative rule listed by
    Li and Packard (p. 286), each value the other rules of its class. Two rules
    are equivalent when one is the mirror image, the complement (colour
    inversion), or the mirrored complement of the other.
``all_Li-Packard_dict.json``
    The five Li-Packard classes, each with all of its rules (24 / 97 / 89 / 10
    / 36). ``nonequiv_Li-Packard_dict.json`` lists only the representatives.

The notebooks looked a rule's class up with ``whichClass``, a Python loop over
``np.isin`` that took minutes on the full dataset. Here both tables become
256-entry integer arrays, so a lookup is a single indexing operation:
``rule_to_class[rules]``.

Label conventions (shared with the notebooks, so saved models stay compatible):

- Li-Packard class index: 0 null, 1 fixed point, 2 periodic, 3 locally chaotic,
  4 chaotic (``class_to_num`` in the notebooks).
- Equivalence-class index: the position of the class's key in
  ``equiv_ECA_rules.json`` (``indep_to_num`` in ``data-augmentation`` cell 7).
  It is *not* the representative's rule number.
- Neighbourhood index: the neighbourhood ``(left, centre, right)`` has index
  ``4*left + 2*centre + right``, and bit ``i`` of the Wolfram rule number is the
  new state for neighbourhood ``i``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import ROOT

#: Li-Packard class names, in label order (the notebooks' ``class_to_num``).
CLASS_NAMES = ("null", "fixed point", "periodic", "locally chaotic", "chaotic")
#: The same, formatted for figure tick labels.
CLASS_TICKLABELS = ("Null", "Fixed\nPoint", "Periodic", "Locally\nChaotic", "Chaotic")
#: Number of output nodes per prediction target.
N_OUTPUTS = {"rules": 256, "independent": 88, "classes": 5}

DEFAULT_LOOKUP_DIR = ROOT / "data" / "lookup"


# -- symmetries of a rule ----------------------------------------------------

def rule_bits(rule: int) -> np.ndarray:
    """The rule table: entry ``i`` is the new state for neighbourhood ``i``."""
    return (int(rule) >> np.arange(8)) & 1


def bits_to_rule(bits) -> int:
    """Inverse of :func:`rule_bits`."""
    return int(sum(int(b) << i for i, b in enumerate(bits)))


def mirror_rule(rule: int) -> int:
    """The rule obtained by mirroring space: ``(l, c, r)`` is read as ``(r, c, l)``.

    Mirroring a spacetime diagram left-right turns a diagram of ``rule`` into a
    diagram of ``mirror_rule(rule)``; this is why the mirror augmentation changes
    the rule but not the equivalence class.
    """
    bits = rule_bits(rule)
    return bits_to_rule([bits[4 * (i & 1) + (i & 2) + (i >> 2)] for i in range(8)])


def complement_rule(rule: int) -> int:
    """The rule obtained by swapping the states 0 and 1 everywhere.

    Inverting the colours of a spacetime diagram of ``rule`` gives a diagram of
    ``complement_rule(rule)``: the new state for neighbourhood ``i`` is the
    complement of the old state for the complementary neighbourhood ``7 - i``.
    """
    bits = rule_bits(rule)
    return bits_to_rule([1 - bits[7 - i] for i in range(8)])


def equivalence_orbit(rule: int) -> frozenset:
    """All rules equivalent to ``rule`` under mirroring and complementation."""
    m, c = mirror_rule(rule), complement_rule(rule)
    return frozenset({int(rule), m, c, mirror_rule(c)})


# -- the tables --------------------------------------------------------------

@dataclass(frozen=True)
class Lookup:
    """The Li-Packard and equivalence tables as fast arrays.

    Attributes
    ----------
    rule_to_class : (256,) uint8
        Li-Packard class index of every rule.
    rule_to_indep : (256,) uint8
        Equivalence-class index (0-87) of every rule.
    representatives : (88,) int
        The representative rule of every equivalence class (the JSON keys, in
        file order).
    members : list of 88 tuples
        All rules of every equivalence class, sorted.
    """

    rule_to_class: np.ndarray
    rule_to_indep: np.ndarray
    representatives: np.ndarray
    members: tuple

    def labels(self, rules, output: str) -> np.ndarray:
        """Integer training targets for an array of rule numbers.

        ``output`` is ``rules`` (the rule itself), ``independent`` (equivalence
        class) or ``classes`` (Li-Packard class). This replaces the label files
        ``*_labels_class.npy`` and ``*_labels_indep.npy`` that the notebooks load
        but that no notebook writes (§4).
        """
        rules = np.asarray(rules).astype(np.int64)
        if output == "rules":
            return rules.astype(np.uint8)
        if output == "independent":
            return self.rule_to_indep[rules]
        if output == "classes":
            return self.rule_to_class[rules]
        raise ValueError(f"unknown output {output!r}")

    def class_sizes(self) -> np.ndarray:
        """Number of rules in each Li-Packard class: 24, 97, 89, 10, 36."""
        return np.bincount(self.rule_to_class, minlength=5)

    def label_names(self, output: str) -> List[str]:
        """Human-readable name of every output node."""
        if output == "rules":
            return [str(r) for r in range(256)]
        if output == "independent":
            return [str(int(r)) for r in self.representatives]
        return list(CLASS_NAMES)


def _read_json(path: Path) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=None)
def load_lookup(lookup_dir=None) -> Lookup:
    """Read the two JSON tables and check that each is a partition of 0-255."""
    d = Path(lookup_dir) if lookup_dir is not None else DEFAULT_LOOKUP_DIR
    lp = _read_json(d / "all_Li-Packard_dict.json")
    equiv = _read_json(d / "equiv_ECA_rules.json")

    if tuple(lp) != CLASS_NAMES:
        raise ValueError(f"unexpected Li-Packard class order {tuple(lp)}; expected {CLASS_NAMES}")

    rule_to_class = np.full(256, 255, dtype=np.uint8)
    for k, rules in enumerate(lp.values()):
        for r in rules:
            if rule_to_class[r] != 255:
                raise ValueError(f"rule {r} appears in more than one Li-Packard class")
            rule_to_class[r] = k
    if (rule_to_class == 255).any():
        raise ValueError(f"rules without a Li-Packard class: {np.flatnonzero(rule_to_class == 255)}")

    rule_to_indep = np.full(256, 255, dtype=np.uint8)
    representatives, members = [], []
    for k, (key, others) in enumerate(equiv.items()):
        cls = sorted({int(key), *map(int, others)})
        for r in cls:
            if rule_to_indep[r] != 255:
                raise ValueError(f"rule {r} appears in more than one equivalence class")
            rule_to_indep[r] = k
        representatives.append(int(key))
        members.append(tuple(cls))
    if (rule_to_indep == 255).any():
        raise ValueError(f"rules without an equivalence class: {np.flatnonzero(rule_to_indep == 255)}")

    return Lookup(rule_to_class, rule_to_indep, np.array(representatives), tuple(members))
