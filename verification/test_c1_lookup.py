"""C1: the Li-Packard and equivalence tables (chapter Tab. 2 and Sec. 3.3).

- 256 rules, each in exactly one Li-Packard class; class sizes 24 / 97 / 89 / 10 / 36.
- 88 equivalence classes covering all 256 rules, and they are exactly the
  orbits of the rules under mirroring and complementation.
- The Li-Packard class is constant on every equivalence class.
- "64 rules are their own mirror images, whilst only 16 rules are their own
  inversions" (Sec. 3.4).
"""

import numpy as np

from eca_cnn.lookup import (CLASS_NAMES, complement_rule, equivalence_orbit, mirror_rule, rule_bits,
                            bits_to_rule)


def test_class_sizes(lookup):
    assert lookup.rule_to_class.shape == (256,)
    assert list(lookup.class_sizes()) == [24, 97, 89, 10, 36]


def test_88_equivalence_classes_partition_all_rules(lookup):
    assert len(lookup.representatives) == 88
    assert sorted(r for cls in lookup.members for r in cls) == list(range(256))
    assert set(np.unique(lookup.rule_to_indep)) == set(range(88))


def test_equivalence_classes_are_symmetry_orbits(lookup):
    orbits = {equivalence_orbit(r) for r in range(256)}
    assert orbits == {frozenset(cls) for cls in lookup.members}


def test_class_constant_on_equivalence_classes(lookup):
    for cls in lookup.members:
        assert len({int(lookup.rule_to_class[r]) for r in cls}) == 1, cls


def test_self_mirror_and_self_complementary_counts():
    assert sum(mirror_rule(r) == r for r in range(256)) == 64
    assert sum(complement_rule(r) == r for r in range(256)) == 16


def test_symmetries_are_involutions():
    for r in range(256):
        assert mirror_rule(mirror_rule(r)) == r
        assert complement_rule(complement_rule(r)) == r
        assert bits_to_rule(rule_bits(r)) == r


def test_known_symmetries():
    # Rule 110 mirrors to 124 and complements to 137 (Li and Packard 1990).
    assert mirror_rule(110) == 124
    assert complement_rule(110) == 137
    assert equivalence_orbit(110) == {110, 124, 137, 193}


def test_label_index_follows_json_key_order(lookup):
    # The 88-class index is the key position in equiv_ECA_rules.json, as in
    # data-augmentation cell 7 (indep_to_num); keys start 0, 1, 2, ...
    assert list(lookup.representatives[:5]) == [0, 1, 2, 3, 4]
    assert lookup.rule_to_indep[255] == 0 and lookup.rule_to_indep[16] == 2
    assert lookup.labels([59, 58], "classes").tolist() == [CLASS_NAMES.index("periodic"),
                                                           CLASS_NAMES.index("fixed point")]
