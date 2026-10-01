"""C3: the rule-table scan (chapter Sec. 2.1) and its padding bug (§3.2).

- Correct periodic padding: the row ``100000`` contains the triplets
  {001, 100, 000, 010}. The notebook's padding yields {110, 100, 000} instead.
- Reading the rule table off a diagram returns the generating rule whenever
  all neighbourhoods occur.
- Alternative-rule enumeration: one missing entry gives one alternative in
  both modes; several missing entries differ between the modes.
"""

import numpy as np

from eca_cnn.benchmark import (NEIGHBOURHOODS, alternative_rules, neighbourhood_indices,
                               observed_neighbourhoods, read_rule_tables, scan_set, table_to_rule)
from eca_cnn.simulate import evolve


def _triplets(row, faithful):
    d = np.array([row, row], dtype=np.uint8)  # two rows: the first is scanned
    idx = neighbourhood_indices(d, faithful)
    return {NEIGHBOURHOODS[i] for i in idx.ravel()}


def test_periodic_padding_example():
    row = [1, 0, 0, 0, 0, 0]
    assert _triplets(row, faithful=False) == {"001", "100", "000", "010"}
    assert _triplets(row, faithful=True) == {"110", "100", "000"}


def test_last_row_is_not_scanned():
    d = np.zeros((2, 6), dtype=np.uint8)
    d[1] = 1  # 111 occurs only in the last row
    assert not observed_neighbourhoods(d[None])[0, 7]


def test_read_rule_table_recovers_rule(rng):
    for rule in (30, 54, 110, 150):
        d = evolve(rng.integers(0, 2, size=(5, 64), dtype=np.uint8), rule, 64)
        tables = read_rule_tables(d)
        assert (tables >= 0).all()
        assert (table_to_rule(tables) == rule).all()


def test_missing_entries_read_as_zero():
    # Rule 204 (identity) from an all-zero row only shows neighbourhood 000.
    d = evolve(np.zeros(16, dtype=np.uint8), 204, 8)[None]
    table = read_rule_tables(d)[0]
    assert table.tolist() == [0, -1, -1, -1, -1, -1, -1, -1]
    assert table_to_rule(table[None])[0] == 0


def test_alternative_rules():
    # One missing entry: both modes flip that bit (rule 59 -> 58, neighbourhood 000).
    assert alternative_rules(59, [0]) == [58]
    assert alternative_rules(59, [0], faithful=True) == [58]
    # Two missing entries: three alternatives, of which faithful mode checks one.
    assert sorted(alternative_rules(0, [0, 1])) == [1, 2, 3]
    assert alternative_rules(0, [0, 1], faithful=True) == [3]


def test_scan_agrees_with_rule_table_reading(small_set, lookup):
    # With correct padding, "undetermined" means exactly "some entry unread".
    res = scan_set(small_set, lookup)
    unread = (read_rule_tables(small_set.diagrams()) == -1).any(axis=1)
    assert res.n_diagrams == len(small_set)
    assert res.undetermined.tolist() == np.flatnonzero(unread).tolist()


def test_scan_flags_a_constructed_undetermined_diagram(lookup):
    from eca_cnn.data import DiagramSet

    # Rule 204 (identity) repeats the row 1101 1101 ..., whose cyclic triplets
    # are only 110, 101, 011 and 111.
    row = np.array([1, 1, 0, 1] * 16, dtype=np.uint8)
    d = evolve(row, 204, 64)[None]
    ds = DiagramSet(np.packbits(d, axis=-1), np.array([204], np.uint8), 64, 64)
    res = scan_set(ds, lookup)
    assert res.n_undetermined == 1
    assert [NEIGHBOURHOODS[k] for k in np.flatnonzero(res.missing[0])] == ["000", "001", "010", "100"]
