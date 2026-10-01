"""C2: the simulator and the generator.

- The vectorised simulator is bit-identical to cellpylib (the notebooks'
  simulator) for every rule, with periodic boundaries and with ``timesteps``
  counting the initial row.
- A fixed seed gives identical datasets; different seeds give different ones.
- Bit-packed storage round-trips exactly.
"""

import numpy as np
import pytest

from eca_cnn.data import generate_set, load_set, save_set
from eca_cnn.simulate import evolve, evolve_cellpylib, rule_table


def test_rule_table_is_wolfram_code():
    # Rule 30: 00011110 -> neighbourhoods 1, 2, 3, 4 map to 1.
    assert rule_table(30).tolist() == [0, 1, 1, 1, 1, 0, 0, 0]


def test_periodic_boundary_and_timesteps():
    row = np.zeros(8, dtype=np.uint8)
    row[0] = 1
    d = evolve(row, 2, timesteps=5)  # rule 2: 001 -> 1, i.e. shift left
    assert d.shape == (5, 8)
    assert (d[0] == row).all()
    # The single 1 moves one cell to the left each step and wraps around.
    assert [int(np.flatnonzero(r)[0]) for r in d] == [0, 7, 6, 5, 4]


@pytest.mark.parametrize("rule", [0, 30, 54, 59, 90, 110, 150, 184, 255])
def test_matches_cellpylib(rule, rng):
    init = rng.integers(0, 2, size=(3, 64), dtype=np.uint8)
    assert np.array_equal(evolve(init, rule, 64), evolve_cellpylib(init, rule, 64))


@pytest.mark.slow
def test_matches_cellpylib_all_rules(rng):
    init = rng.integers(0, 2, size=(1, 64), dtype=np.uint8)
    for rule in range(256):
        assert np.array_equal(evolve(init, rule, 64), evolve_cellpylib(init, rule, 64)), rule


def test_generator_is_seeded():
    a = generate_set(2, 64, 64, seed=7)
    b = generate_set(2, 64, 64, seed=7)
    c = generate_set(2, 64, 64, seed=8)
    assert a.checksum() == b.checksum()
    assert a.checksum() != c.checksum()


def test_generator_layout(small_set):
    assert len(small_set) == 4 * 256
    assert small_set.rules.tolist() == np.repeat(np.arange(256), 4).tolist()
    d = small_set.diagrams(slice(0, 8))
    assert d.shape == (8, 64, 64) and d.dtype == np.uint8 and set(np.unique(d)) <= {0, 1}
    # Every diagram follows its own rule.
    for i in (0, 123, 517, 1023):
        diagram = small_set.diagrams([i])[0]
        assert np.array_equal(diagram, evolve(diagram[0], int(small_set.rules[i]), 64))


def test_packed_storage_roundtrip(small_set, tmp_path):
    path = tmp_path / "s.npz"
    save_set(small_set, path)
    back = load_set(path)
    assert back.checksum() == small_set.checksum()
    assert back.meta == small_set.meta
    assert np.array_equal(back.diagrams(), small_set.diagrams())
