"""Elementary cellular automata with periodic boundaries.

The original ``create-dataset`` notebook evolved every diagram with cellpylib::

    diagram = cpl.evolve(cpl.init_random(64), timesteps=64, memoize=True,
                         apply_rule=lambda n, c, t: cpl.nks_rule(n, rule), r=1)

cellpylib applies the rule cell by cell in Python, which takes hours for the
294,912 diagrams of the chapter. :func:`evolve` does the same computation for
a whole batch of initial rows at once with numpy. Two conventions are kept
from cellpylib, and checked against it for all 256 rules in
``verification/test_c2_simulation.py``:

- ``timesteps`` counts the rows of the diagram *including* the initial row, so
  a 64-row diagram contains 63 updates.
- Boundaries are periodic: the left neighbour of cell 0 is the last cell.
"""

from __future__ import annotations

import numpy as np


def rule_table(rule: int) -> np.ndarray:
    """Lookup table ``new_state = table[4*left + 2*centre + right]`` (Wolfram code)."""
    if not 0 <= int(rule) <= 255:
        raise ValueError(f"an elementary rule lies in 0..255, got {rule}")
    return ((int(rule) >> np.arange(8)) & 1).astype(np.uint8)


def step(rows: np.ndarray, table: np.ndarray) -> np.ndarray:
    """One synchronous update of every row in ``rows`` (shape ``(..., width)``)."""
    left = np.roll(rows, 1, axis=-1)    # left[i] = rows[i-1], wrapping around
    right = np.roll(rows, -1, axis=-1)  # right[i] = rows[i+1], wrapping around
    return table[(left << 2) | (rows << 1) | right]


def evolve(initial: np.ndarray, rule: int, timesteps: int) -> np.ndarray:
    """Spacetime diagrams of ``rule`` from a batch of initial rows.

    Parameters
    ----------
    initial : (n, width) or (width,) array of 0/1
        Initial configurations; becomes row 0 of each diagram.
    rule : int
        Wolfram rule number.
    timesteps : int
        Rows per diagram, including the initial row (cellpylib convention).

    Returns
    -------
    (n, timesteps, width) uint8 array, or (timesteps, width) for a single row.
    """
    initial = np.asarray(initial, dtype=np.uint8)
    single = initial.ndim == 1
    rows = initial[None] if single else initial
    table = rule_table(rule)
    out = np.empty((rows.shape[0], timesteps, rows.shape[1]), dtype=np.uint8)
    out[:, 0] = rows
    for t in range(1, timesteps):
        out[:, t] = step(out[:, t - 1], table)
    return out[0] if single else out


def evolve_cellpylib(initial: np.ndarray, rule: int, timesteps: int) -> np.ndarray:
    """The notebook's simulator, kept as the reference for :func:`evolve`.

    Same signature and output as :func:`evolve`, but one diagram at a time
    through ``cellpylib.evolve`` with ``memoize=True``, exactly as in
    ``create-dataset`` cell 10.
    """
    import cellpylib as cpl

    initial = np.asarray(initial, dtype=np.uint8)
    single = initial.ndim == 1
    rows = initial[None] if single else initial
    out = np.empty((rows.shape[0], timesteps, rows.shape[1]), dtype=np.uint8)
    for i, row in enumerate(rows):
        out[i] = cpl.evolve(np.array([row], dtype=np.int32), timesteps=timesteps, memoize=True,
                            apply_rule=lambda n, c, t: cpl.nks_rule(n, rule), r=1)
    return out[0] if single else out
