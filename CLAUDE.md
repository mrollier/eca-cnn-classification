# Orientation for future Claude Code sessions

This repository is the reproducibility package of the book chapter
*Convolutional Neural Networks for Automated Cellular Automaton Classification*
(Rollier, Daly, Baetens; Advances in Cellular Automata, ECC 53, pp. 49-70,
Springer 2025; arXiv:2409.02740) and of its revised version, Ch. 8 of Michiel
Rollier's PhD thesis (Ghent University, 2026). The main message: a 64 x 64 ECA
diagram nearly always contains its rule table, so a default CNN identifies the
rule rather than the pattern. A trimmed CNN with augmentation keeps the
Li-Packard class while losing most of the rule.

## Layout
- `src/eca_cnn/`: the package. Scripts and the notebook import from here and
  never duplicate logic.
- `configs/`: one YAML per experiment, inheriting `base.yaml`. Unknown keys are
  errors. A sweep (`augmentation_table.yaml`) expands into cells.
- `scripts/`: thin CLIs. `reproduce.py quick|all|train` runs them in order.
- `verification/`: pytest, claim tests `test_c<N>_*.py` (C1-C9, listed in
  `docs/provenance.md` §6). `-m "not slow"` is the fast part; the slow part
  needs the generated data. Must exit zero.
- `data/results/`: committed outputs. `reproduce.py quick` must regenerate them
  byte for byte (CI checks with `git diff`). Never put timings in these files.
- `notebooks/archive/`: the ten original Colab notebooks with outputs. Read-only
  evidence: never edit or delete.
- `input/` (ignored): the arXiv PDF and the thesis chapter .tex, the specification.

## Hazards
- **Faithful vs corrected.** `mode` switches exactly three behaviours
  (`config.py` docstring). Any new deviation from the notebooks needs its own
  config flag and a pointer to a `docs/provenance.md` section. Do not change the
  science silently.
- **Do not invent results.** The README states only numbers this code produced.
  Chapter values are always labelled as such. Training results go through
  `scripts/collect_results.py`.
- **TensorFlow 2.15 / Keras 2** is pinned (Python <= 3.11, numpy < 2). Keras 3
  would break the saved-model compatibility with the original models.
- **The split depends on the label shape.** The notebooks stratify on `(n, 1)`
  label arrays, which scikit-learn turns into strings; 1-D labels give a
  different split for rules and equivalence classes. `data.split_indices`
  reshapes on purpose (docs/provenance.md §7.1); keep it.
- **Kernel notation.** Keras `(2, 3)` is (time, space) and is the chapter's
  "3 x 2".
- **TF logging in Jupyter.** TF switches its logger to INFO in interactive
  sessions; configure logging before importing it (see the notebook's first
  cell). Do not call `matplotlib.use` in the package; only scripts do.
- **Windows.** Set `PYTHONIOENCODING=utf-8` (`reproduce.py` does). The repo lives
  under OneDrive, and its path has spaces: quote it. Do not install a user-level
  Jupyter kernelspec; `notebooks/execute.py` uses the venv's own kernel.
- **Disk.** The laptop has little free space. Generated data are bit-packed
  (150 MB); keep pip's cache off (`--no-cache-dir`).

## Environment
- venv: `%LOCALAPPDATA%\venvs\eca-cnn` (Python 3.11.5, TF 2.15.0), package
  installed editable.
- No GPU on the laptop. Smoke-run epoch estimates: default CNN ~100 s, trimmed
  ~40-60 s; all 34 training runs take about a day (`docs/runbook.md`).

## Conventions
- UK English in prose and comments (behaviour, neighbourhood, normalise).
- NumPy-style docstrings that explain *why*, assuming the reader knows CAs but
  not this code.
- Every random draw is seeded; randomness in batches depends only on
  (seed, epoch, batch index).
- Tests encode the chapter's claims or exact values. Never loosen a failing
  check; report the discrepancy with the actual value.
- Ask the owner before choosing hosting for data or models, or deleting any
  archived notebook.
