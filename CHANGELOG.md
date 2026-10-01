# Changelog

## 1.0.0 (2026-10-01)

First release: the ten Colab notebooks behind the chapter, rebuilt as a
tested package.

- **Data**: seeded, vectorised generation of the 64 × 64 datasets
  (bit-identical to cellpylib, about 12 s for all 294,912 diagrams),
  bit-packed storage (150 MB instead of 1.2 GB), and a loader for the
  original unseeded Colab files.
- **Faithful and corrected mode** for the two bugs that change published
  numbers: the coarse-graining augmentation that blanked every image
  (docs/provenance.md §3.1) and the reversed periodic padding in the benchmark
  scan (§3.2).
- **Experiments as configs**: default CNN, exact CNN, trimmed CNN,
  augmentation table (18 runs), final models, fully-connected baselines, with
  the hyperparameters of the notebook each comes from.
- **Results** in `data/results/` (dataset checksums, benchmark in both modes,
  exact CNN, parameter counts). CI regenerates them on Linux and checks that
  nothing changes.
- **Verification** suite C1–C9: 88 tests, including the notebooks' own code
  run next to the faithful implementations.
- **Figures**: one function per chapter figure, named after the thesis file;
  published colours by default, a colour-vision-deficiency-safe palette as an
  option.
- **Not yet run**: the 34 training runs (`python reproduce.py train`,
  docs/runbook.md). Every config has passed a one-epoch smoke test.
- docs/provenance.md extends the notebook audit with the claim-to-command map
  (§6) and new findings (§7). Among them: the chapter's "331 of 370" argument
  combines two paddings, and the confusion-matrix numbers are confirmed by the
  saved image.
