# Convolutional neural networks for classifying elementary cellular automata

A 64 × 64 spacetime diagram of an elementary cellular automaton (ECA) almost
always contains its complete rule table. A network that sees the diagram can
therefore classify it by reading off the rule rather than by recognising
patterns, and a default CNN does exactly that. The chapter this repository
accompanies shows this. It then trims the network and augments the data, so
that the Li-Packard class remains easy to recognise while the rule becomes
hard to recover.

This repository accompanies

- M. Rollier, A. J. Daly and J. M. Baetens, *Convolutional Neural Networks for
  Automated Cellular Automaton Classification*, in A. Adamatzky,
  G. Ch. Sirakoulis and G. J. Martínez (eds), *Advances in Cellular Automata*,
  Emergence, Complexity and Computation 53, pp. 49–70, Springer, 2025.
  [doi:10.1007/978-3-031-81097-8_3](https://doi.org/10.1007/978-3-031-81097-8_3),
  [arXiv:2409.02740](https://arxiv.org/abs/2409.02740);
- Ch. 8 of M. Rollier, *The rich landscape of iterated simplicity: cellular
  automata and network automata*, PhD thesis, Ghent University, 2026
  ("Automated classification of ECAs using CNNs"), a revised version of the
  chapter.

It replaces the original Google Colab notebooks (kept unchanged in
[`notebooks/archive/`](notebooks/archive/)) with a tested package. The data
are regenerated from a seed, and every experiment is a config file. Two bugs
in the notebooks change published numbers. Both can be switched on
(`faithful` mode) or off (`corrected` mode, the default); see
[Faithful and corrected mode](#faithful-and-corrected-mode).

## Coming from the chapter or the thesis? Start here

| Where the text points | What to run or open |
|---|---|
| Sec. 1.3, data: 1024 + 128 diagrams per rule | `python scripts/make_dataset.py` (~12 s) |
| Tab. 2, Li-Packard class sizes; the 88 equivalence classes | [`data/lookup/`](data/lookup/), [`verification/test_c1_lookup.py`](verification/test_c1_lookup.py) |
| Sec. 2.1, benchmark scan (370 / 33 undetermined diagrams) | `python scripts/run_benchmark.py` → [`data/results/benchmark_*.json`](data/results/) |
| Sec. 2.2–2.3, default CNN (6016 / 7301 parameters) | [`configs/default_rules.yaml`](configs/default_rules.yaml), [`configs/default_classes.yaml`](configs/default_classes.yaml) |
| Sec. 3.1, exact CNN (2360 parameters, 331 errors) | `python scripts/run_exact_cnn.py` → [`data/results/exact_cnn.json`](data/results/exact_cnn.json) |
| Sec. 3.2, trimmed CNN (389 / 1504 parameters) | [`configs/trimmed_classes.yaml`](configs/trimmed_classes.yaml), [`configs/trimmed_rules.yaml`](configs/trimmed_rules.yaml) |
| Sec. 3.2, fully-connected comparison | [`configs/mlp_cropped_classes.yaml`](configs/mlp_cropped_classes.yaml), [`configs/mlp_single_layer_classes.yaml`](configs/mlp_single_layer_classes.yaml) |
| Sec. 3.3–3.4, augmentation table (18 values) | [`configs/augmentation_table.yaml`](configs/augmentation_table.yaml) |
| Sec. 3.4, final model (98.17% / 61.25%) and confusion matrix | [`configs/final_classes.yaml`](configs/final_classes.yaml), [`configs/final_independent.yaml`](configs/final_independent.yaml) |
| Every figure, by thesis file name | `python scripts/make_figures.py` → `output/figures/` |
| Which notebook cell produced which number | [`docs/provenance.md`](docs/provenance.md) |
| A guided tour | [`notebooks/reproduce_chapter.ipynb`](notebooks/reproduce_chapter.ipynb) |

## What the chapter shows, in brief

1. **The rule table is in the diagram.** Scanning a 64 × 64 diagram for the
   eight neighbourhoods and their successors fills in the rule table, and with
   it the Li-Packard class, for all but about 0.2% of the diagrams (601 of
   262,144 here).
2. **A CNN can do the same scan.** One 8-channel convolution with hand-set
   weights detects each neighbourhood together with a resulting 1. A dense layer
   turns the eight maxima into the rule number. No training is needed.
3. **A default CNN learns to do this.** Two 16-channel convolutions with
   T-tetromino kernels (after Silverman 2019) identify the rule in 99.85% of
   the validation diagrams, according to the chapter.
4. **Trimming and augmentation shift the balance.** 2 × 2 kernels, fewer
   channels and a narrow dense layer, together with inversion, mirroring and
   coarse-graining of the training diagrams, push rule identification down while
   class identification stays high.

## Install

Python 3.9–3.11 (TensorFlow 2.15 does not support later versions).

```bash
pip install -r requirements.txt     # exact versions: TensorFlow 2.15.0, numpy 1.26.4, cellpylib 2.4.0, ...
pip install -e . --no-deps
```

TensorFlow 2.15 is the version the chapter's models were trained with. It is
the last version whose `tf.keras` is Keras 2, the format of the original saved
models (`scripts/evaluate_model.py` reads them; they are not in this
repository, so this is untested here). A GPU is optional.

## Reproduce everything

```bash
python reproduce.py quick   # data, parameter counts, benchmark, exact CNN, fast tests   (~3 min)
python reproduce.py all     # + slow tests, figures, walkthrough notebook                (~6 min)
python reproduce.py train   # + every training run of the chapter                        (about a day on a laptop CPU)
```

`quick` rewrites the committed results in [`data/results/`](data/results/).
CI runs it on Linux and fails if a single byte of those files changes, which
checks that the seeded data and every number derived from them are identical
across machines.
Training is described in [`docs/runbook.md`](docs/runbook.md), including how
to evaluate the original Colab models and data.

## Results produced by this repository

All values below come from running the code on the regenerated datasets
(seeds 20230001 and 20230002). They are a fresh random sample, so the
chapter's exact counts are not expected; what should agree is their size.

**Parameter counts** (`scripts/count_parameters.py`): all six of the
chapter's values are reproduced. Default CNN 6016 (rules) / 7301 (classes),
exact CNN 2360, trimmed CNN 389 (classes) / 1504 (rules) / 664 (88
equivalence classes).

**Benchmark scan** (Sec. 2.1, `scripts/run_benchmark.py`):

| Undetermined diagrams | train + val (262,144) | test (32,768) | class could be guessed wrong |
|---|---|---|---|
| Chapter (notebook padding, original data) | 370 | 33 | 1 |
| Notebook padding (`faithful`), regenerated data | 345 | 43 | 4 + 1 |
| Correct padding (`corrected`), regenerated data | 601 | 83 | 5 + 2 |

The notebook's padding reproduces the chapter's counts to within sampling
noise. With correct periodic padding, about 1.7 times as many diagrams lack
at least one rule-table entry.

**Exact CNN** (Sec. 3.1, `scripts/run_exact_cnn.py`): 299 of 262,144
training-and-validation diagrams misidentified (accuracy 99.886%; chapter:
331, 99.87%), and 40 of 32,768 test diagrams (99.878%). For every diagram,
the network's answer equals the rule table read off by the corrected scan
with missing entries set to 0.

**Training results**: not yet run. Every training config has passed a
one-epoch smoke test (27 smoke runs, including all 18 cells of the
augmentation table and the faithful final model). `python reproduce.py train`
produces the 34 runs, and
`scripts/collect_results.py` writes them next to the chapter's values in
`data/results/training_results.md`.

## Faithful and corrected mode

Every experiment runs in `corrected` mode unless given `--set mode=faithful`.
The mode switches exactly these behaviours (see
[`src/eca_cnn/config.py`](src/eca_cnn/config.py) and
[`docs/provenance.md`](docs/provenance.md) §3):

- **Coarse-graining** (§3.1, high impact). The notebook divided each 2 × 2 block
  mean once more by 64², so every "coarse-grained" training diagram was almost
  black (maximum 1/4096) and still carried its label. The coarse-grained and
  "all augmentations" rows of the augmentation table, and the final model, were
  trained this way.
- **Periodic padding in the scan** (§3.2). The notebook padded with the first
  column on the left and the last on the right. This invents neighbourhoods
  that do not occur and misses some that do.
- **Alternative rules in the scan** (§7.4). With several missing entries, the
  notebook checks one alternative rule instead of all of them. This cannot
  affect the chapter's data if, as reported, no diagram there misses more than
  one entry.

`verification/test_c6_notebook_fidelity.py` runs the notebooks' own code
next to the faithful implementations and checks that they agree.

## Differences from the chapter

Details and evidence in [`docs/provenance.md`](docs/provenance.md) §7.

- **The 331 / 370 argument mixes two paddings.** The chapter says the exact CNN
  can only fail on the 370 undetermined diagrams, and fails on 331 of them. But
  the 370 come from the scan with reversed padding, the 331 from the network,
  which pads correctly. With one padding, the network fails on about half of
  the undetermined diagrams (299 of 601 here). Only 163 of its 299 errors are
  among the 345 diagrams the reversed-padding scan flags
  (`data/results/exact_cnn.json`). The notebook's own
  comments give 384 errors, not 331; the original data can settle which is
  right (runbook §4).
- **"Not a single diagram contains fewer than seven entries" and "all but one
  diagram" are sample properties.** On the regenerated data, diagrams missing
  two and three entries occur, and 5–7 diagrams are class-ambiguous, in either
  mode.
- **Numbers not traceable to a saved output.** These are the exact CNN's 331,
  the trimmed CNN's 71.19%, and all 18 values of the augmentation table
  (docs/provenance.md §2). The configs reproduce the settings the notebooks
  record. The chapter's table reports the best of several runs whose settings
  were not kept. Only "likely" are the default CNN's 99.99% after 12 epochs
  with 6 errors, and the final model's 61.25% for the 88 equivalence classes:
  the notebooks computed them but did not print them.
- **99.86% versus 99.85%.** The chapter gives the default CNN 99.86% training
  accuracy for rules. The notebook printed 0.998530, which rounds to 99.85%
  (§7.9).
- **The fully-connected comparison** describes `MLP-verification`'s
  architecture with `MLP-verification_cropped`'s accuracy (§7.2). Both are
  configs here.
- **Confusion matrix confirmed.** The saved image of the notebook's confusion
  matrix confirms the final model's 98.17% test accuracy and every per-class
  figure the chapter quotes (§7.7).

## Repository layout

```
configs/                one YAML file per experiment; base.yaml holds the shared settings
src/eca_cnn/
  config.py             experiment schema, faithful/corrected mode
  lookup.py             Li-Packard and equivalence tables as 256-entry arrays; rule symmetries
  simulate.py           vectorised ECA evolution (bit-identical to cellpylib)
  data.py               seeded generation, bit-packed storage, labels, the notebooks' split
  batches.py            Keras batch generator (unpacks on the fly, applies augmentation)
  benchmark.py          rule-table scan, both paddings; rule tables read off diagrams
  models.py             default, exact, trimmed and fully-connected networks
  augment.py            inversion, mirroring, coarse-graining, salt-and-pepper; one of n
  train.py, evaluate.py training with the chapter's hyperparameters; accuracies, confusion matrices
  figures.py            one function per chapter figure
scripts/                make_dataset, run_benchmark, run_exact_cnn, count_parameters, train,
                        evaluate_model, collect_results, make_figures
verification/           pytest: claims C1-C9 (docs/provenance.md §6); -m "not slow" for the fast part
data/lookup/            the three Li-Packard JSON tables (as written by the original notebook)
data/results/           committed results: dataset checksums, benchmark, exact CNN, parameter counts
data/generated/         the datasets (ignored by git; about 150 MB, ~12 s to generate)
notebooks/archive/      the ten original Colab notebooks, with their outputs, untouched
notebooks/              reproduce_chapter.ipynb: a walkthrough calling the package
docs/                   provenance.md (claim -> notebook -> command -> value), runbook.md (training)
output/                 figures, trained models, run metrics (ignored by git)
reproduce.py            single entry point: quick | all | train
```

## Determinism

- The datasets are drawn from `numpy.random.default_rng(seed)`, so the same
  seeds give identical diagrams everywhere. The checksums of the arrays are
  committed (the `.npz` containers themselves may differ in their metadata).
- The scan and the exact CNN are exact integer computations.
- Training seeds TensorFlow, numpy and Python
  (`tf.keras.utils.set_random_seed`) and enables deterministic kernels.
  Shuffling and augmentation depend only on (seed, epoch, batch). A run
  therefore repeats exactly on the same hardware and software
  (`verification/test_c7_configs_and_training.py`). Across different hardware,
  floating-point results of training may differ slightly.
- The original datasets were not seeded and exist only on the authors' Google
  Drive; they are the only exact copy of the chapter's data. All scripts accept
  them unchanged (`--set data.source=original`).

## Citation and licence

If you use this code, please cite the chapter (see
[`CITATION.cff`](CITATION.cff)):

```bibtex
@incollection{rollier2025convolutional,
  author    = {Rollier, Michiel and Daly, Aisling J. and Baetens, Jan M.},
  title     = {Convolutional Neural Networks for Automated Cellular Automaton Classification},
  booktitle = {Advances in Cellular Automata},
  editor    = {Adamatzky, Andrew and Sirakoulis, Georgios Ch. and Mart{\'i}nez, Genaro J.},
  series    = {Emergence, Complexity and Computation},
  volume    = {53},
  pages     = {49--70},
  publisher = {Springer, Cham},
  year      = {2025},
  doi       = {10.1007/978-3-031-81097-8_3}
}
```

The code is released under the MIT licence ([`LICENSE`](LICENSE)).
