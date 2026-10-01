# Provenance: which notebook produced what

§1–§5 are the audit of the original Colab notebooks that came with the source
material for this repository. They are reproduced unchanged; code comments
refer to their section numbers (e.g. "§3.1"). The notebooks are in
`notebooks/archive/`. The thesis chapter (`paper/…tex` below) is not part of
the repository. §6 maps every claim to a command, a value produced here and a
test; §7 lists what building the repository added or corrected.

Audit of the original Colab notebooks against the book chapter

> Rollier, M., Daly, A. J., Baetens, J. M. (2025). *Convolutional Neural Networks for Automated Cellular Automaton Classification.* In Adamatzky, Sirakoulis, Martínez (eds), *Advances in Cellular Automata*, Emergence, Complexity and Computation vol. 53, Springer, pp. 49–70. doi:10.1007/978-3-031-81097-8_3. Preprint: arXiv:2409.02740.

and its revised version as thesis Chapter 8 (`paper/thesis-ch08-automated_classification.tex`).

Audit done on 1 Oct 2026 by reading every code cell and saved text output. Images inside the notebooks were not inspected.

## How to read this file

- **Confirmed** = the saved code and/or saved output match the chapter.
- **Likely** = the code fits, but the saved output is missing or comes from a different setting.
- **Not found** = the number appears in the chapter but in no saved output.

The notebooks were run cell by cell with toggles (`RUN_AGAIN`, `OUTPUT_TYPE`, `SAVEFIG`, …) flipped between runs. Their saved state is therefore **not** a linear, top-to-bottom record. Several notebooks would crash if run top to bottom as saved.

## 1. Notebook inventory

| Notebook | Role in the chapter | Status |
|---|---|---|
| `create-dataset` | Generates the 64×64 ECA datasets (1024/rule train+val, 128/rule test) and the Li–Packard lookup JSONs | Essential. Saved state is set to 112×112; the 64×64 run used `WIDTH=LENGTH=64`, `NSIM_PER_RULE_TRAIN=1024`, `NSIM_PER_RULE_TEST=128`. The noise section is for the earlier noisy-CA work (SummerSolstice 2023), **not** this chapter. |
| `benchmark-classification-by-scanning` | Sec. "Benchmark: a simple grid search" (370 / 33 undetermined diagrams, the rule-59 case) | Essential, **has a bug** (see §3.2) |
| `reproducing-silverman2019_default` | Sec. "The default CNN" and its performance | Essential |
| `custom-made-cnn` | Sec. "An exact CNN for rule identification" (hand-set weights, 2360 params) | Essential |
| `reproducing-silverman2019_trimmed` | Sec. "Trimming the default CNN" (2×2 kernels, 4-4-MP-4-4-16, 389 params) | Essential |
| `data-augmentation` | Sec. "Data augmentation", the augmentation table, the final model and confusion matrix | Essential, **has a bug** (see §3.1) |
| `MLP-verification_cropped` | Probably the "fully-connected ANN … just over 95% … well over a million parameters" remark | Supporting. Uses a 28×28 crop and 1024→512 dense layers; the chapter does not mention the crop. |
| `MLP-verification` | Earlier MLP attempt (full 64×64, 256-node layer, rule output) | Exploratory, not reported |
| `reproducing-comelli2021_default` | Reproduction of Comelli et al. 2021 on 32×32 crops (98.11% val) | Exploratory, not reported (the chapter cites Comelli's own 97.58%) |
| `create-illustration-of-convolution` | The convolution + ReLU illustration figure | Figure only |

## 2. Numbers and figures → source

### Numbers

| Chapter claim | Source | Status |
|---|---|---|
| Class sizes 24 / 97 / 89 / 10 / 36 | `create-dataset` cell 7 | Confirmed (re-run here) |
| 88 independent rules | `create-dataset` cell 5 → `equiv_ECA_rules.json` | Confirmed (re-run here) |
| 370 train+val / 33 test undetermined diagrams | `benchmark-…` cell 10 | Confirmed as output, but see §3.2 |
| Only one diagram (rule 59) misclassifiable by the scan | `benchmark-…` cells 13, 17 | Confirmed as output, but see §3.2 |
| Default CNN: 6016 / 7301 params | `silverman2019_default` cell 11 | Confirmed by arithmetic |
| Default CNN rule ID: 99.85% train / 99.85% val | `silverman2019_default` cell 13 | Confirmed (0.99853 / 0.99852) |
| Default CNN LP class: 99.99%, 12 epochs, 6 wrong | `silverman2019_default` cells 17–19 | Likely (6 diagrams plotted; the accuracy itself is not in a saved output) |
| Exact CNN: 2360 params | `custom-made-cnn` cell 12 | Confirmed |
| Exact CNN: 331 misidentified, 99.87% | `custom-made-cnn` cells 14–15 | **Not found.** Inference is not saved. Code comments give 175 + 9 + 12 + 188 = **384** errors, which contradicts both 331 and the "≤ 370" logic (see §3.2). |
| Trimmed CNN: 389 / 1504 params | `silverman2019_trimmed` cell 11 | Confirmed by arithmetic |
| Trimmed CNN LP val 99.10% | `silverman2019_trimmed` | Likely. Saved run gives 98.90%; the notebook header notes 99.10% from an earlier run. |
| Trimmed CNN rule val 71.19% | — | **Not found.** Header notes 67.10% from a different run. |
| Equivalence-class model: 664 params | `data-augmentation` cell 16 (`output_type='independent'`) | Confirmed by arithmetic |
| Augmentation table (all 18 values) | `data-augmentation` cell 17 with toggles | **Not found** in saved outputs; header notes a different "all augmentations" run (92.03% / 22.78%) |
| Final model LP test 98.17% | `data-augmentation` cell 24 | Confirmed (0.981689) |
| Final model independent-rule test 61.25% | `data-augmentation` cell 26 | Likely (predictions made, value not printed) |
| MLP "just over 95%", over a million params | `MLP-verification_cropped` | Likely (val accuracy 95.2–95.9% during training; ≈1.3 M params) |

Parameter-count check: default rules model = 112 + 1552 + 4352 = 6016, plus 1285 for the 5-class head = 7301. Trimmed = 20 + 68 + 68 + 68 + 80 + 85 = 389.

### Figures (thesis file names)

| Figure | Source |
|---|---|
| `LP_classes-examples.pdf` | `silverman2019_default` cell 8 (or same cell in `_trimmed`) |
| `benchmark-undetermined_diagrams.pdf` | `benchmark-…` cell 15 |
| `benchmark-undetermined_diagrams-examples2.pdf` | `benchmark-…` cell 17 |
| `default_CNN-architecture.jpg`, `trimmed_CNN-architecture.jpg` | Not from these notebooks; probably drawn with NN-SVG |
| `example-convolution_and_relu.pdf` | `create-illustration-of-convolution` cell 7 |
| `default_model-misidentified_val_set.pdf` | `silverman2019_default` cell 16 |
| `default_model-all_wrong_LP_classes.pdf` | `silverman2019_default` cell 19 |
| `convolutional_decomposition-example_120.pdf` | Probably `custom-made-cnn` cell 9 (no save call; the example diagram is random, so rule 120 was presumably picked by hand) |
| `augmentation-examples.pdf` | `data-augmentation` cell 11 |
| `confusion_matrix-final.pdf` | `data-augmentation` cell 26 (saves a two-panel figure; the published one is the LP panel) |

## 3. Bugs that affect published results

### 3.1 Coarse-graining augmentation blanks the image (high impact)

In `data-augmentation`, `CustomDataGenerator._coarse_grain` does:

```python
width = image.shape[0]                  # = 64
aug_image = image.reshape(32, 2, 32, 2)
aug_image = aug_image / width**2        # divides by 4096 (!)
aug_image = aug_image.mean(axis=(1,3))  # mean already normalises
```

The `.mean()` already averages each 2×2 block, so the extra division is wrong. Because `width` is 64 here, every pixel ends up ≤ 1/4096. Verified numerically: the maximum output value is 0.000244 instead of 1.0.

**In plain terms:** every diagram that was "coarse-grained" during training was fed to the network as an almost completely black image, still carrying its original label. The network never saw a real coarse-grained diagram.

The standalone `coarse_grain()` used for `augmentation-examples.pdf` divides by 4 instead (max 0.25). `imshow` rescales colours automatically, so the figure still looks correct.

**Affected:** the "Coarse-grained" and "All augmentations" rows of the augmentation table, and the final model (98.17% LP, 61.25% independent rule), which was trained with coarse-graining on. The chapter's explanation of why coarse-graining hurts rule identification does not describe what was actually tested.

**Fix:** delete the `/ width**2` line. The block mean is already in [0, 1].

### 3.2 Benchmark scan uses wrong periodic padding (medium impact)

`benchmark-…` → `_return_periodic_diagram` pads with the **first** column on the left and the **last** column on the right. Periodic boundaries need the opposite: last column on the left, first on the right.

Example: for the row `100000` the notebook's padding yields triplets {110, 100, 000}. The true periodic row yields {001, 100, 000, 010}. So the scan both invents triplets and misses real ones.

**Affected:** the 370 / 33 undetermined counts, and possibly the single rule-59 example. `custom-made-cnn` pads correctly (`np.pad(..., mode='wrap')`). Its code comments report 384 errors, more than 370. That is impossible if 370 were the true number of undetermined diagrams, because the exact CNN can only fail on undetermined ones. The thesis version states this subset relation explicitly ("the CNN can only fail on the 370 diagrams"), so it is the more exposed of the two texts.

**Caveat:** the 384 comes from comments, not saved output, so its provenance is uncertain. Re-run both with correct padding before trusting either number.

**Fix:** `np.concatenate((last_col, diagram, first_col), axis=1)`.

## 4. Bugs and loose ends that do not affect published numbers

- `create-dataset`: no random seed is set, so the dataset cannot be regenerated bit-for-bit. The `.npy` files on Drive are the only exact copy.
- `create-dataset` `add_noise(..., noise_type='missing')` writes −1 into a `uint8` array. It wraps to 255 on older NumPy and raises an error on NumPy ≥ 2. Not used in this chapter.
- The label files `*_labels_class.npy` and `*_labels_indep.npy` are loaded by `silverman2019_default` and `data-augmentation`, but **no notebook creates them**. They must be derived from `*_labels.npy` and the JSON lookups. The 88-class index is the key order of `equiv_ECA_rules.json` (`indep_to_num` in `data-augmentation` cell 7).
- `silverman2019_default`: a stray `zz` line in cell 15 (NameError), and cell 9 uses `r_train5` and `r_train_rule`, which this notebook never defines.
- `silverman2019_trimmed` cell 12: when `output_type == 'rules'` it saves `model_trimmed_rules`, which is never created.
- `data-augmentation` and `reproducing-comelli2021_default` cells 19–21 reference undefined variables (`misclassified`, `test_predicted_class`, `r_val_class`).
- `data-augmentation`: `salt_and_pepper_fraction` is hard-coded to 0.01 inside the class, so the constructor argument is ignored. It is consistent with the chapter's 1%.
- `whichClass` is very slow (a Python loop with `np.isin`); replace it with a 256-entry lookup array.
- Kernel shape notation: Keras `(2, 3)` = 2 rows (time) × 3 columns (space). The chapter writes this as "3×2". Both mean the same T-tetromino window.

## 5. Not in this bundle

- **Datasets.** `ECAs_1024sims-per-rule_64x64_train_*` and `ECAs_128sims-per-rule_64x64_test_*` (`.npy`, about 1.1 GB + 0.13 GB). Drive: `Colab Notebooks/Cellular Automata/Spacetime Diagram Classification/Files`.
- **Trained models.** `Spacetime Diagram Classification/Models` (Keras SavedModel folders, names as in the notebooks).
- **Architecture diagrams.** The two `.jpg` files (NN-SVG).

## 6. This repository: claim → command → value → test

Values in the "This repository" column were produced on 1 October 2026 by
the commands shown, on the seeded datasets (`data/results/dataset_checksums.json`),
with Python 3.11.5, TensorFlow 2.15.0 and numpy 1.26.4 on Windows 11 (Intel
i7-9850H, CPU only). The committed files in `data/results/` hold the full
output; CI regenerates them on Linux and fails if a byte differs.

Status: ✅ reproduced · ≈ consistent (a fresh random sample, so exact equality is not expected) ·
⚠ differs from the chapter (see §7) · ⏳ needs a training run that has not been done yet.

| Claim (chapter section) | Chapter | Command | This repository | Test | Status |
|---|---|---|---|---|---|
| Class sizes (Tab. 2) | 24 / 97 / 89 / 10 / 36 | `pytest verification/test_c1_lookup.py` | 24 / 97 / 89 / 10 / 36 | C1 | ✅ |
| 88 independent rules (Sec. 3.3) | 88 | same | 88; exactly the orbits under mirror and complement | C1 | ✅ |
| Self-mirror / self-complementary rules (Sec. 3.4) | 64 / 16 | same | 64 / 16 | C1 | ✅ |
| Simulator | cellpylib | `pytest verification/test_c2_simulation.py` | numpy, bit-identical to cellpylib for all 256 rules | C2 | ✅ |
| Undetermined diagrams, train+val / test (Sec. 2.1) | 370 / 33 | `python scripts/run_benchmark.py` | faithful padding: **345 / 43**; corrected: **601 / 83** | C3, C9 | ≈ faithful, ⚠ corrected (§7.5) |
| "Not a single diagram contains fewer than seven rule-table entries" | 0 diagrams | same | one diagram misses 2 (train+val), one misses 3 (test), in both modes | C9 | ⚠ (§7.5) |
| Diagrams whose class the scan can get wrong | 1 (rule 59) | same | faithful 4 + 1, corrected 5 + 2 (train+val + test) | C9 | ⚠ (§7.5) |
| Default CNN parameters (Sec. 2.2) | 6016 / 7301 | `python scripts/count_parameters.py` | 6016 / 7301 | C4 | ✅ |
| Default CNN channel sizes (Sec. 2.2) | 62 × 63, 60 × 62 | `pytest verification/test_c4_models.py` | 62 × 63, 60 × 62 | C4 | ✅ |
| Exact CNN parameters (Sec. 3.1) | 2360 | `python scripts/count_parameters.py` | 2360 | C4 | ✅ |
| Exact CNN returns the rule whenever all neighbourhoods occur | — | `pytest verification/test_c4_models.py` | all 256 rules | C4 | ✅ |
| Exact CNN errors on train+val (Sec. 3.1) | 331 (99.87%) | `python scripts/run_exact_cnn.py` | **299 (99.886%)**; test: 40 (99.878%) | C9 | ≈ (§7.6) |
| Exact CNN fails only on undetermined diagrams, exactly where the missing entry is 1 | yes | same | yes, against the *corrected* scan (diagram for diagram); 136 of its 299 errors are outside the faithful scan's undetermined set | C4, C9 | ✅ corrected, ⚠ as argued (§7.6) |
| Undetermined diagrams the exact CNN guesses right | 39 of 370 | same | 302 of 601 | C9 | ⚠ (§7.6) |
| Trimmed CNN parameters (Sec. 3.2, 3.3) | 389 / 1504 / 664 | `python scripts/count_parameters.py` | 389 / 1504 / 664 | C4 | ✅ |
| Trimmed CNN receptive field (Sec. 3.2) | 8 × 8 | — | 8 × 8 (2 → 3 → pool → 4 → 6 → 8; not tested) | — | ✅ |
| Fully-connected network, "well over a million" parameters (Sec. 3.2) | > 10⁶ | `python scripts/count_parameters.py` | 1,050,117 (single 256 layer) / 1,461,253 (cropped) | C4 | ✅ (§7.2) |
| Inversion → complement rule, mirroring → mirror rule (Sec. 3.4) | stated | `pytest verification/test_c5_augment.py` | checked on simulated diagrams | C5 | ✅ |
| Coarse-graining bug (§3.1) | — | `pytest verification/test_c5_augment.py verification/test_c6_notebook_fidelity.py` | faithful max 1/4096 = 0.000244, equal to the notebook's own code | C5, C6 | ✅ |
| Benchmark padding bug (§3.2) | — | `pytest verification/test_c3_benchmark.py verification/test_c6_notebook_fidelity.py` | faithful scan equal to the notebook's string scan, diagram for diagram | C3, C6 | ✅ |
| Default CNN rule identification (Sec. 2.3) | 99.86% / 99.85% | `python scripts/train.py --config configs/default_rules.yaml` | not run | — | ⏳ |
| Default CNN class identification (Sec. 2.3) | 99.99%, 12 epochs, 6 wrong | `... configs/default_classes.yaml` | not run | — | ⏳ |
| Trimmed CNN (Sec. 3.2) | 99.10% / 71.19% | `... configs/trimmed_classes.yaml`, `trimmed_rules.yaml` | not run | — | ⏳ |
| Augmentation table (Sec. 3.4), 18 values | see chapter | `... configs/augmentation_table.yaml` | not run | — | ⏳ |
| Final model, test set (Sec. 3.4) | 98.17% / 61.25% | `... configs/final_classes.yaml`, `final_independent.yaml` | not run | — | ⏳ |
| Fully-connected network, validation (Sec. 3.2) | just over 95% | `... configs/mlp_cropped_classes.yaml` | not run | — | ⏳ |

Every training config has been smoke-tested (one epoch on six diagrams per
rule; `python scripts/train.py --config <file> --smoke`), including all 18
cells of the augmentation table and the faithful final model.

## 7. Findings while building the repository

These add to §1–§5; they were found by running the code above against the
notebooks, the saved outputs (including the images, which §1–§5 did not
inspect) and the chapter.

### 7.1 The validation split is stratified on different labels

`reproducing-silverman2019_default` and `data-augmentation` call
`train_test_split(..., stratify=r_train_all)`, where `r_train_all` is
whatever label file was loaded: the Li-Packard class, the equivalence class
or the rule, depending on `OUTPUT_TYPE`. `reproducing-silverman2019_trimmed`
and the MLP notebooks stratify on the rule. The validation set therefore
differs between experiments. The config field `split.stratify_by`
(`label` or `rule`) records which one each experiment used.

The label arrays have shape `(n, 1)` (`create-dataset` writes the rules that
way, and the derived label arrays in the notebooks have the same shape). For a
2-D `stratify` array, scikit-learn's `StratifiedShuffleSplit` turns every row
into a string, so the classes are ordered as strings ("0", "1", "10", "100",
…) when it draws the split. With 256 rules this gives a split that shares only
16,644 of the 65,536 validation diagrams with the split from 1-D integer
labels. For the five classes the two orders coincide. `data.split_indices`
therefore stratifies on `(n, 1)` arrays, after the same pre-shuffle with
`np.random.seed(2023)`. C6 checks it against the notebooks' own calls. Two
assumptions remain unverifiable from the saved notebooks: that the global
generator was untouched between the seed and the shuffle in the Colab session,
and that the never-written `*_labels_indep.npy` file (§4) also had shape
`(n, 1)`. The runbook gives a check on the original data.

### 7.2 The fully-connected comparison mixes two notebooks

The chapter describes "a fully-connected ANN with a single 256-node hidden
layer" with a validation accuracy "just over 95%" and "well over a million"
parameters.

- The single-256-layer network is `MLP-verification` (64 × 64 → 256 → output).
  Its only saved run is for *rules*: 73.96% validation accuracy, 1,114,624
  parameters. For classes it would have 1,050,117 parameters, which is just
  over a million rather than well over.
- The ~95% comes from `MLP-verification_cropped`: a 28 × 28 crop of the
  top-left corner → 1024 → 512 → 256 → 5, with 1,461,253 parameters (saved
  model summary). §2 above gives "≈1.3 M"; the saved summary says 1,461,253.
  Its saved log reaches 95.2–95.9% validation accuracy at epochs 4, 5 and
  7 (94.4% at epoch 6), then stops at epoch 8, apparently interrupted.

Both are configs here (`mlp_single_layer_classes.yaml`,
`mlp_cropped_classes.yaml`). The two notebooks also differ in learning rate:
`MLP-verification` uses 1e-4, `MLP-verification_cropped` 1e-3.

### 7.3 Salt-and-pepper level in the augmentation figure

The caption of `augmentation-examples.pdf` says 5% noise. The saved state of
`data-augmentation` cell 11 uses `salt_and_pepper_fraction = 0.1`, and the
saved image is titled "level 0.1". `scripts/make_figures.py` follows the
caption (0.05). Training always uses 1%, as the chapter says.

### 7.4 Alternative rules when more than one entry is missing

`suggest_alternative_rule` flips *all* missing outputs at once. Its own comment
warns that it "only works well when only one triplet is missing". With k
missing entries there are 2^k − 1 alternative rules. Corrected mode checks them
all; faithful mode checks only the notebook's one. The two agree whenever one
entry is missing. On the regenerated test set, one diagram of rule 232 lacks
011, 110 and 111. Faithful mode finds one alternative rule in another class
(rule 32, null); corrected mode finds six. Both flag the diagram as
class-ambiguous.

### 7.5 The benchmark on regenerated data

| | train+val | test |
|---|---|---|
| Chapter (faithful padding, original data) | 370 | 33 |
| Faithful padding, regenerated data | 345 | 43 |
| Corrected padding, regenerated data | 601 | 83 |

- With the notebook's padding, the counts on fresh data agree with the
  chapter's within sampling noise (Poisson σ ≈ 19 and 6). So the 370 / 33 are
  very probably what that procedure gives. The correct counts are about 1.7×
  higher.
- Missing neighbourhoods are almost all 000 or 111 (293 and 283 of 601),
  followed by 010 and 101. Undetermined diagrams lie mostly in the fixed-point
  (326) and periodic (147) classes, then null (109). Only 3 are locally
  chaotic and 16 chaotic (corrected mode, train+val).
- The chapter's "not a single diagram … contains fewer than seven rule
  table entries" and "correct LP classification for all but one spacetime
  diagram" are properties of that one sample. On the regenerated data both
  modes find diagrams with two and with three missing entries, and 5–7
  class-ambiguous diagrams (rules 4, 114, 138, 208, 255 in train+val; 112, 232
  in test, corrected mode). None of them is rule 59. Statements of this kind
  should be phrased as expectations ("about one in 10⁵ diagrams") rather than
  as exact counts.

### 7.6 The exact CNN and the "331 of 370" argument

On the regenerated data, the exact CNN misidentifies 299 of 262,144
training-and-validation diagrams (99.886%) and 40 of 32,768 test diagrams. Its
answer equals the rule table read off by the corrected scan with missing
entries set to 0, for every diagram. Its errors are therefore exactly the
undetermined diagrams whose missing entry maps to 1. By missing bit they are
144 (bit 0, i.e. 000), 6 (bit 2), 9 (bit 5) and 140 (bit 7, i.e. 111). This is
the same pattern as the notebook's comments: 175 / 9 / 12 / 188 = 384.

The chapter's argument combines two procedures. "The CNN can only fail on the
370 diagrams with incomplete information … for the remaining 39 diagrams, the
implicit guess happens to be correct" takes the 370 from the scan with
reversed padding (§3.2) and the 331 from the network, which pads correctly.
With one consistent padding:

- the network fails on 299 of the 601 undetermined diagrams and guesses 302
  correctly, roughly half, as expected when a missing entry is equally likely
  to be 0 or 1;
- of the 299 errors, only 163 lie within the 345 diagrams that the faithful
  scan flags (`n_misidentified_among_faithful_undetermined` in
  `data/results/exact_cnn.json`; on the test set 20 of 40).

On the original data, 331 errors would imply roughly 660 undetermined
diagrams, not 370. The 384 of the notebook's comments, larger than 370, is
consistent with this (§3.2). Which of 331 and 384 belongs to the final
dataset cannot be decided without the original files: run
`python scripts/run_exact_cnn.py --set data.source=original --set paths.original_data_dir=<Drive>/Files`.
Suggested wording: "it misidentifies N diagrams, precisely those undetermined
diagrams whose missing entry maps to 1; the other M undetermined diagrams
happen to be guessed correctly", with N and M from that run.

### 7.7 The confusion-matrix numbers are in the saved image

The saved image of `data-augmentation` cell 26 (5 × 5 panel, normalised over
predictions) contains the absolute counts. They confirm every number the
chapter quotes for the final model:

- accuracy (3018 + 12287 + 11149 + 1188 + 4526) / 32768 = 98.169%;
- per actual class: null 98.24%, fixed point 98.96%, periodic 97.87%,
  locally chaotic 92.81%, chaotic 98.22%, mean 97.22%;
- 219 periodic diagrams predicted as fixed point;
- chaotic → locally chaotic 64 / 4608 = 1.39%, and 64 / 1282 = 4.99% of the
  locally-chaotic predictions.

The row totals are 128 × the class sizes (3072, 12416, 11392, 1280, 4608), so
the matrix is indeed for the full test set. Status of "Final model LP test
98.17%" and of the per-class claims: **Confirmed**. The 61.25% for the 88
equivalence classes is in the same cell's right-hand panel only as colours.

### 7.8 Smaller observations

- **Linear layers.** The Dense(256) layer before the 5-class output of the
  default CNN, and the Dense(16) of the trimmed CNN, have no activation.
  Together with the output layer they form a single linear map followed by a
  softmax. This changes the parameter count, not the function class. It is
  kept as in the notebooks.
- **Early stopping.** Keras restores the best weights only when early stopping
  triggers. A run that reaches the epoch limit keeps its last weights.
- **Simulation speed.** The vectorised simulator generates all 294,912
  diagrams in about 12 s, against hours for cellpylib, with identical output.
- **Run times on a laptop CPU.** Smoke runs estimate a full epoch at about
  100 s for the default CNN, 40–50 s for the trimmed CNN (60 s on all 262,144
  diagrams) and 35–42 s for the MLPs, on an Intel i7-9850H without GPU.
  The chapter reports 25 s per epoch for the default CNN on an NVIDIA T4.
  See `docs/runbook.md`.
- **Architecture figures.** `default_CNN-architecture.jpg` and
  `trimmed_CNN-architecture.jpg` were drawn with NN-SVG and are not
  regenerated.

### 7.9 Errata to §1–§5

- §2, "Default CNN rule ID: 99.85% train / 99.85% val … Confirmed": the
  chapter actually writes 99.86% for the training set. The saved output,
  0.998530, rounds to 99.85%. The validation value matches.
- §2, MLP: "≈1.3 M params". The saved model summary of
  `MLP-verification_cropped` gives 1,461,253 (§7.2).
- §2, "Default CNN LP class: 99.99%, 12 epochs, 6 wrong" and "Final model
  independent-rule test 61.25%" remain "Likely": the predictions were made but
  the values were never printed.
