"""Convolutional neural networks for automated classification of elementary CAs.

Reproducibility package for M. Rollier, A. J. Daly and J. M. Baetens,
*Convolutional Neural Networks for Automated Cellular Automaton Classification*,
in *Advances in Cellular Automata*, Emergence, Complexity and Computation 53,
pp. 49-70, Springer, 2025 (doi:10.1007/978-3-031-81097-8_3, arXiv:2409.02740),
and of its revised version, Ch. 8 of M. Rollier's PhD thesis (Ghent University,
2026).

Modules
-------
config
    The experiment schema (one YAML file per experiment in ``configs/``), and the
    switch between ``faithful`` and ``corrected`` mode.
lookup
    Li-Packard classes and the 88 equivalence classes of ECA rules, as 256-entry
    lookup arrays; the mirror and complement symmetries of a rule.
simulate
    Vectorised ECA evolution with periodic boundaries (bit-identical to
    cellpylib, which the original notebooks used).
data
    Seeded generation of the 64 x 64 datasets, bit-packed storage, labels, the
    stratified train/validation split, and the Keras batch generator.
benchmark
    The rule-table scan of Sec. 2.1 ("Benchmark: a simple grid search").
models
    The default CNN (after Silverman 2019), the exact hand-weighted CNN, the
    trimmed CNN, and the fully-connected baselines.
augment
    Online data augmentation: inversion, mirroring, coarse-graining and
    salt-and-pepper noise, one of n per diagram.
train, evaluate
    Training with the chapter's hyperparameters; accuracies, misclassification
    counts and confusion matrices.
figures
    One function per chapter figure, named after the thesis figure file.
"""

__version__ = "1.0.0"
