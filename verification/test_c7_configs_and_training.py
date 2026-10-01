"""C7: experiment configs and the training pipeline.

- Every config in ``configs/`` loads and validates; unknown keys are rejected.
- The configs carry the chapter's hyperparameters.
- The augmentation table expands to the 18 runs of the chapter's table.
- A tiny training run goes end to end and is reproducible.
"""

from pathlib import Path

import numpy as np
import pytest

from eca_cnn.config import ROOT, expand_sweep, load_config, parse_overrides

CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_config_loads(path):
    for cfg in expand_sweep(load_config(path)):
        assert cfg.name


def test_unknown_key_is_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("inherits: " + str(ROOT / "configs" / "base.yaml") + "\ntrain:\n  epoch: 3\n")
    with pytest.raises(ValueError, match="unknown key"):
        load_config(p)


def test_chapter_hyperparameters():
    d = load_config("configs/default_classes.yaml")
    assert (d.train.batch_size, d.train.learning_rate, d.train.epochs) == (64, 1e-3, 50)
    es = d.train.early_stopping
    assert (es.monitor, es.patience, es.min_delta, es.restore_best_weights) == ("val_accuracy", 5, 1e-5, True)
    t = load_config("configs/trimmed_classes.yaml")
    assert (t.train.learning_rate, t.train.epochs, t.train.early_stopping.patience) == (5e-4, 100, 10)
    f = load_config("configs/final_classes.yaml")
    assert f.augment.augmentations == ["invert", "mirror", "coarse_grain"]
    assert (f.train.train_on, f.train.learning_rate, f.train.epochs, f.train.early_stopping.enabled) == \
        ("trainval", 5e-4, 100, False)
    assert f.mode == "corrected"
    assert load_config("configs/final_classes.yaml", parse_overrides(["mode=faithful"])).faithful


def test_augmentation_table_has_18_cells():
    cells = expand_sweep(load_config("configs/augmentation_table.yaml"))
    assert len(cells) == 18
    assert len({c.run_name for c in cells}) == 18
    assert {tuple(c.augment.augmentations) for c in cells} == {
        (), ("invert",), ("mirror",), ("coarse_grain",), ("salt_and_pepper",),
        ("invert", "mirror", "coarse_grain", "salt_and_pepper")}
    assert {c.model.output for c in cells} == {"rules", "independent", "classes"}


def test_child_config_does_not_inherit_name():
    assert load_config("configs/default_classes.yaml").name == "default_classes"


@pytest.fixture
def tiny_config(tmp_path, small_set):
    """A trimmed-CNN config on a 4-per-rule dataset written to tmp_path."""
    from eca_cnn.data import save_set

    data_dir = tmp_path / "data"
    save_set(small_set, data_dir / "trainval.npz")   # small_set was generated with seed 12345
    save_set(small_set, data_dir / "test.npz")
    overrides = parse_overrides([
        f"paths.data_dir={data_dir.as_posix()}", f"paths.output_dir={(tmp_path / 'out').as_posix()}",
        "data.n_per_rule_trainval=4", "data.n_per_rule_test=4",
        "data.seed_trainval=12345", "data.seed_test=12345",
        "train.epochs=2", "augment.augmentations=[invert, coarse_grain]",
        "train.evaluate_on=[train, val, test]", "sweep=null", "model.output=classes",
    ])
    return load_config("configs/augmentation_table.yaml", overrides)


@pytest.mark.slow
def test_tiny_training_run_is_reproducible(tiny_config):
    from eca_cnn.evaluate import load_run
    from eca_cnn.train import train

    cfg = tiny_config
    m1 = train(cfg, verbose=0)
    p1 = load_run(cfg.run_dir())["predictions"]["val"]["pred"]
    m2 = train(cfg, verbose=0)
    p2 = load_run(cfg.run_dir())["predictions"]["val"]["pred"]
    assert m1["n_train"] == 768 and m1["n_val"] == 256
    assert set(m1["results"]) == {"train", "val", "test"}
    assert m1["results"]["val"]["accuracy"] == m2["results"]["val"]["accuracy"]
    assert np.array_equal(p1, p2)
    assert (Path(cfg.run_dir()) / "model.keras").exists()


def test_faithful_runs_are_separate_only_when_training_differs():
    faithful = parse_overrides(["mode=faithful"])
    # No coarse-graining: both modes train the same network, one shared folder.
    assert load_config("configs/default_rules.yaml", faithful).run_name == "default_rules"
    # Coarse-graining: the faithful run reproduces the bug and is kept apart.
    assert load_config("configs/final_classes.yaml", faithful).run_name == "final_classes__faithful"
    cells = expand_sweep(load_config("configs/augmentation_table.yaml", faithful))
    separate = [i for i, c in enumerate(cells) if c.run_name.endswith("__faithful")]
    assert separate == [9, 10, 11, 15, 16, 17]   # the cells reproduce.py trains in faithful mode


def test_loss_and_accuracy_equal_model_evaluate(small_set, lookup):
    """evaluate.predict/summarise report what Keras's model.evaluate reports."""
    import tensorflow as tf

    from eca_cnn.batches import DiagramSequence
    from eca_cnn.data import one_hot
    from eca_cnn.evaluate import predict, summarise
    from eca_cnn.models import trimmed_cnn

    tf.keras.utils.set_random_seed(0)
    model = trimmed_cnn("rules")
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
    idx = np.arange(0, len(small_set), 2)
    labels = lookup.labels(small_set.rules[idx], "rules")
    s = summarise(predict(model, small_set, idx, labels))
    x = small_set.diagrams(idx).astype(np.float32)[..., None]
    loss, acc = model.evaluate(x, one_hot(labels, 256), verbose=0)
    assert np.isclose(s["loss"], loss, rtol=1e-4)
    assert np.isclose(s["accuracy"], acc, atol=1e-6)
