"""C6: faithful mode reproduces the notebooks' code, bug for bug.

The functions prefixed ``nb_`` are copied from the archived notebooks (only
indentation and names changed). Each test checks that the vectorised
reimplementation returns the same result.
"""

import numpy as np
import pytest

from eca_cnn.augment import coarse_grain
from eca_cnn.benchmark import NEIGHBOURHOODS, observed_neighbourhoods


# -- benchmark-classification-by-scanning, cell 6 ---------------------------

def nb_return_periodic_diagram(diagram):
    first_col = np.expand_dims(diagram[:, 0], axis=1)
    last_col = np.expand_dims(diagram[:, -1], axis=1)
    periodic_diagram = np.copy(diagram)
    periodic_diagram = np.concatenate((first_col, periodic_diagram, last_col), axis=1)
    return periodic_diagram


def nb_scan_diagram_for_triplets(diagram):
    timesteps, cells, _ = diagram.shape
    periodic_diagram = nb_return_periodic_diagram(diagram)
    remaining_triplets = set({'000', '001', '010', '011', '100', '101', '110', '111'})
    found_triplets = set({})
    for timestep in range(timesteps - 1):
        remaining_triplets = remaining_triplets.difference(found_triplets)
        current_config = periodic_diagram[timestep, :, 0]
        current_config_as_str = np.array2string(current_config, separator='')[1:-1]
        for triplet in remaining_triplets:
            if triplet in current_config_as_str:
                found_triplets.add(triplet)
                if len(found_triplets) == 8:
                    return found_triplets
    return found_triplets


def test_faithful_scan_equals_notebook_scan(small_set):
    diagrams = small_set.diagrams()
    ours = observed_neighbourhoods(diagrams, faithful=True)
    for i in range(len(diagrams)):
        expected = nb_scan_diagram_for_triplets(diagrams[i][..., None])
        assert {NEIGHBOURHOODS[k] for k in np.flatnonzero(ours[i])} == expected, i


def test_faithful_scan_equals_notebook_on_sparse_diagrams():
    # Rows with few triplets stress the padding; small_set rarely has them.
    from eca_cnn.simulate import evolve

    rng = np.random.default_rng(3)
    for rule in (0, 4, 8, 59, 204, 232, 255):
        for p in (0.05, 0.5, 0.95):
            init = (rng.random((4, 64)) < p).astype(np.uint8)
            d = evolve(init, rule, 64)
            ours = observed_neighbourhoods(d, faithful=True)
            for i in range(len(d)):
                assert {NEIGHBOURHOODS[k] for k in np.flatnonzero(ours[i])} == \
                    nb_scan_diagram_for_triplets(d[i][..., None])


# -- data-augmentation, cell 13 (CustomDataGenerator._coarse_grain) ----------

def nb_coarse_grain(image):
    width = image.shape[0]
    height = image.shape[1]
    if width % 2 != 0 or height % 2 != 0:
        raise Exception("Image dimensions must be even for coarse graining to apply properly.")
    aug_image = image.reshape(image.shape[0] // 2, 2, image.shape[0] // 2, 2)
    aug_image = aug_image / width**2  # normalise
    aug_image = aug_image.mean(axis=(1, 3))
    aug_image = np.repeat(np.repeat(aug_image, 2, axis=0), 2, axis=1)
    aug_image = np.expand_dims(aug_image, axis=2)
    return aug_image


def test_faithful_coarse_grain_equals_notebook(rng):
    x = rng.integers(0, 2, size=(6, 64, 64, 1)).astype(np.float32)
    ours = coarse_grain(x, faithful=True)
    for i in range(len(x)):
        np.testing.assert_allclose(ours[i], nb_coarse_grain(x[i]), rtol=1e-6)


def test_corrected_coarse_grain_is_notebook_without_the_division(rng):
    x = rng.integers(0, 2, size=(6, 64, 64, 1)).astype(np.float32)
    ours = coarse_grain(x, faithful=False)
    for i in range(len(x)):
        np.testing.assert_allclose(ours[i], nb_coarse_grain(x[i]) * 64**2, rtol=1e-6)


# -- custom-made-cnn, cells 4, 8, 10, 13 --------------------------------------

def nb_exact_weights():
    def _base_repr(number, base=2, length=8):
        base_repr = np.base_repr(number, base=base)
        if len(base_repr) < length:
            base_repr = (length - len(base_repr)) * '0' + base_repr
        return base_repr

    conv_kernels = np.array([[[-1, -1, -1], [0, 1, 0]], [[-1, -1, +1], [0, 1, 0]],
                             [[-1, +1, -1], [0, 1, 0]], [[-1, +1, +1], [0, 1, 0]],
                             [[+1, -1, -1], [0, 1, 0]], [[+1, -1, +1], [0, 1, 0]],
                             [[+1, +1, -1], [0, 1, 0]], [[+1, +1, +1], [0, 1, 0]]])
    biases = np.array([-np.sum(conv_kernels[i, 0, :] == 1) for i in range(8)])
    weight_factor = 10
    dense_weights = np.array([[2 * int(i) - 1 for i in _base_repr(j)] for j in range(256)])
    dense_biases = -np.array([_base_repr(j).count('1') for j in range(256)]) + 1
    dense_weights *= weight_factor
    dense_biases *= weight_factor
    conv_kernels_cnn = np.expand_dims(np.transpose(conv_kernels, (1, 2, 0)), axis=2)
    return [conv_kernels_cnn, biases.copy(), np.flip(dense_weights, axis=1).T, dense_biases.copy()]


def test_exact_weights_equal_notebook():
    from eca_cnn.models import exact_weights

    for ours, theirs in zip(exact_weights(10.0), nb_exact_weights()):
        assert ours.shape == theirs.shape
        np.testing.assert_array_equal(ours, theirs)


def test_periodic_padding_layer_equals_np_pad_wrap(small_set):
    import tensorflow as tf

    from eca_cnn.models import PeriodicPadding

    x = small_set.diagrams(slice(0, 16))[..., None].astype(np.float32)
    ours = PeriodicPadding(1)(tf.constant(x)).numpy()
    np.testing.assert_array_equal(ours, np.pad(x, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="wrap"))


# -- the split (every training notebook, cells 7/8) --------------------------

@pytest.mark.parametrize("stratify_by, output", [("rule", "rules"), ("label", "classes"), ("label", "independent")])
def test_split_equals_notebook_procedure(small_set, lookup, stratify_by, output):
    from sklearn.model_selection import train_test_split

    from eca_cnn.config import SplitConfig
    from eca_cnn.data import split_indices

    rules = small_set.rules
    labels = lookup.labels(rules, output)
    # As in the notebooks: labels of shape (n, 1), as create-dataset writes them.
    r_all = (rules if stratify_by == "rule" else labels).reshape(-1, 1)
    x_all = np.arange(len(rules))  # stands in for the diagrams
    np.random.seed(2023)
    shuffle_indices = np.random.permutation(range(r_all.shape[0]))
    x_all, r_all = x_all[shuffle_indices], r_all[shuffle_indices]
    val_size = int(x_all.shape[0] * 1 / 4)
    nb_train, nb_val, _, _ = train_test_split(x_all, r_all, test_size=val_size, random_state=0,
                                              shuffle=True, stratify=r_all)
    train, val = split_indices(rules, labels, SplitConfig(stratify_by=stratify_by))
    assert np.array_equal(train, nb_train) and np.array_equal(val, nb_val)


def test_split_depends_on_label_shape(small_set):
    """Why the (n, 1) shape matters: 1-D rule labels give another split."""
    from sklearn.model_selection import train_test_split

    from eca_cnn.config import SplitConfig
    from eca_cnn.data import split_indices

    rules = small_set.rules
    _, val = split_indices(rules, rules, SplitConfig(stratify_by="rule"))
    perm = np.random.RandomState(2023).permutation(len(rules))
    _, val_1d = train_test_split(perm, test_size=len(rules) // 4, random_state=0, shuffle=True,
                                 stratify=rules[perm])
    assert not np.array_equal(np.sort(val), np.sort(val_1d))


# -- reproducing-silverman2019_default cell 11, data-augmentation cell 16 -----

def nb_silverman_model(diagram_dims=(60, 120, 1), conv_kernel_dims=(2, 3), dilation_rate=(1, 1),
                       output_type='classes'):
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, GlobalMaxPooling2D

    inputs = Input(diagram_dims, dtype=tf.float32)
    x = Conv2D(16, conv_kernel_dims, padding='valid', dilation_rate=dilation_rate)(inputs)
    x = Activation('relu')(x)
    x = Conv2D(16, conv_kernel_dims, padding='valid')(x)
    x = Activation('relu')(x)
    x = GlobalMaxPooling2D()(x)
    x = Flatten()(x)
    if output_type == 'classes':
        x = Dense(256)(x)
        output = Dense(5, activation="softmax")(x)
    elif output_type == 'rules':
        output = Dense(256, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=output)


def nb_silverman_model_trimmed(diagram_dims=(64, 64, 1), conv_kernel_dims=(3, 3), dilation_rate=(1, 1),
                               output_type='classes'):
    import tensorflow as tf
    from tensorflow.keras import Input
    from tensorflow.keras.layers import Activation, Conv2D, Dense, GlobalMaxPooling2D, MaxPooling2D

    inputs = Input(diagram_dims, dtype=tf.float32)
    x = Conv2D(4, conv_kernel_dims, padding='valid', dilation_rate=dilation_rate)(inputs)
    x = Activation('relu')(x)
    x = Conv2D(4, conv_kernel_dims, padding='valid', dilation_rate=dilation_rate)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(4, conv_kernel_dims, padding='valid')(x)
    x = Activation('relu')(x)
    x = Conv2D(4, conv_kernel_dims, padding='valid')(x)
    x = Activation('relu')(x)
    x = GlobalMaxPooling2D()(x)
    if output_type == 'classes':
        x = Dense(16)(x)
        output = Dense(5, activation="softmax")(x)
    elif output_type == 'rules':
        output = Dense(256, activation="softmax")(x)
    elif output_type == 'independent':
        output = Dense(88, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=output)


def _signature(model):
    """Layer types (without the no-op Flatten), weight shapes and activations."""
    layers = [l for l in model.layers if l.__class__.__name__ not in ("InputLayer", "Flatten")]
    return ([l.__class__.__name__ for l in layers], [tuple(w.shape) for w in model.weights],
            [l.get_config().get("activation") for l in layers])


@pytest.mark.parametrize("output", ["classes", "rules"])
def test_default_cnn_equals_notebook(output):
    from eca_cnn.models import default_cnn

    assert _signature(default_cnn(output)) == _signature(nb_silverman_model((64, 64, 1), (2, 3), output_type=output))


@pytest.mark.parametrize("output", ["classes", "rules", "independent"])
def test_trimmed_cnn_equals_notebook(output):
    from eca_cnn.models import trimmed_cnn

    assert _signature(trimmed_cnn(output)) == _signature(nb_silverman_model_trimmed((64, 64, 1), (2, 2),
                                                                                    output_type=output))
