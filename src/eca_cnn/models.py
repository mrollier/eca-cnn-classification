"""The networks of the chapter, built with tf.keras (Keras 2.15).

``default``  (Sec. 2.2, after Silverman 2019; ``reproducing-silverman2019_default``)
    Conv2D(16, 2x3) - ReLU - Conv2D(16, 2x3) - ReLU - GlobalMaxPool, then
    Dense(256) - Dense(5, softmax) for classes, or Dense(256, softmax) for rules.
    6016 parameters (rules) / 7301 (classes).
``exact``    (Sec. 3.1; ``custom-made-cnn``)
    Periodic padding - Conv2D(8, 2x3) - ReLU - GlobalMaxPool - Dense(256,
    softmax), every weight set by hand so that the network reads the rule
    table. 2360 parameters, none trained.
``trimmed``  (Sec. 3.2-3.4; ``reproducing-silverman2019_trimmed``,
    ``data-augmentation``)
    4 x Conv2D(4, 2x2) with a 2x2 MaxPool after the second, GlobalMaxPool,
    then Dense(16) - Dense(5, softmax) for classes, Dense(256, softmax) for
    rules or Dense(88, softmax) for equivalence classes.
    389 / 1504 / 664 parameters.
``mlp``      (the fully-connected comparison in Sec. 3.2; ``MLP-verification``
    and ``MLP-verification_cropped``)
    Flatten - Dense(h1, ReLU) - ... - Dense(n, softmax).

Kernel shapes are Keras ``(rows, columns)`` = (time, space). Keras ``(2, 3)``
spans two timesteps and three cells: one neighbourhood and its successor (a
T-tetromino). The chapter writes this as "3 x 2".

The hidden Dense(256) and Dense(16) layers of the class heads have no
activation, exactly as in the notebooks. Together with the output layer they
are therefore one linear map followed by a softmax, factorised through 256 (or
16) dimensions. This matters for the parameter counts, not for what the
network can express.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Activation, Conv2D, Dense, Flatten, GlobalMaxPooling2D,
                                     MaxPooling2D)

from .config import Config
from .lookup import N_OUTPUTS

DEFAULT_KERNELS = {"default": (2, 3), "trimmed": (2, 2), "exact": (2, 3)}


@tf.keras.utils.register_keras_serializable(package="eca_cnn")
class PeriodicPadding(tf.keras.layers.Layer):
    """Wrap the last ``width`` columns to the left and the first to the right.

    ``custom-made-cnn`` pads the data with ``np.pad(..., mode='wrap')`` before
    feeding it to the exact CNN (input 64 x 66). Doing it inside the model lets
    the exact CNN take the same 64 x 64 diagrams as every other network; the
    result is identical.
    """

    def __init__(self, width: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.width = width

    def call(self, x):
        w = self.width
        return tf.concat([x[:, :, -w:], x, x[:, :, :w]], axis=2)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        if shape[2] is not None:
            shape[2] += 2 * self.width
        return tuple(shape)

    def get_config(self):
        return {**super().get_config(), "width": self.width}


def _softmax_head(x, output: str, hidden: int):
    """Output layers shared by the default and trimmed CNNs.

    For ``classes`` a linear hidden Dense layer of ``hidden`` nodes precedes
    the 5-node softmax. For ``rules`` and ``independent`` the softmax layer
    follows the pooled features directly.
    """
    if output == "classes":
        x = Dense(hidden)(x)
        return Dense(N_OUTPUTS["classes"], activation="softmax")(x)
    return Dense(N_OUTPUTS[output], activation="softmax")(x)


def default_cnn(output: str = "classes", input_shape=(64, 64, 1), kernel=(2, 3)) -> Model:
    """The default CNN (``silverman_model`` in ``reproducing-silverman2019_default``).

    Two 16-channel convolutions can each hold a detector for every one of the
    eight neighbourhood-and-successor patterns. The network can therefore read
    the rule table, which is what the chapter shows it does.
    """
    inputs = Input(input_shape, dtype=tf.float32)
    x = Conv2D(16, tuple(kernel), padding="valid")(inputs)
    x = Activation("relu")(x)
    x = Conv2D(16, tuple(kernel), padding="valid")(x)
    x = Activation("relu")(x)
    x = GlobalMaxPooling2D()(x)
    # The notebook has a Flatten here; after global pooling it is a no-op.
    return Model(inputs, _softmax_head(x, output, hidden=256), name=f"default_{output}")


def trimmed_cnn(output: str = "classes", input_shape=(64, 64, 1), kernel=(2, 2)) -> Model:
    """The trimmed CNN (``silverman_model_trimmed``), Fig. "trimmed_CNN-architecture".

    Three design principles (chapter Sec. 3.2):

    1. A 2x2 kernel cannot hold a neighbourhood together with its successor,
       so one convolution cannot read a rule-table entry.
    2. Four layers and a central max-pool widen the receptive field to 8 x 8
       cells, enough to see mesoscopic structure.
    3. A 16-node layer fed by only four pooled values limits how much
       rule-specific information can be stored.
    """
    inputs = Input(input_shape, dtype=tf.float32)
    x = inputs
    for i in range(4):
        x = Conv2D(4, tuple(kernel), padding="valid")(x)
        x = Activation("relu")(x)
        if i == 1:
            x = MaxPooling2D()(x)  # 2x2, stride 2
    x = GlobalMaxPooling2D()(x)
    return Model(inputs, _softmax_head(x, output, hidden=16), name=f"trimmed_{output}")


def exact_weights(weight_factor: float = 10.0) -> List[np.ndarray]:
    """The hand-set weights of the exact CNN (``custom-made-cnn`` cells 8, 10, 13).

    Convolution, channel ``k`` = neighbourhood ``k`` = ``(l, c, r)``
        The kernel's top row (time t) is ``2 * (l, c, r) - 1``, i.e. +1 where
        the neighbourhood has a 1 and -1 where it has a 0. Its bottom row (time
        t+1) is ``(0, 1, 0)``. The bias is minus the number of 1s in the
        neighbourhood. The response is ``1`` exactly where neighbourhood ``k``
        occurs *and* produces a 1, and at most ``0`` elsewhere. After ReLU and
        global max pooling, channel ``k`` is bit ``k`` of the rule, or 0 if the
        neighbourhood never occurs.

    Output node ``i`` (rule ``i``)
        Weight ``w_{i,j} = 2 bin(i)_j - 1`` and bias ``b_i = 1 - sum_j
        bin(i)_j`` (chapter Sec. 3.1). For an observed rule ``r`` the node
        receives ``1 - hamming(i, r)``: 1 for ``i = r``, at most 0 otherwise.
        All dense weights are multiplied by ``weight_factor`` (10 in the
        notebook) to sharpen the softmax; the argmax is unaffected.

    Returns ``[conv_kernel (2, 3, 1, 8), conv_bias (8,), dense_kernel (8, 256),
    dense_bias (256,)]`` in the order of ``model.set_weights``.
    """
    nbh = np.array([[(k >> 2) & 1, (k >> 1) & 1, k & 1] for k in range(8)])  # (8, 3): l, c, r
    kernels = np.zeros((8, 2, 3))
    kernels[:, 0, :] = 2 * nbh - 1
    kernels[:, 1, 1] = 1
    conv_kernel = np.transpose(kernels, (1, 2, 0))[:, :, None, :]  # (2, 3, 1, 8)
    conv_bias = -nbh.sum(axis=1).astype(float)

    bits = (np.arange(256)[None, :] >> np.arange(8)[:, None]) & 1  # (8, 256): bit j of rule i
    dense_kernel = weight_factor * (2 * bits - 1).astype(float)
    dense_bias = weight_factor * (1 - bits.sum(axis=0)).astype(float)
    return [conv_kernel, conv_bias, dense_kernel, dense_bias]


def exact_cnn(input_shape=(64, 64, 1), weight_factor: float = 10.0) -> Model:
    """The exact CNN of Sec. 3.1: reads the rule off a diagram, without training.

    It returns the true rule whenever all eight neighbourhoods occur in the
    diagram. A missing neighbourhood is read as a 0 in the rule table, so it
    fails exactly on the undetermined diagrams whose missing entry is a 1
    (:func:`eca_cnn.benchmark.table_to_rule` with ``missing_as=0`` gives the
    same answer for every diagram).
    """
    inputs = Input(input_shape, dtype=tf.float32)
    x = PeriodicPadding(1)(inputs)
    x = Conv2D(8, (2, 3), padding="valid")(x)
    x = Activation("relu")(x)
    x = GlobalMaxPooling2D()(x)
    outputs = Dense(256, activation="softmax")(x)
    model = Model(inputs, outputs, name="exact_rules")
    model.set_weights(exact_weights(weight_factor))
    model.trainable = False
    return model


def mlp(output: str = "classes", input_shape=(64, 64, 1), hidden_units: Sequence[int] = (256,)) -> Model:
    """Fully-connected baseline (``dense_model`` in the ``MLP-verification`` notebooks).

    The first hidden layer uses the He-uniform initialiser, as in the
    notebooks; the others use the Keras default.
    """
    inputs = Input(input_shape, dtype=tf.float32)
    x = Flatten()(inputs)
    for i, h in enumerate(hidden_units):
        x = Dense(h, activation="relu", **({"kernel_initializer": "he_uniform"} if i == 0 else {}))(x)
    outputs = Dense(N_OUTPUTS[output], activation="softmax")(x)
    return Model(inputs, outputs, name=f"mlp_{output}")


def input_shape_for(cfg: Config) -> Tuple[int, int, int]:
    """``(timesteps, width, 1)``, or the crop for a cropped MLP."""
    if cfg.model.crop is not None:
        return (int(cfg.model.crop[0]), int(cfg.model.crop[1]), 1)
    return (cfg.data.timesteps, cfg.data.width, 1)


def build_model(cfg: Config) -> Model:
    """The (uncompiled) network an experiment describes."""
    m = cfg.model
    shape = input_shape_for(cfg)
    kernel: Optional[Sequence[int]] = m.kernel or DEFAULT_KERNELS.get(m.architecture)
    if m.architecture == "default":
        return default_cnn(m.output, shape, kernel)
    if m.architecture == "trimmed":
        return trimmed_cnn(m.output, shape, kernel)
    if m.architecture == "exact":
        if m.output != "rules":
            raise ValueError("the exact CNN identifies rules; set model.output: rules")
        return exact_cnn(shape)
    if m.architecture == "mlp":
        return mlp(m.output, shape, m.hidden_units)
    raise ValueError(f"unknown architecture {m.architecture!r}")


def count_parameters(model: Model) -> int:
    """Number of weights and biases (the chapter's "trainable parameters")."""
    return int(sum(np.prod(w.shape) for w in model.weights))
