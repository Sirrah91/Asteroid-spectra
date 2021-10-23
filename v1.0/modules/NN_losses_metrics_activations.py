import numpy as np
from keras import backend as K
import tensorflow as tf
from tensorflow.experimental.numpy import heaviside
from tensorflow.python.framework.ops import EagerTensor
from tensorflow_probability import distributions as tfd

from modules.NN_config import *


# @tf.autograph.experimental.do_not_convert
def pen_PLG(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for PLG

    beta = 5  # penalisation of the forbidden regions

    w_true = y_true[:, plg_position]

    start = num_minerals + np.sum(subtypes[:plg_position])
    stop = start + subtypes[plg_position]

    z_pred = y_pred[:, start:stop]

    # This is here to penalize PLG forbidden region
    dist1, dist2, dist3 = z_pred[:, 0] - 0.1, 0.5 - z_pred[:, 1], z_pred[:, 2] - 0.1
    # Forbidden region -> all distances must be positive
    # dist1 *= heaviside(dist1, 0)
    # dist2 *= heaviside(dist2, 1)
    # dist3 *= heaviside(dist3, 0)

    # removing NaNs (K.pow return NaN is number is negative); should not be there just in case
    # dist1 = tf.where(tf.math.is_nan(dist1), tf.zeros_like(dist1), dist1)
    # dist2 = tf.where(tf.math.is_nan(dist2), tf.zeros_like(dist2), dist2)
    # dist3 = tf.where(tf.math.is_nan(dist3), tf.zeros_like(dist3), dist3)

    # scaling
    # dist = tf.math.pow(dist1 * dist2 * dist3, 1.0 / 3.0)
    # Forbidden region -> all distances must be positive
    dist = K.sqrt(dist1**2 + dist2**2 + dist3**3) * heaviside(dist1, 0) * heaviside(dist2, 1) * heaviside(dist3, 0)

    penalisation = K.sum(dist * w_true * beta)

    return penalisation


# @tf.autograph.experimental.do_not_convert
def pen_CPX(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for CPX

    beta = 5  # penalisation of the forbidden regions

    w_true = y_true[:, cpx_position]

    start = num_minerals + np.sum(subtypes[:cpx_position])
    stop = start + subtypes[cpx_position]

    z_pred = y_pred[:, start:stop]

    # This is here to penalize Wo > 60 in CPX
    dist = z_pred[:, 2] - 0.6
    # Forbidden region -> distance must be positive
    dist *= heaviside(dist, 0)

    penalisation = K.sum(dist * w_true * beta)

    return penalisation


def my_rmse(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # If numpy, convert to tensor
    if isinstance(y_true, np.ndarray):
        y_true, y_pred = K.constant(y_true), K.constant(y_pred)
    diff = y_true - y_pred

    return K.sqrt(tf.experimental.numpy.nanmean(K.pow(diff, 2), axis=0)) * 100


def my_r2_v1(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This function is used as metrics if num_minerals > 0

    # If numpy, convert to tensor
    if isinstance(y_true, np.ndarray):
        y_true, y_pred = K.constant(y_true), K.constant(y_pred)

    nspectra = K.shape(y_true)[0]

    start, stop = 0, num_minerals

    # Contribution of modal compositions
    w_true, w_pred = y_true[:, start:stop], y_pred[:, start:stop]

    SS_res = K.sum(K.square(w_true - w_pred))
    SS_tot = K.sum(K.square(w_true - K.mean(w_true, axis=0)))

    # Do this separately for each subtype
    for k in range(len(subtypes)):
        start, stop = stop, stop + subtypes[k]
        z_true, z_pred = y_true[:, start:stop], y_pred[:, start:stop]

        SS_res += K.sum(K.square(z_true - z_pred) * K.reshape(w_true[:, k], (nspectra, 1)))
        SS_tot += K.sum(K.square(z_true - K.mean(z_true, axis=0)))

    return 1 - SS_res / (SS_tot + K.epsilon())


def my_r2_v2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This function is used as metrics if num_minerals == 0

    # If numpy, convert to tensor
    if isinstance(y_true, np.ndarray):
        y_true, y_pred = K.constant(y_true), K.constant(y_pred)

    stop = 0
    SS_res, SS_tot = 0, 0

    # Do this separately for each subtype
    for k in range(len(subtypes)):
        start, stop = stop, stop + subtypes[k]
        z_true, z_pred = y_true[:, start:stop], y_pred[:, start:stop]

        SS_res += K.sum(K.square(z_true - z_pred))
        SS_tot += K.sum(K.square(z_true - K.mean(z_true, axis=0)))

    return 1 - SS_res / (SS_tot + K.epsilon())


# @tf.autograph.experimental.do_not_convert
def my_loss_v1(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a used as a loss function if num_minerals > 0

    start, stop = 0, num_minerals

    w_true, w_pred = y_true[:, start:stop], y_pred[:, start:stop]
    w_square = K.sum(K.square(w_true - w_pred))

    wz = 0

    for k in range(len(subtypes)):
        start, stop = stop, stop + subtypes[k]
        z_true, z_pred = y_true[:, start:stop], y_pred[:, start:stop]

        z_square = K.square(z_true - z_pred)
        wz += K.sum(K.transpose(K.transpose(z_square) * w_true[:, k]))

    wz += pen_CPX(y_true, y_pred)
    # wz += pen_PLG(y_true, y_pred)

    return w_square + alpha * wz


# @tf.autograph.experimental.do_not_convert
def my_loss_v2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a used as a loss function if num_minerals == 0

    stop = num_minerals
    wz = 0

    for k in range(len(subtypes)):
        start, stop = stop, stop + subtypes[k]
        z_true, z_pred = y_true[:, start:stop], y_pred[:, start:stop]

        wz += K.sum(K.square(z_true - z_pred))

    # wz += pen_CPX(y_true, y_pred)
    # wz += pen_PLG(y_true, y_pred)

    return wz


def my_softmax(x: EagerTensor) -> EagerTensor:
    # This is softmax by parts

    # define heaviside for threshold
    # heaviside

    start, stop = 0, num_minerals
    x_new = K.softmax(x[..., start:stop])

    # use heaviside to make a threshold?
    # heaviside

    # renormalise the results again
    # renormalisation

    for k in range(len(subtypes)):
        start, stop = stop, stop + subtypes[k]
        tmp = K.softmax(x[..., start:stop])

        # use heaviside to make a threshold?
        # heaviside

        # renormalise the results again
        # renormalisation

        x_new = K.concatenate([x_new, tmp], axis=1)

    return x_new
