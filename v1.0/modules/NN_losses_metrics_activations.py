# here you define the losses, metrics, activations
# AT THE END OF THE FILE, YOU SELECT WHICH OF THESE ARE USED!!!

import numpy as np
from keras import backend as K
import tensorflow as tf
import tensorflow.experimental.numpy as tfnp
from tensorflow.python.framework.ops import EagerTensor
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from keras.activations import relu, sigmoid
from typing import Tuple, Union

from modules.NN_config import *


@tf.function
def smooth_minmax(d1: EagerTensor, d2: EagerTensor, d3: EagerTensor, alpha: float = -50) -> EagerTensor:
    # alpha > 0 for approx of maximum
    # alpha < 0 for approx of minimum

    # return (d1 * K.exp(alpha * d1) + d2 * K.exp(alpha * d2) + d3 * K.exp(alpha * d3)) / (K.exp(alpha * d1) + K.exp(
    # alpha * d2) + K.exp(alpha * d3))
    # return K.log(K.exp(alpha * d1) + K.exp(alpha * d2) + K.exp(alpha * d3)) / alpha

    min1 = (d1 + d2 - K.abs(d1 - d2)) / 2
    min2 = (d3 + min1 - K.abs(d3 - min1)) / 2

    return min2


@tf.function
def pen_OPX_v1(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for OPX; num_minerals > 0

    beta = 5  # penalisation of the forbidden regions

    w_true = y_true[:, opx_position]

    start = int(num_minerals + np.sum(subtypes[:opx_position]))
    stop = start + subtypes[opx_position]

    z_pred = y_pred[:, start:stop]

    # This is here to penalize Wo > 10 in OPX
    dist = relu(z_pred[:, -1] - 0.1) * beta
    penalisation = K.sum(dist * w_true)

    return penalisation


@tf.function
def pen_OPX_v2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for OPX; num_minerals == 0

    beta = 5  # penalisation of the forbidden regions

    start = int(num_minerals + np.sum(subtypes[:opx_position]))
    stop = start + subtypes[opx_position]

    z_pred = y_pred[:, start:stop]

    # This is here to penalize Wo > 10 in OPX
    dist = relu(z_pred[:, -1] - 0.1) * beta
    penalisation = K.sum(dist)

    return penalisation


@tf.function
def pen_CPX_v1(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for CPX; num_minerals > 0

    beta = 5  # penalisation of the forbidden regions

    w_true = y_true[:, cpx_position]

    start = int(num_minerals + np.sum(subtypes[:cpx_position]))
    stop = start + subtypes[cpx_position]

    z_pred = y_pred[:, start:stop]

    # This is here to penalize Wo > 60 in CPX
    dist = relu(z_pred[:, -1] - 0.6) * beta

    penalisation = K.sum(dist * w_true)

    return penalisation


@tf.function
def pen_CPX_v2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for CPX; num_minerals == 0

    beta = 5  # penalisation of the forbidden regions

    start = int(num_minerals + np.sum(subtypes[:cpx_position]))
    stop = start + subtypes[cpx_position]

    z_pred = y_pred[:, start:stop]

    # This is here to penalize Wo > 60 in CPX
    dist = relu(z_pred[:, -1] - 0.6) * beta
    penalisation = K.sum(dist)

    return penalisation


@tf.function
def pen_PLG_v1(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for PLG; num_minerals > 0

    beta = 5  # penalisation of the forbidden regions

    w_true = y_true[:, plg_position]

    start = int(num_minerals + np.sum(subtypes[:plg_position]))
    stop = start + subtypes[plg_position]

    z_pred = y_pred[:, start:stop]

    # This is here to penalize PLG forbidden region
    dist1 = relu(z_pred[:, use_plg_pen_idx[0]] - 0.15)
    dist2 = relu(0.5 - z_pred[:, use_plg_pen_idx[1]])
    dist3 = relu(z_pred[:, use_plg_pen_idx[2]] - 0.15)

    # closest distance from the borderline
    # dist = smooth_minmax(dist1, dist2, dist3) * beta
    dist = tfnp.min((dist1, dist2, dist3), axis=0) * beta
    penalisation = K.sum(dist * w_true)

    return penalisation


@tf.function
def pen_PLG_v2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for PLG; num_minerals == 0

    beta = 5  # penalisation of the forbidden regions

    start = int(num_minerals + np.sum(subtypes[:plg_position]))
    stop = start + subtypes[plg_position]

    z_pred = y_pred[:, start:stop]

    # This is here to penalize PLG forbidden region
    dist1 = relu(z_pred[:, use_plg_pen_idx[0]] - 0.15)
    dist2 = relu(0.5 - z_pred[:, use_plg_pen_idx[1]])
    dist3 = relu(z_pred[:, use_plg_pen_idx[2]] - 0.15)

    # closest distance from the borderline
    # dist = smooth_minmax(dist1, dist2, dist3) * beta
    dist = tfnp.min((dist1, dist2, dist3), axis=0) * beta
    penalisation = K.sum(dist)

    return penalisation


def my_loss_general(alpha: float) -> EagerTensor:
    # loss if num_minerals > 0

    @tf.function
    def loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        start, stop = 0, num_minerals

        w_true, w_pred = y_true[:, start:stop], y_pred[:, start:stop]
        w_square = K.sum(K.square(w_true - w_pred))

        wz = 0

        for k in range(len(subtypes)):
            start, stop = stop, stop + subtypes[k]
            z_true, z_pred = y_true[:, start:stop], y_pred[:, start:stop]

            z_square = K.square(z_true - z_pred)
            wz += K.sum(K.transpose(K.transpose(z_square) * w_true[:, k]))

        wz += use_opx_pen * pen_OPX_v1(y_true, y_pred)
        wz += use_cpx_pen * pen_CPX_v1(y_true, y_pred)
        wz += use_plg_pen * pen_PLG_v1(y_true, y_pred)

        return w_square + alpha * wz

    return loss


def my_loss_no_minerals() -> EagerTensor:
    # loss if num_minerals == 0

    @tf.function
    def loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        # loss in num_minerals == 0
        stop = 0
        wz = 0
    
        for k in range(len(subtypes)):
            start, stop = stop, stop + subtypes[k]
            z_true, z_pred = y_true[:, start:stop], y_pred[:, start:stop]
    
            z_square = K.square(z_true - z_pred)
            wz += K.sum(z_square)
    
        wz += use_opx_pen * pen_OPX_v2(y_true, y_pred)
        wz += use_cpx_pen * pen_CPX_v2(y_true, y_pred)
        wz += use_plg_pen * pen_PLG_v2(y_true, y_pred)
    
        return wz

    return loss


def my_loss_no_subtypes() -> EagerTensor:
    # loss if subtypes in empty

    @tf.function
    def loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        return K.sum(K.square(y_true - y_pred))

    return loss


def delete_wtrue_zero_samples(z_true_part: EagerTensor, z_pred_part: EagerTensor,
                                 w_true_part: EagerTensor) -> EagerTensor:
    # this is only needed if num_minerals > 0

    mask = tf.greater(w_true_part, 0)

    if K.any(mask):
        mask = K.reshape(mask, (tf.size(mask), 1))
        mask = tf.repeat(mask, repeats=K.shape(z_true_part)[1], axis=1)

        z_true_clean = tf.where(mask, z_true_part, tf.fill(K.shape(z_true_part), np.nan))
        z_pred_clean = tf.where(mask, z_pred_part, tf.fill(K.shape(z_pred_part), np.nan))

        """
        # fill with NaNs to get correct dimensions (after this, individual rows won't be of the same sample)
        z_true_clean = tf.boolean_mask(z_true_part, mask)
        z_pred_clean = tf.boolean_mask(z_pred_part, mask)

        # add nans to the end of the tensors
        shape = K.stack((tfnp.sum(~mask, dtype=np.int32), K.shape(z_true_clean)[1]))
        remaining_rows = tf.fill(shape, np.nan)

        z_true_clean = K.concatenate((z_true_clean, remaining_rows), axis=0)
        z_pred_clean = K.concatenate((z_pred_clean, remaining_rows), axis=0)
        """
    else:  # this is here if mask is empty due to low batch_size
        z_true_clean = tf.fill(K.shape(z_true_part), 0.0)
        z_pred_clean = tf.fill(K.shape(z_pred_part), 0.0)

    return z_true_clean, z_pred_clean


def clean_ytrue_ypred(y_true: EagerTensor, y_pred: EagerTensor, num_minerals: int,
                        all_to_one: bool = False) -> Tuple[EagerTensor, ...]:
    # If numpy, convert to tensor
    if isinstance(y_true, np.ndarray):
        y_true, y_pred = K.constant(y_true), K.constant(y_pred)

    start, stop = 0, num_minerals

    if num_minerals > 0:
        # Contribution of modal compositions
        y_true_clean, y_pred_clean = y_true[:, start:stop], y_pred[:, start:stop]

        # Do this separately for each subtype
        for k in range(len(subtypes)):
            start, stop = stop, stop + subtypes[k]

            # If the mineral is not present, we put there some values due to normalisation.
            # These are artificial and should not enter the MSE.
            z_true_clean, z_pred_clean = delete_wtrue_zero_samples(y_true[:, start:stop], y_pred[:, start:stop],
                                                                   y_true[:, k])

            y_true_clean = K.concatenate((y_true_clean, z_true_clean), axis=1)
            y_pred_clean = K.concatenate((y_pred_clean, z_pred_clean), axis=1)
    else:
        y_true_clean, y_pred_clean = y_true, y_pred

    if all_to_one:
        y_true_clean = K.reshape(y_true_clean, (tf.size(y_true_clean), 1))
        y_pred_clean = K.reshape(y_pred_clean, (tf.size(y_pred_clean), 1))

    return y_true_clean * 100, y_pred_clean * 100


def my_ae(num_minerals: int, all_to_one: bool = False):
    def ae(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        yt, yp = clean_ytrue_ypred(y_true, y_pred, num_minerals, all_to_one)
        return K.abs(yt - yp)
    return ae


def my_mae(num_minerals: int, all_to_one: bool = False):
    def mae(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        abs_error = my_ae(num_minerals, all_to_one)(y_true, y_pred)
        return tfnp.nanmean(abs_error, axis=0)
    return mae


def my_mse(num_minerals: int, all_to_one: bool = False):
    def mse(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        abs_error = my_ae(num_minerals, all_to_one)(y_true, y_pred)
        return tfnp.nanmean(K.square(abs_error), axis=0)
    return mse


def my_rmse(num_minerals: int, all_to_one: bool = False):
    def rmse(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        return K.sqrt(my_mse(num_minerals, all_to_one)(y_true, y_pred))

    return rmse


def my_Lp_norm(num_minerals: int, p_coef: float, all_to_one: bool = False):
    if p_coef < 1:
        raise ValueError('p_coef >= 1 in Lp_norm.')

    def Lp_norm(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        abs_error = my_ae(num_minerals, all_to_one)(y_true, y_pred)
        return K.pow(tfnp.nanmean(K.pow(abs_error, p_coef), axis=0), 1 / p_coef)
    return Lp_norm


def my_quantile(num_minerals: int, percentile: Union[np.ndarray, float], all_to_one: bool = False):
    def quantile(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        abs_error = my_ae(num_minerals, all_to_one)(y_true, y_pred)
        return K.constant(np.nanpercentile(abs_error, percentile, interpolation='midpoint', axis=0))
    return quantile


def my_r2(num_minerals: int, all_to_one: bool = False):
    def r2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        yt, yp = clean_ytrue_ypred(y_true, y_pred, num_minerals, all_to_one)
        abs_error = K.abs(yt - yp)

        SS_res = tfnp.nansum(K.square(abs_error), axis=0)
        SS_tot = tfnp.nansum(K.square(yt - tfnp.nanmean(yt, axis=0)), axis=0)

        return 1 - SS_res / (SS_tot + K.epsilon())
    return r2


def my_sam(num_minerals: int, all_to_one: bool = False):
    def sam(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        yt, yp = clean_ytrue_ypred(y_true, y_pred, num_minerals, all_to_one)

        s1_norm = K.sqrt(tfnp.nansum(K.square(yt), axis=0))
        s2_norm = K.sqrt(tfnp.nansum(K.square(yp), axis=0))
        sum_s1_s2 = tfnp.nansum(yt * yp, axis=0)

        return tf.math.acos(sum_s1_s2 / (s1_norm * s2_norm + K.epsilon()))
    return sam


def my_softmax(x: EagerTensor) -> EagerTensor:
    start, stop = 0, num_minerals
    x_new = K.softmax(x[..., start:stop])

    for k in range(len(subtypes)):
        start, stop = stop, stop + subtypes[k]
        tmp = K.softmax(x[..., start:stop])

        x_new = K.concatenate([x_new, tmp], axis=1)

    return x_new


def my_relu(x: EagerTensor) -> EagerTensor:
    scale = 5

    start, stop = 0, num_minerals
    x_new = (relu(x[..., start:stop]) - relu(x[..., start:stop] - scale)) / scale  # to keep results between 0 and 1
    x_new /= (K.sum(x_new, axis=1, keepdims=True) + K.epsilon())  # normalisation to unit sum

    for k in range(len(subtypes)):
        start, stop = stop, stop + subtypes[k]
        tmp = (relu(x[..., start:stop]) - relu(x[..., start:stop] - scale)) / scale  # to keep results between 0 and 1
        tmp /= (K.sum(tmp, axis=1, keepdims=True) + K.epsilon())  # normalisation to unit sum

        x_new = K.concatenate([x_new, tmp], axis=1)

    return x_new


def my_sigmoid(x: EagerTensor) -> EagerTensor:
    start, stop = 0, num_minerals
    x_new = sigmoid(x[..., start:stop])
    x_new /= (K.sum(x_new, axis=1, keepdims=True) + K.epsilon())  # normalisation to unit sum

    for k in range(len(subtypes)):
        start, stop = stop, stop + subtypes[k]
        tmp = sigmoid(x[..., start:stop])
        tmp /= (K.sum(tmp, axis=1, keepdims=True) + K.epsilon())  # normalisation to unit sum

        x_new = K.concatenate([x_new, tmp], axis=1)

    return x_new


def my_plu(x: EagerTensor) -> EagerTensor:
    # https://arxiv.org/pdf/1809.09534.pdf

    alpha, c = 0.1, 1.0

    start, stop = 0, num_minerals
    x_new = relu(x[..., start:stop] + c) - c - (1 - alpha) * relu(x[..., start:stop] - c) - alpha * relu(
        -x[..., start:stop] - c)
    x_new /= (K.sum(x_new, axis=1, keepdims=True) + K.epsilon())  # normalisation to unit sum

    for k in range(len(subtypes)):
        start, stop = stop, stop + subtypes[k]
        tmp = relu(x[..., start:stop] + c) - c - (1 - alpha) * relu(x[..., start:stop] - c) - alpha * relu(
            -x[..., start:stop] - c)
        tmp /= (K.sum(tmp, axis=1, keepdims=True) + K.epsilon())  # normalisation to unit sum

        x_new = K.concatenate([x_new, tmp], axis=1)

    return x_new


def my_softmax_chem(x: EagerTensor) -> EagerTensor:
    start, stop = 0, 0
    x_new = K.softmax(x[..., start:stop])

    for k in range(len(subtypes)):
        start, stop = stop, stop + subtypes[k]
        tmp = K.softmax(x[..., start:stop])

        x_new = K.concatenate([x_new, tmp], axis=1)

    return x_new


# losses
if num_minerals > 0 and subtypes:
    loss_name, loss = 'loss', my_loss_general(p['alpha'])
    loss_Bayes = my_loss_general    # Bayes enters 'alpha' HP during compilation
elif num_minerals == 0:
    loss_name, loss = 'loss', my_loss_no_minerals()
elif not subtypes:
    loss_name, loss = 'loss', my_loss_no_subtypes()

# metrics
# mse is main metrics, and it should be minimised for all data at once (otherwise mean(mse_individual) is minimised)
mse_name, mse = 'mse', my_mse(num_minerals, True)
rmse_name, rmse = 'rmse', my_rmse(num_minerals)
quantile_name, quantile = 'quantile', my_quantile(num_minerals, 50.0)
mae_name, mae = 'mae', my_mae(num_minerals)
Lp_norm_name, Lp_norm = 'Lp_norm', my_Lp_norm(num_minerals, 1.5)
r2_name, r2 = 'r2', my_r2(num_minerals)
sam_name, sam = 'sam', my_sam(num_minerals)

# important for hp tuning and early stopping
# must be included in NN_models.py metrics variable and in custom_objects in NN_evaluate.py
main_acc_name, main_acc, direction = 'val_mse', mse, 'min'  # minimise or maximise the main_acc metric?

# activations
if p['output_activation'] == 'softmax':
    output_activation_name, output_activation = 'my_softmax', my_softmax
elif p['output_activation'] == 'relu':
    output_activation_name, output_activation = 'my_relu', my_relu
elif p['output_activation'] == 'sigmoid':
    output_activation_name, output_activation = 'my_sigmoid', my_sigmoid
elif p['model_type'] == 'two_thread':
    output_activation_name, output_activation = 'my_softmax_chem', my_softmax_chem
else:
    raise NameError("p['output_activation'] must be one of 'relu', 'sigmoid', 'softmax'")
