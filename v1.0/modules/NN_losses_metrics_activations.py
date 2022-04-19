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

    start = int(num_minerals + np.sum(subtypes_pen[:opx_position]))
    stop = start + subtypes_pen[opx_position]

    z_pred = y_pred[:, start:stop]

    # This is here to penalize Wo > 10 in OPX
    dist = relu(z_pred[:, -1] - 0.1) * beta
    penalisation = K.sum(dist * w_true)

    return penalisation


@tf.function
def pen_OPX_v2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for OPX; num_minerals == 0

    beta = 5  # penalisation of the forbidden regions

    start = int(num_minerals + np.sum(subtypes_pen[:opx_position]))
    stop = start + subtypes_pen[opx_position]

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

    start = int(num_minerals + np.sum(subtypes_pen[:cpx_position]))
    stop = start + subtypes_pen[cpx_position]

    z_pred = y_pred[:, start:stop]

    # This is here to penalize Wo > 60 in CPX
    dist = relu(z_pred[:, -1] - 0.6) * beta

    penalisation = K.sum(dist * w_true)

    return penalisation


@tf.function
def pen_CPX_v2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for CPX; num_minerals == 0

    beta = 5  # penalisation of the forbidden regions

    start = int(num_minerals + np.sum(subtypes_pen[:cpx_position]))
    stop = start + subtypes_pen[cpx_position]

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

    start = int(num_minerals + np.sum(subtypes_pen[:plg_position]))
    stop = start + subtypes_pen[plg_position]

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

    start = int(num_minerals + np.sum(subtypes_pen[:plg_position]))
    stop = start + subtypes_pen[plg_position]

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


def my_loss(alpha: float) -> EagerTensor:
    # loss in num_minerals > 0

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


def delete_wtrue_zero_samples(z_true_part: EagerTensor, z_pred_part: EagerTensor,
                              w_true_part: EagerTensor) -> EagerTensor:
    # this is only needed if num_minerals > 0
    mask = tf.greater(w_true_part, 0)
    if K.any(mask):
        z_true_clean = tf.boolean_mask(z_true_part, mask)
        z_pred_clean = tf.boolean_mask(z_pred_part, mask)
    else:  # this is here if mask is empty due to low batch_size
        z_true_clean = tf.fill(K.shape(z_true_part), 0.0)
        z_pred_clean = tf.fill(K.shape(z_pred_part), 0.0)

    return z_true_clean, z_pred_clean


def my_mse(num_minerals: int):
    def mse(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        # If numpy, convert to tensor
        if isinstance(y_true, np.ndarray):
            y_true, y_pred = K.constant(y_true), K.constant(y_pred)

        start, stop = 0, num_minerals

        if num_minerals > 0:
            # Contribution of modal compositions
            w_true, w_pred = y_true[:, start:stop], y_pred[:, start:stop]
            res = tfnp.nanmean(K.square(w_true - w_pred), axis=0)

            # Do this separately for each subtype
            for k in range(len(subtypes)):
                start, stop = stop, stop + subtypes[k]

                # If the mineral is not present, we put there some values due to normalisation.
                # These are artificial and should not enter the MSE.
                z_true, z_pred = delete_wtrue_zero_samples(y_true[:, start:stop], y_pred[:, start:stop], w_true[:, k])

                res = K.concatenate((res, tfnp.nanmean(K.square(z_true - z_pred), axis=0)))
        else:
            res = tfnp.nanmean(K.square(y_true - y_pred), axis=0)

        return res * 100 * 100

    return mse


def my_rmse(num_minerals: int):
    def rmse(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        return K.sqrt(my_mse(num_minerals)(y_true, y_pred))

    return rmse


def my_mae(num_minerals: int):
    def mae(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        # If numpy, convert to tensor
        if isinstance(y_true, np.ndarray):
            y_true, y_pred = K.constant(y_true), K.constant(y_pred)

        start, stop = 0, num_minerals

        if num_minerals > 0:
            # Contribution of modal compositions
            w_true, w_pred = y_true[:, start:stop], y_pred[:, start:stop]
            res = tfnp.nanmean(K.abs(w_true - w_pred), axis=0)

            # Do this separately for each subtype
            for k in range(len(subtypes)):
                start, stop = stop, stop + subtypes[k]

                # If the mineral is not present, we put there some values due to normalisation.
                # These are artificial and should not enter the RMSE.
                z_true, z_pred = delete_wtrue_zero_samples(y_true[:, start:stop], y_pred[:, start:stop], w_true[:, k])

                res = K.concatenate((res, tfnp.nanmean(K.abs(z_true - z_pred), axis=0)))
        else:
            res = tfnp.nanmean(K.abs(y_true - y_pred), axis=0)

        return res * 100

    return mae


def my_Lp_norm(num_minerals: int, p_coef: float):
    if p_coef < 1:
        raise ValueError('p_coef >= 1 in Lp_norm.')

    def Lp_norm(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        # If numpy, convert to tensor
        if isinstance(y_true, np.ndarray):
            y_true, y_pred = K.constant(y_true), K.constant(y_pred)

        start, stop = 0, num_minerals

        if num_minerals > 0:
            # Contribution of modal compositions
            w_true, w_pred = y_true[:, start:stop], y_pred[:, start:stop]
            res = tfnp.nanmean(K.pow(K.abs(w_true - w_pred), p_coef), axis=0)

            # Do this separately for each subtype
            for k in range(len(subtypes)):
                start, stop = stop, stop + subtypes[k]

                # If the mineral is not present, we put there some values due to normalisation.
                # These are artificial and should not enter the Lp norm.
                z_true, z_pred = delete_wtrue_zero_samples(y_true[:, start:stop], y_pred[:, start:stop], w_true[:, k])

                res = K.concatenate((res, tfnp.nanmean(K.pow(K.abs(z_true - z_pred), p_coef), axis=0)))
        else:
            res = tfnp.nanmean(K.pow(K.abs(y_true - y_pred), p_coef), axis=0)

        return K.pow(res, 1 / p_coef) * 100

    return Lp_norm


def my_quantile(num_minerals: int, percentile: float):
    def quantile(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        # If numpy, convert to tensor
        if isinstance(y_true, np.ndarray):
            y_true, y_pred = K.constant(y_true), K.constant(y_pred)

        start, stop = 0, num_minerals

        # y_true, y_pred should not contain NaNs. Use numpy.nanmean() in such a case

        if num_minerals > 0:
            # Contribution of modal compositions
            w_true, w_pred = y_true[:, start:stop], y_pred[:, start:stop]
            res = tfp.stats.percentile(K.abs(w_true - w_pred), percentile, interpolation='midpoint', axis=0)

            # Do this separately for each subtype
            for k in range(len(subtypes)):
                start, stop = stop, stop + subtypes[k]

                # If the mineral is not present, we put there some values due to normalisation.
                # These are artificial and should not enter the RMSE.
                z_true, z_pred = delete_wtrue_zero_samples(y_true[:, start:stop], y_pred[:, start:stop], w_true[:, k])

                res = K.concatenate((res, tfp.stats.percentile(K.abs(z_true - z_pred), percentile,
                                                               interpolation='midpoint', axis=0)))
        else:
            res = tfp.stats.percentile(K.abs(y_true - y_pred), percentile, interpolation='midpoint', axis=0)

        return res * 100

    return quantile


def my_r2(num_minerals: int):
    def r2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        # If numpy, convert to tensor
        if isinstance(y_true, np.ndarray):
            y_true, y_pred = K.constant(y_true), K.constant(y_pred)

        start, stop = 0, num_minerals

        if num_minerals > 0:
            # Contribution of modal compositions
            w_true, w_pred = y_true[:, start:stop], y_pred[:, start:stop]

            SS_res = K.sum(K.square(w_true - w_pred), axis=0)
            SS_tot = K.sum(K.square(w_true - K.mean(w_true, axis=0)), axis=0)

            res = 1 - SS_res / (SS_tot + K.epsilon())

            # Do this separately for each subtype
            for k in range(len(subtypes)):
                start, stop = stop, stop + subtypes[k]

                # If the mineral is not present, we put there some values due to normalisation.
                # These are artificial and should not enter the R2.
                z_true, z_pred = delete_wtrue_zero_samples(y_true[:, start:stop], y_pred[:, start:stop], w_true[:, k])

                SS_res = K.sum(K.square(z_true - z_pred), axis=0)
                SS_tot = K.sum(K.square(z_true - K.mean(z_true, axis=0)), axis=0)

                res = K.concatenate((res, 1 - SS_res / (SS_tot + K.epsilon())))
        else:
            SS_res = K.sum(K.square(y_true - y_pred), axis=0)
            SS_tot = K.sum(K.square(y_true - K.mean(y_true, axis=0)), axis=0)

            res = 1 - SS_res / (SS_tot + K.epsilon())

        return res

    return r2


def my_sam(num_minerals: int):
    def sam(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        # If numpy, convert to tensor
        if isinstance(y_true, np.ndarray):
            y_true, y_pred = K.constant(y_true), K.constant(y_pred)

        start, stop = 0, num_minerals

        if num_minerals > 0:
            # Contribution of modal compositions
            w_true, w_pred = y_true[:, start:stop], y_pred[:, start:stop]

            s1_norm = K.sqrt(K.sum(w_true * w_true, axis=0))
            s2_norm = K.sqrt(K.sum(w_pred * w_pred, axis=0))
            sum_s1_s2 = K.sum(w_true * w_pred, axis=0)

            res = tf.math.acos(sum_s1_s2 / (s1_norm * s2_norm + K.epsilon()))

            # Do this separately for each subtype
            for k in range(len(subtypes)):
                start, stop = stop, stop + subtypes[k]

                # If the mineral is not present, we put there some values due to normalisation.
                # These are artificial and should not enter the SAM.
                z_true, z_pred = delete_wtrue_zero_samples(y_true[:, start:stop], y_pred[:, start:stop], w_true[:, k])

                s1_norm = K.sqrt(K.sum(z_true * z_true, axis=0))
                s2_norm = K.sqrt(K.sum(z_pred * z_pred, axis=0))
                sum_s1_s2 = K.sum(z_true * z_pred, axis=0)

                res = K.concatenate((res, tf.math.acos(sum_s1_s2 / (s1_norm * s2_norm + K.epsilon()))))
        else:
            s1_norm = K.sqrt(K.sum(y_true * y_true, axis=0))
            s2_norm = K.sqrt(K.sum(y_pred * y_pred, axis=0))
            sum_s1_s2 = K.sum(y_true * y_pred, axis=0)

            res = tf.math.acos(sum_s1_s2 / (s1_norm * s2_norm + K.epsilon()))  # in radians

        return res

    return sam


def my_sam_all(num_minerals: int):
    # SAM computed for the whole vector
    def sam(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        # If numpy, convert to tensor
        if isinstance(y_true, np.ndarray):
            y_true, y_pred = K.constant(y_true), K.constant(y_pred)

        start, stop = 0, num_minerals

        if num_minerals > 0:
            # filter out where w_true = 0
            w_true, w_pred = y_true[:, start:stop], y_pred[:, start:stop]

            # vectorise it
            vec_true, vec_pred = K.reshape(w_true, (tf.size(w_true), 1)), K.reshape(w_pred, (tf.size(w_pred), 1))

            # Do this separately for each subtype
            for k in range(len(subtypes)):
                start, stop = stop, stop + subtypes[k]

                # If the mineral is not present, we put there some values due to normalisation.
                # These are artificial and should not enter the SAM.
                z_true, z_pred = delete_wtrue_zero_samples(y_true[:, start:stop], y_pred[:, start:stop], w_true[:, k])

                # vectorise it
                tmp_true, tmp_pred = K.reshape(z_true, (tf.size(z_true), 1)), K.reshape(z_pred, (tf.size(z_pred), 1))

                vec_true = K.concatenate((vec_true, tmp_true), axis=0)
                vec_pred = K.concatenate((vec_pred, tmp_pred), axis=0)
        else:
            vec_true, vec_pred = y_true, y_pred

        s1_norm = K.sqrt(K.sum(vec_true * vec_true))
        s2_norm = K.sqrt(K.sum(vec_pred * vec_pred))
        sum_s1_s2 = K.sum(vec_true * vec_pred)

        res = tf.math.acos(sum_s1_s2 / (s1_norm * s2_norm + K.epsilon()))  # in radians

        return res

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
if num_minerals > 0:
    loss_name, loss = 'loss', my_loss(p['alpha'])
else:
    loss_name, loss = 'loss', loss

# metrics
mse_name, mse = 'mse', my_mse(num_minerals)
rmse_name, rmse = 'rmse', my_rmse(num_minerals)
quantile_name, quantile = 'quantile', my_quantile(num_minerals, 50.0)
mae_name, mae = 'mae', my_mae(num_minerals)
Lp_norm_name, Lp_norm = 'Lp_norm', my_Lp_norm(num_minerals, 1.5)
r2_name, r2 = 'r2', my_r2(num_minerals)
sam_name, sam = 'sam', my_sam(num_minerals)

# important for hp tuning and early stopping
# must be included in NN_models.py metrics variable and in custom_objects in NN_evaluate.py
main_acc_name = 'val_mse'

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
