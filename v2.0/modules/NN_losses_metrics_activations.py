# AT THE END OF THE FILE, YOU SELECT WHICH OBJECT IS USED IN CALLBACKS

import numpy as np
from keras import backend as K
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tensorflow.python.framework.ops import EagerTensor
from keras.activations import relu, sigmoid
from typing import Callable

from modules.utilities_spectra import gimme_indices
from modules.NN_config import num_minerals, endmembers_counts, p, endmembers_used, minerals_all


def gimme_penalisation_setup(penalised_mineral: str, used_endmembers: list[list[bool]], all_minerals: np.ndarray,
                             counts_endmembers: np.ndarray) -> tuple[int, float, int | tuple[int, ...], int, int]:

    all_minerals = tf.dtypes.cast(all_minerals, dtype=tf.int32)

    # these are needed due to "abandon" regions
    if penalised_mineral == "orthopyroxene":
        i, j = 1, 2  # OPX, Wo (indices in endmembers_used)
        if used_endmembers[i][j]:
            start = num_minerals + K.cast(K.sum(counts_endmembers[:i]), dtype="int32")
            stop = start + K.cast(counts_endmembers[i], dtype="int32")
            position, use_pen, use_pen_idx = K.sum(all_minerals[:i]), 1., -1
        else:
            position, use_pen, use_pen_idx, start, stop = 0, 0., 0, 0, 1  # use first index which is always present

        return position, use_pen, use_pen_idx, start, stop

    if penalised_mineral == "clinopyroxene":
        i, j = 2, 2  # CPX, Wo (indices in endmembers_used)
        # these are needed due to "abandon" regions
        if used_endmembers[i][j]:
            start = num_minerals + K.cast(K.sum(counts_endmembers[:i]), dtype="int32")
            stop = start + K.cast(counts_endmembers[i], dtype="int32")
            position, use_pen, use_pen_idx = K.sum(all_minerals[:i]), 1., -1
        else:
            position, use_pen, use_pen_idx, start, stop = 0, 0., 0, 0, 1  # use first index which is always present

        return position, use_pen, use_pen_idx, start, stop

    if penalised_mineral == "plagioclase":
        i = 3  # PLG
        if np.array(endmembers_used[i]).all():
            start = num_minerals + K.cast(K.sum(counts_endmembers[:i]), dtype="int32")
            stop = start + K.cast(counts_endmembers[i], dtype="int32")
            position, use_pen, use_pen_idx = K.sum(all_minerals[:i]), 1., (0, 1, 2)
        else:
            # use first index which is always present
            position, use_pen, use_pen_idx, start, stop = 0, 0., (0, 0, 0), 0, 1

        return position, use_pen, use_pen_idx, start, stop


@tf.function
def pen_OPX_v1(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for OPX; num_minerals > 0

    beta = 5.0  # penalisation of the forbidden regions
    OPX_Wo_limit = 0.10  # 10%

    position, use_penalisation, index, start, stop = gimme_penalisation_setup("orthopyroxene", endmembers_used,
                                                                              minerals_all, endmembers_counts)

    w_true = y_true[:, position]
    z_pred = y_pred[:, start:stop]

    # This is here to penalize Wo in OPX
    dist = relu(z_pred[:, index] - OPX_Wo_limit) * beta
    penalisation = K.sum(dist * w_true)

    return penalisation * use_penalisation


@tf.function
def pen_OPX_v2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for OPX; num_minerals == 0

    beta = 5.0  # penalisation of the forbidden regions
    OPX_Wo_limit = 0.10  # 10%

    position, use_penalisation, index, start, stop = gimme_penalisation_setup("orthopyroxene", endmembers_used,
                                                                              minerals_all, endmembers_counts)

    z_pred = y_pred[:, start:stop]

    # This is here to penalize Wo in OPX
    dist = relu(z_pred[:, index] - OPX_Wo_limit) * beta
    penalisation = K.sum(dist)

    return penalisation * use_penalisation


@tf.function
def pen_CPX_v1(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for CPX; num_minerals > 0

    beta = 5.0  # penalisation of the forbidden regions
    CPX_Wo_limit = 0.60  # 60%

    position, use_penalisation, index, start, stop = gimme_penalisation_setup("clinopyroxene", endmembers_used,
                                                                              minerals_all, endmembers_counts)

    w_true = y_true[:, position]
    z_pred = y_pred[:, start:stop]

    # This is here to penalize Wo in CPX
    dist = relu(z_pred[:, index] - CPX_Wo_limit) * beta
    penalisation = K.sum(dist * w_true)

    return penalisation * use_penalisation


@tf.function
def pen_CPX_v2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for CPX; num_minerals == 0

    beta = 5.0  # penalisation of the forbidden regions
    CPX_Wo_limit = 0.60  # 60%

    position, use_penalisation, index, start, stop = gimme_penalisation_setup("clinopyroxene", endmembers_used,
                                                                              minerals_all, endmembers_counts)

    z_pred = y_pred[:, start:stop]

    # This is here to penalize Wo in CPX
    dist = relu(z_pred[:, index] - CPX_Wo_limit) * beta
    penalisation = K.sum(dist)

    return penalisation * use_penalisation


@tf.function
def pen_PLG_v1(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for PLG; num_minerals > 0

    beta = 5.0  # penalisation of the forbidden regions
    # this roughly delimits the forbidden region
    PLG_An_limit = 0.15  # 15%
    PLG_Ab_limit = 0.50  # 50%
    PLG_Or_limit = 0.15  # 15%

    position, use_penalisation, index, start, stop = gimme_penalisation_setup("plagioclase", endmembers_used,
                                                                              minerals_all, endmembers_counts)

    w_true = y_true[:, position]
    z_pred = y_pred[:, start:stop]

    # This is here to penalize PLG forbidden region
    dist1 = relu(z_pred[:, index[0]] - PLG_An_limit)
    dist2 = relu(PLG_Ab_limit - z_pred[:, index[1]])
    dist3 = relu(z_pred[:, index[2]] - PLG_Or_limit)

    # closest distance from the borderline
    dist = tnp.min((dist1, dist2, dist3), axis=0) * beta
    penalisation = K.sum(dist * w_true)

    return penalisation * use_penalisation


@tf.function
def pen_PLG_v2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
    # This is a penalisation function for PLG; num_minerals == 0

    beta = 5.0  # penalisation of the forbidden regions
    # this roughly delimits the forbidden region
    PLG_An_limit = 0.15  # 15%
    PLG_Ab_limit = 0.50  # 50%
    PLG_Or_limit = 0.15  # 15%

    position, use_penalisation, index, start, stop = gimme_penalisation_setup("plagioclase", endmembers_used,
                                                                              minerals_all, endmembers_counts)

    z_pred = y_pred[:, start:stop]

    # This is here to penalize PLG forbidden region
    dist1 = relu(z_pred[:, index[0]] - PLG_An_limit)
    dist2 = relu(PLG_Ab_limit - z_pred[:, index[1]])
    dist3 = relu(z_pred[:, index[2]] - PLG_Or_limit)

    # closest distance from the borderline
    dist = tnp.min((dist1, dist2, dist3), axis=0) * beta
    penalisation = K.sum(dist)

    return penalisation * use_penalisation


def my_loss_general(alpha: float) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    # loss if num_minerals > 0
    @tf.function
    def loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        indices = gimme_indices(num_minerals, endmembers_counts, return_mineral_indices=True)
        start, stop = indices[0, 1:]

        w_true, w_pred = y_true[:, start:stop], y_pred[:, start:stop]
        w_square = K.sum(K.square(w_true - w_pred))

        wz = 0.0

        for i, start, stop in indices[1:]:
            z_true, z_pred = y_true[:, start:stop], y_pred[:, start:stop]

            z_square = K.square(z_true - z_pred)
            wz += K.sum(K.transpose(K.transpose(z_square) * w_true[:, i]))

        wz += pen_OPX_v1(y_true, y_pred)
        wz += pen_CPX_v1(y_true, y_pred)
        wz += pen_PLG_v1(y_true, y_pred)

        return w_square + alpha * wz

    return loss


def my_loss_no_minerals() -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    # loss if num_minerals == 0
    @tf.function
    def loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        # loss in num_minerals == 0
        wz = 0.0

        for start, stop in gimme_indices(num_minerals, endmembers_counts):
            z_true, z_pred = y_true[:, start:stop], y_pred[:, start:stop]

            z_square = K.square(z_true - z_pred)
            wz += K.sum(z_square)

        wz += pen_OPX_v2(y_true, y_pred)
        wz += pen_CPX_v2(y_true, y_pred)
        wz += pen_PLG_v2(y_true, y_pred)

        return wz

    return loss


def my_loss_no_endmembers() -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    # loss if end-members are empty
    @tf.function
    def loss(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        return K.sum(K.square(y_true - y_pred))

    return loss


# @tf.function
def delete_wtrue_zero_samples(z_true_part: EagerTensor, z_pred_part: EagerTensor,
                              w_true_part: EagerTensor) -> tuple[EagerTensor, ...]:
    # this is only needed if num_minerals > 0

    mask = tf.greater(w_true_part, 0)

    if K.any(mask):
        mask = K.reshape(mask, (tf.size(mask), 1))
        mask = tf.repeat(mask, repeats=K.shape(z_true_part)[1], axis=1)

        z_true_clean = tf.where(mask, z_true_part, tf.fill(K.shape(z_true_part), np.nan))
        z_pred_clean = tf.where(mask, z_pred_part, tf.fill(K.shape(z_pred_part), np.nan))
    else:  # this is here if mask is empty due to low batch_size
        z_true_clean = tf.fill(K.shape(z_true_part), 0.0)
        z_pred_clean = tf.fill(K.shape(z_pred_part), 0.0)

    return z_true_clean, z_pred_clean


def clean_ytrue_ypred(y_true: EagerTensor, y_pred: EagerTensor, num_minerals: int,
                      all_to_one: bool = False) -> tuple[EagerTensor, ...]:
    # If numpy, convert to tensor
    if isinstance(y_true, np.ndarray):
        y_true, y_pred = K.constant(y_true), K.constant(y_pred)

    if num_minerals > 0:
        for i, start, stop in gimme_indices(num_minerals, endmembers_counts, return_mineral_indices=True):
            if i < 0:  # Contribution of modal compositions
                y_true_clean, y_pred_clean = y_true[:, start:stop], y_pred[:, start:stop]
            else:  # Contribution of chemical compositions
                # If the mineral is not present, we put there some values due to normalisation.
                # These are artificial and should not enter the MSE.
                z_true_clean, z_pred_clean = delete_wtrue_zero_samples(y_true[:, start:stop], y_pred[:, start:stop],
                                                                       y_true[:, i])

                y_true_clean = K.concatenate((y_true_clean, z_true_clean), axis=1)
                y_pred_clean = K.concatenate((y_pred_clean, z_pred_clean), axis=1)
    else:
        y_true_clean, y_pred_clean = y_true, y_pred

    if all_to_one:
        y_true_clean = K.reshape(y_true_clean, (tf.size(y_true_clean), 1))
        y_pred_clean = K.reshape(y_pred_clean, (tf.size(y_pred_clean), 1))

    return y_true_clean * 100., y_pred_clean * 100.


def my_ae(num_minerals: int, all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    # @tf.function
    def ae(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        yt, yp = clean_ytrue_ypred(y_true, y_pred, num_minerals, all_to_one)
        return K.abs(yt - yp)

    return ae


def my_mae(num_minerals: int, all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    # @tf.function
    def mae(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        abs_error = my_ae(num_minerals, all_to_one)(y_true, y_pred)
        return tnp.nanmean(abs_error, axis=0)

    return mae


def my_mse(num_minerals: int, all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    # @tf.function
    def mse(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        abs_error = my_ae(num_minerals, all_to_one)(y_true, y_pred)
        return tnp.nanmean(K.square(abs_error), axis=0)

    return mse


def my_sse(num_minerals: int, all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    # @tf.function
    def sse(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        abs_error = my_ae(num_minerals, all_to_one)(y_true, y_pred)
        return tnp.nansum(K.square(abs_error), axis=0)

    return sse


def my_rmse(num_minerals: int, all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    # @tf.function
    def rmse(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        return K.sqrt(my_mse(num_minerals, all_to_one)(y_true, y_pred))

    return rmse


def my_Lp_norm(num_minerals: int, p_coef: float, all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor],
                                                                                       EagerTensor]:
    if p_coef < 1:
        raise ValueError("p_coef >= 1 in Lp_norm.")

    # @tf.function
    def Lp_norm(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        abs_error = my_ae(num_minerals, all_to_one)(y_true, y_pred)
        return K.pow(tnp.nansum(K.pow(abs_error, p_coef), axis=0), 1. / p_coef)

    return Lp_norm


def my_quantile(num_minerals: int, percentile: np.ndarray | float,
                all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    # @tf.function
    def quantile(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        abs_error = my_ae(num_minerals, all_to_one)(y_true, y_pred)\
        # to avoid issues with np.nanpercentile when passing in model.fit
        return tf.numpy_function(lambda error, perc:
                                 np.nanpercentile(error, perc, method="midpoint", axis=0).astype(np.float32),
                                 inp=[abs_error, percentile], Tout=tf.float32)
        # return K.constant(np.nanpercentile(abs_error, percentile, method="midpoint", axis=0))

    return quantile


def my_r2(num_minerals: int, all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    # @tf.function
    def r2(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        yt, yp = clean_ytrue_ypred(y_true, y_pred, num_minerals, all_to_one)
        abs_error = K.abs(yt - yp)

        SS_res = tnp.nansum(K.square(abs_error), axis=0)
        SS_tot = tnp.nansum(K.square(yt - tnp.nanmean(yt, axis=0)), axis=0)

        return 1.0 - SS_res / (SS_tot + K.epsilon())

    return r2


def my_sam(num_minerals: int, all_to_one: bool = False) -> Callable[[EagerTensor, EagerTensor], EagerTensor]:
    # @tf.function
    def sam(y_true: EagerTensor, y_pred: EagerTensor) -> EagerTensor:
        yt, yp = clean_ytrue_ypred(y_true, y_pred, num_minerals, all_to_one)

        s1_norm = K.sqrt(tnp.nansum(K.square(yt), axis=0))
        s2_norm = K.sqrt(tnp.nansum(K.square(yp), axis=0))
        sum_s1_s2 = tnp.nansum(yt * yp, axis=0)

        return tf.math.acos(sum_s1_s2 / (s1_norm * s2_norm + K.epsilon()))

    return sam


@tf.function
def my_softmax(x: EagerTensor) -> EagerTensor:
    x_new = K.zeros_like(x[:, 0:0])

    for start, stop in gimme_indices(num_minerals, endmembers_counts):
        tmp = K.softmax(x[..., start:stop])
        x_new = K.concatenate([x_new, tmp], axis=1)

    return x_new


@tf.function
def my_relu(x: EagerTensor) -> EagerTensor:
    scale = 5.0
    x_new = K.zeros_like(x[:, 0:0])

    for start, stop in gimme_indices(num_minerals, endmembers_counts):
        tmp = (relu(x[..., start:stop]) - relu(x[..., start:stop] - scale)) / scale  # to keep results between 0 and 1
        tmp /= (K.sum(tmp, axis=1, keepdims=True) + K.epsilon())  # normalisation to unit sum

        x_new = K.concatenate([x_new, K.softmax(x[..., start:stop])], axis=1)

    return x_new


@tf.function
def my_sigmoid(x: EagerTensor) -> EagerTensor:
    x_new = K.zeros_like(x[:, 0:0])

    for start, stop in gimme_indices(num_minerals, endmembers_counts):
        tmp = sigmoid(x[..., start:stop])
        tmp /= (K.sum(tmp, axis=1, keepdims=True) + K.epsilon())  # normalisation to unit sum

        x_new = K.concatenate([x_new, tmp], axis=1)

    return x_new


@tf.function
def my_plu(x: EagerTensor) -> EagerTensor:
    # https://arxiv.org/pdf/1809.09534.pdf
    alpha, c = 0.1, 1.0
    x_new = K.zeros_like(x[:, 0:0])

    for start, stop in gimme_indices(num_minerals, endmembers_counts):
        tmp = relu(x[..., start:stop] + c) - c - (1 - alpha) * relu(x[..., start:stop] - c) - alpha * relu(
            -x[..., start:stop] - c)
        tmp /= (K.sum(tmp, axis=1, keepdims=True) + K.epsilon())  # normalisation to unit sum

        x_new = K.concatenate([x_new, tmp], axis=1)

    return x_new


# losses
if num_minerals > 0 and np.sum(endmembers_counts) > 0:
    loss_name, loss = "loss", my_loss_general(p["alpha"])
    loss_tuner = my_loss_general  # tuner enters "alpha" HP during compilation
elif num_minerals == 0:
    loss_name, loss = "loss", my_loss_no_minerals()
else:  # np.sum(endmembers_counts) == 0
    loss_name, loss = "loss", my_loss_no_endmembers()

# metrics
mae_name, mae = "mae", my_mae(num_minerals, all_to_one=True)
sse_name, sse = "sse", my_sse(num_minerals, all_to_one=True)
mse_name, mse = "mse", my_mse(num_minerals, all_to_one=True)
rmse_name, rmse = "rmse", my_rmse(num_minerals, all_to_one=True)
Lp_norm_name, Lp_norm = "Lp_norm", my_Lp_norm(num_minerals, p_coef=1.5, all_to_one=True)

quantile_name, quantile = "quantile", my_quantile(num_minerals, percentile=50.0, all_to_one=True)
r2_name, r2 = "r2", my_r2(num_minerals, all_to_one=True)
sam_name, sam = "sam", my_sam(num_minerals, all_to_one=True)

# important for hp tuning and early stopping
objective, direction = "val_loss", "min"  # minimise or maximise the objective?

# activations
if p["output_activation"] == "softmax":
    output_activation_name, output_activation = "my_softmax", my_softmax
elif p["output_activation"] == "relu":
    output_activation_name, output_activation = "my_relu", my_relu
elif p["output_activation"] == "sigmoid":
    output_activation_name, output_activation = "my_sigmoid", my_sigmoid
else:
    raise NameError('p["output_activation"] must be one of "relu", "sigmoid", "softmax"')

# used by load_model function
custom_objects = {loss_name: loss, output_activation_name: output_activation,
                  mse_name: mse, rmse_name: rmse, quantile_name: quantile, mae_name: mae, Lp_norm_name: Lp_norm,
                  r2_name: r2, sam_name: sam, sse_name: sse}
