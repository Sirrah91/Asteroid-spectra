# These are control plot used in the NN pipeline
import numpy as np
import pandas as pd
from typing import Tuple

from modules.NN_losses_metrics_activations import my_rmse, my_r2, my_sam, my_quantile
from modules.NN_config import *


def accuracy_table(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print('Print accuracy metrics')

    RMSE = np.round(my_rmse(num_minerals)(y_true, y_pred), 1)
    R2 = np.round(my_r2(num_minerals)(y_true, y_pred), 2)
    SAM = np.round(my_sam(num_minerals)(y_true, y_pred) * 180 / np.pi, 1)

    print('RMSE:', '[' + ', '.join('{:.1f}'.format(k) for k in RMSE) + ']')
    print('R2:', '[' + ', '.join('{:.2f}'.format(k) for k in R2) + ']')
    print('SAM:', '[' + ', '.join('{:.1f}'.format(k) for k in SAM) + ']')


def quantile_table(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print('Print quantile table')
    percentile = np.arange(0, 101, 1)

    quantile = my_quantile(num_minerals, percentile)(y_true, y_pred)

    print('5 procent: ' + str(list(percentile[np.where(quantile[:, i] < 5)[0]][-1] for i in range(num_labels))))
    print('10 procent: ' + str(list(percentile[np.where(quantile[:, i] < 10)[0]][-1] for i in range(num_labels))))
    print('15 procent: ' + str(list(percentile[np.where(quantile[:, i] < 15)[0]][-1] for i in range(num_labels))))
    print('20 procent: ' + str(list(percentile[np.where(quantile[:, i] < 20)[0]][-1] for i in range(num_labels))))


def mean_asteroid_type(y_pred: np.ndarray) -> None:
    print('Print mean composition of each asteroid type')

    predictions = y_pred * 100

    filename = project_dir + '/Datasets/taxonomy/asteroid_spectra-denoised-norm-meta.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = data[:, 1].astype(np.str)

    inds_S = np.array(['S' in ast_type for ast_type in types])
    inds_Q = np.array(['Q' in ast_type for ast_type in types])
    inds_V = np.array(['V' in ast_type for ast_type in types])
    inds_A = np.array(['A' in ast_type for ast_type in types])

    S = np.round(np.mean(predictions[inds_S, :], axis=0), 1)
    Q = np.round(np.mean(predictions[inds_Q, :], axis=0), 1)
    V = np.round(np.mean(predictions[inds_V, :], axis=0), 1)
    A = np.round(np.mean(predictions[inds_A, :], axis=0), 1)

    sigma_S = np.round(np.std(predictions[inds_S, :], axis=0, ddof=1), 1)
    sigma_Q = np.round(np.std(predictions[inds_Q, :], axis=0, ddof=1), 1)
    sigma_V = np.round(np.std(predictions[inds_V, :], axis=0, ddof=1), 1)
    sigma_A = np.round(np.std(predictions[inds_A, :], axis=0, ddof=1), 1)

    S, sigma_S = round_data_with_errors(S, sigma_S)
    Q, sigma_Q = round_data_with_errors(Q, sigma_Q)
    V, sigma_V = round_data_with_errors(V, sigma_V)
    A, sigma_A = round_data_with_errors(A, sigma_A)

    tmp = ['{:.1f}'.format(k1) + '+-' + '{:.1f}'.format(k2) for k1, k2 in zip(S, sigma_S)]
    print('S:', '[' + ', '.join(line for line in tmp) + ']')

    tmp = ['{:.1f}'.format(k1) + '+-' + '{:.1f}'.format(k2) for k1, k2 in zip(Q, sigma_Q)]
    print('Q:', '[' + ', '.join(line for line in tmp) + ']')

    tmp = ['{:.1f}'.format(k1) + '+-' + '{:.1f}'.format(k2) for k1, k2 in zip(V, sigma_V)]
    print('V:', '[' + ', '.join(line for line in tmp) + ']')

    tmp = ['{:.1f}'.format(k1) + '+-' + '{:.1f}'.format(k2) for k1, k2 in zip(A, sigma_A)]
    print('A:', '[' + ', '.join(line for line in tmp) + ']')


def mean_S_asteroid_type(y_pred: np.ndarray) -> None:
    print('Print mean composition of each asteroid type')

    predictions = y_pred * 100

    filename = project_dir + '/Datasets/taxonomy/asteroid_spectra-denoised-norm-meta.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = data[:, 1].astype(np.str)

    inds_S = np.array(['S' in ast_type for ast_type in types])
    unique, counts = np.unique(types[inds_S], return_counts=True)

    for i in range(len(unique)):
        inds = types == unique[i]
        column = np.round(np.mean(predictions[inds, :], axis=0), 1)
        sigma_column = np.round(np.std(predictions[inds, :], axis=0, ddof=1), 1)

        if np.sum(sigma_column) > 0:  # not NaNs
            column, sigma_column = round_data_with_errors(column, sigma_column)

        tmp = ['{:.1f}'.format(k1) + '+-' + '{:.1f}'.format(k2) for k1, k2 in zip(column, sigma_column)]
        print(unique[i] + ':', '[' + ', '.join(line for line in tmp) + ']')


def chlyabinsk_composition() -> None:
    densities = np.array([4.49, 3.27, 3.95, 3.20, 2.90, 3.95, 3.20, 2.90, 2.73, 2.62, 2.56])
    modal = np.array([56.2, 26.9, 0, 0])
    modal /= np.sum(modal)
    modal *= 100

    chemical = np.array([28.6, 100-28.6, 23.5, 100-23.5-1.6, 1.6, 0, 0, 0, 8.5, 84.5, 7])
    mineral_density = chemical * densities
    ol = np.sum(mineral_density[:2], keepdims=True)
    opx = np.sum(mineral_density[2:5], keepdims=True)
    cpx = np.sum(mineral_density[5:8], keepdims=True)
    plg = np.sum(mineral_density[8:], keepdims=True)

    mineral_density = np.concatenate((ol, opx, cpx, plg))
    mineral_density[mineral_density == 0] = 1

    modal /= mineral_density
    norm = np.sum(modal)
    modal = np.transpose(np.divide(np.transpose(modal), norm)) * 100
    chemical[2:4] /= np.sum(chemical[2:4])
    chemical[2:4] *= 100

    composition = np.concatenate((modal, chemical))
    print(np.round(composition, 1))


def how_many_data(y_data: np.ndarray) -> None:
    # table with specific mixtures
    binary = (y_data[:, :3] > 0).astype(int)    # :3 is potentially dangerous if not all minerals are in the model
    base = np.array([4, 2, 1])

    result = np.sum(binary * base, axis=1)

    # PURE MINERALS
    OL = np.sum(result == 4)
    OPX = np.sum(result == 2)
    CPX = np.sum(result == 1)

    # BINARY MIXTURES
    OL_OPX = np.sum(result == 6)
    OL_CPX = np.sum(result == 5)
    OPX_CPX = np.sum(result == 3)

    # TERNARY MIXTURES
    OL_OPX_CPX = np.sum(result == 7)

    rows = np.array([np.array([np.sort(np.round(y_data[result == res, i]*100, 1))[[0, -1]]
                               for res in [4, 2, 1, 6, 5, 3, 7]]).ravel() for i in [0, 1, 2, 3, 5, 7, 8, 9]])
    print(rows)

    '''
    y_data = y_data[:, :4]  # :4 is potentially dangerous if not all minerals are in the model

    binary = (y_data > 0).astype(int)
    base = np.array([8, 4, 2, 1])

    result = np.sum(binary * base, axis=1)

    # PURE MINERALS
    OL = np.sum(result == 8)
    OPX = np.sum(result == 4)
    CPX = np.sum(result == 2)
    PLG = np.sum(result == 1)

    # BINARY MIXTURES
    OL_OPX = np.sum(result == 12)
    OL_CPX = np.sum(result == 10)
    OL_PLG = np.sum(result == 9)
    OPX_CPX = np.sum(result == 6)
    OPX_PLG = np.sum(result == 5)
    CPX_PLG = np.sum(result == 3)

    # TERNARY MIXTURES
    OL_OPX_CPX = np.sum(result == 14)
    OL_OPX_PLG = np.sum(result == 13)
    OL_CPX_PLG = np.sum(result == 11)
    OPX_CPX_PLG = np.sum(result == 7)

    # QUATERNARY MIXTURES
    OL_OPX_CPX_PLG = np.sum(result == 15)
    
    rows = np.array([np.array([np.sort(np.round(y_train[result == res, i]*100, 1))[[0, -1]] 
                               for res in [4, 2, 1, 6, 5, 3, 7]]).ravel() for i in [0, 1, 2, 3, 4, 6, 8, 9, 10]])
    print(rows)
    '''


def A_type_properties(y_pred: np.ndarray) -> None:
    print('Print composition of the selected A-type asteroids')

    ind_Fa, ind_Fs = num_minerals, num_minerals + subtypes[0]

    filename = project_dir + '/Datasets/taxonomy/asteroid_spectra-denoised-norm-meta.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    ast_nums = data[:, 0].astype(np.str)

    nums = ['246', '289', '354', '1951', '5261']
    indices = np.array([np.where(ast_nums == N)[0][0] for N in nums])

    Fa, Fs = y_pred[:, ind_Fa] * 100, y_pred[:, ind_Fs] * 100

    Fa_A, Fs_A = np.round(Fa[indices], 1), np.round(Fs[indices], 1)

    print('Fa:', '[' + ', '.join('{:.1f}'.format(k) for k in Fa_A) + ']')
    print('Fs:', '[' + ', '.join('{:.1f}'.format(k) for k in Fs_A) + ']')


def round_data_with_errors(data: np.ndarray, errors: np.ndarray) -> Tuple[np.ndarray, ...]:
    n = 2 - np.ceil(np.log10(errors))  # rounding to 2 valid number

    data_rounded, errors_rounded = np.zeros(np.shape(data)), np.zeros(np.shape(errors))

    for i in range(len(data)):
        data_rounded[i] = np.round(data[i], n[i])
        errors_rounded[i] = np.round(errors[i], n[i])

    return np.round(data_rounded, 1), np.round(errors_rounded, 1)
