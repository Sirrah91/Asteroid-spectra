# These are control plot used in the NN pipeline
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.metrics import classification_report

from modules.NN_evaluate import evaluate
from modules.NN_losses_metrics_activations import my_rmse, my_r2, my_sam, my_quantile
from modules.utilities import round_data_with_errors, replace_spaces_with_phantom
from modules.NN_config import *


def how_many_data(y_data: np.ndarray) -> None:
    # table with specific mixtures
    binary = (y_data[:, :num_minerals] > 0).astype(int)
    base = np.array(2**(np.arange(num_minerals - 1, -1, -1)))

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

    counts = np.array([[OL, OPX, CPX, OL_OPX, OL_CPX, OPX_CPX, OL_OPX_CPX]])

    rows = np.array([np.array([np.sort(np.round(y_data[result == res, i] * 100, 1))[[0, -1]]
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


def accuracy_table(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print('Print accuracy metrics')

    RMSE = np.round(my_rmse(num_minerals)(y_true, y_pred), 1)
    R2 = np.round(my_r2(num_minerals)(y_true, y_pred), 2)
    SAM = np.round(my_sam(num_minerals)(y_true, y_pred) * 180 / np.pi, 1)

    print('RMSE:', '[' + ', '.join('{:.1f}'.format(k) for k in RMSE) + ']')
    print('R2:', '[' + ', '.join('{:.2f}'.format(k) for k in R2) + ']')
    print('SAM:', '[' + ', '.join('{:.1f}'.format(k) for k in SAM) + ']')

    inds_to_delete = np.array([1, 8, 16, 25, 45])
    y_true, y_pred = np.delete(y_true, inds_to_delete, axis=0), np.delete(y_pred, inds_to_delete, axis=0)

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
    quantile_all = my_quantile(num_minerals, percentile, True)(y_true, y_pred)

    print('5 procent: ' + str(percentile[np.where(quantile_all <= 5)[0]][-1]))
    print('10 procent: ' + str(percentile[np.where(quantile_all <= 10)[0]][-1]))
    print('15 procent: ' + str(percentile[np.where(quantile_all <= 15)[0]][-1]))
    print('20 procent: ' + str(percentile[np.where(quantile_all <= 20)[0]][-1]))

    print('5 procent: ' + str(list(percentile[np.where(quantile[:, i] <= 5)[0]][-1] for i in range(num_labels))))
    print('10 procent: ' + str(list(percentile[np.where(quantile[:, i] <= 10)[0]][-1] for i in range(num_labels))))
    print('15 procent: ' + str(list(percentile[np.where(quantile[:, i] <= 15)[0]][-1] for i in range(num_labels))))
    print('20 procent: ' + str(list(percentile[np.where(quantile[:, i] <= 20)[0]][-1] for i in range(num_labels))))

    print('1-sigma error: ' + str(np.round(quantile[np.where(percentile == 68)[0][0], :].numpy(), 1)))


def mean_asteroid_type(y_pred: np.ndarray) -> None:
    print('Print mean composition of each asteroid type')

    predictions = y_pred * 100

    filename = project_dir + '/Datasets/taxonomy/asteroid_spectra-denoised-norm.npz'
    data = np.load(filename, allow_pickle=True)  # to read the file
    types = data["metadata"][:, 1].astype(np.str)

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

    tmp = np.array([['{:4.1f}'.format(k1) + '\\pm' + '{:4.1f}'.format(k2) for k1, k2 in zip(S, sigma_S)]])
    tmp = np.vstack((tmp, np.array([['{:4.1f}'.format(k1) + '\\pm' + '{:4.1f}'.format(k2) for k1, k2 in zip(Q, sigma_Q)]])))
    tmp = np.vstack((tmp, np.array([['{:4.1f}'.format(k1) + '\\pm' + '{:4.1f}'.format(k2) for k1, k2 in zip(V, sigma_V)]])))
    tmp = np.vstack((tmp, np.array([['{:4.1f}'.format(k1) + '\\pm' + '{:4.1f}'.format(k2) for k1, k2 in zip(A, sigma_A)]]))).T.astype("<U100")

    # replace spaces with phantom numbers
    for i in range(len(tmp)):
        tmp[i] = replace_spaces_with_phantom(tmp[i])

    lines = [" & ".join(line) + " \\\\" for line in tmp]

    names = ['OL [vol\%]', 'OPX [vol\%]', 'CPX [vol\%]', 'Fa', 'Fo', 'Fs (OPX)', 'En (OPX)', 'Fs (CPX)', 'En (CPX)', 'Wo (CPX)']

    for i, line in enumerate(lines):
        print(names[i] + ' & ' + line.replace("\\pm", " $\\pm$ "))
        if i in [2, 4, 6]:
            print('%\n\\hdashline\n%')


def mean_S_asteroid_type(y_pred: np.ndarray) -> None:
    print('Print mean composition of each asteroid type')

    predictions = y_pred * 100

    filename = project_dir + '/Datasets/taxonomy/asteroid_spectra-denoised-norm.npz'
    data = np.load(filename, allow_pickle=True)  # to read the file
    types = data["metadata"][:, 1].astype(np.str)

    inds_S = np.array(['S' in ast_type for ast_type in types])
    unique, table = np.unique(types[inds_S], return_counts=True)

    inds_order = [0, -1, 2, 4, 5, 6, 7, 8, 1]
    table = table[inds_order]
    table = np.reshape(table, (1, len(table)))

    for c, i in enumerate(inds_order):  # range(len(unique)):
        inds = types == unique[i]
        column = np.round(np.mean(predictions[inds, :], axis=0), 1)
        sigma_column = np.round(np.std(predictions[inds, :], axis=0, ddof=1), 1)

        if np.sum(sigma_column) > 0:  # not NaNs
            column, sigma_column = round_data_with_errors(column, sigma_column)

        if c == 0:
            table_tmp = np.array([['{:4.1f}'.format(k1) + '\\pm' + '{:4.1f}'.format(k2) for k1, k2 in zip(column, sigma_column)]])
        else:
            table_tmp = np.vstack((table_tmp, np.array([['{:4.1f}'.format(k1) + '\\pm' + '{:4.1f}'.format(k2) for k1, k2 in zip(column, sigma_column)]])))

    table = np.vstack((table, table_tmp.T)).astype("<U100")

    # replace spaces with phantom numbers
    for i in range(len(table)):
        table[i] = replace_spaces_with_phantom(table[i])

    lines = [" & ".join(line) + " \\\\" for line in table]

    names = ['Number', 'OL [vol\%]', 'OPX [vol\%]', 'CPX [vol\%]', 'Fa', 'Fo', 'Fs (OPX)', 'En (OPX)', 'Fs (CPX)', 'En (CPX)', 'Wo (CPX)']

    for i, line in enumerate(lines):
        print(names[i] + ' & ' + line.replace("\\pm", " $\\pm$ "))
        if i in [0, 3, 5, 7]:
            print('%\n\\hdashline\n%')


def chelyabinsk_composition() -> None:
    densities = np.array([4.39, 3.27, 3.95, 3.20, 2.90, 3.95, 3.20, 2.90, 2.73, 2.62, 2.56])
    modal = np.array([56.2, 26.9, 0, 0])
    modal /= np.sum(modal)
    modal *= 100

    chemical = np.array([28.6, 71.4, 23.5, 74.9, 1.6, 0, 0, 0, 8.5, 84.5, 7])
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
    chemical[2:4] /= np.sum(chemical[2:4])  # normalise Wo (CPX) out, but is still in the array...
    chemical[2:4] *= 100

    composition = np.concatenate((modal, chemical))
    print(np.round(composition, 1))


def kachr_experiment() -> None:
    filename = 'Kachr_ol_opx-denoised-norm.npz'
    predictions = evaluate(['20220330113805_CNN.h5'], filename)

    ol_stop = 24 if 'raw' in filename else 34

    ol = predictions[:ol_stop, [0, 1, 2, 3]] * 100
    opx = predictions[ol_stop:, [0, 1, 2, 5]] * 100

    ol = np.round(np.concatenate((np.array([[100, 0, 0, 9.9]]), ol)), 1)
    opx = np.round(np.concatenate((np.array([[0, 84/(84+5) * 100, 5/(84+5) * 100, 33]]), opx)), 1)  # wt%

    print(np.sum(ol[1:, 0] >= 99) / len(opx[1:, 0]))
    print(np.sum(np.sum(opx[1:, 1:3], 1) >= 99) / len(np.sum(opx[1:, 1:3], 1)))

    if 'raw' not in filename:
        ol_ar, ol_h, ol_he, ol_laser = ol[1:9], ol[9:17], ol[17:22], ol[22:]
        opx_ar, opx_h, opx_he, opx_laser = opx[1:9], opx[9:15], opx[15:20], opx[20:]


def chelyabinsk_sw(y_pred: np.ndarray) -> None:
    predictions = np.round(y_pred * 100, 1)

    SD_names = np.array(['SD 0\\%', 'SD 5\\%', 'SD 10\\%', 'SD 20\\%', 'SD 30\\%', 'SD 40\\%', 'SD 50\\%',
                         'SD 60\\%', 'SD 70\\%', 'SD 80\\%', 'SD 90\\%', 'SD 95\\%', 'SD 100\\%'])
    pred_SD = predictions[:len(SD_names), [0, 1, 2, 3, 5, 7, 9]]

    IM_names = np.array(['IM 0\\%', 'IM 10\\%', 'IM 20\\%', 'IM 30\\%', 'IM 40\\%', 'IM 50\\%',
                         'IM 60\\%', 'IM 70\\%', 'IM 80\\%', 'IM 90\\%', 'IM 95\\%', 'IM 100\\%'])
    pred_IM = predictions[len(SD_names):len(SD_names) + len(IM_names), [0, 1, 2, 3, 5, 7, 9]]

    SW_names = np.array(['fresh', 'SW 400', 'SW 500', 'SW 600', 'SW 700'])
    pred_SW = predictions[len(SD_names) + len(IM_names):, [0, 1, 2, 3, 5, 7, 9]]

    for what, names in zip([pred_SD, pred_IM, pred_SW], [SD_names, IM_names, SW_names]):
        rows = len(what) * [0]

        for i, row in enumerate(what):
            tmp = np.array(['{:4.1f}'.format(k1) for k1 in row]).astype("<U100")
            tmp = replace_spaces_with_phantom(tmp.ravel())

            rows[i] = np.hstack((names[i], tmp))

        rows = np.array(rows)

        lines = [" & ".join(line) + " \\\\" for line in rows]
        for line in lines:
            print(line)


def kachr_sw(y_pred: np.ndarray) -> None:
    predictions = np.round(y_pred * 100, 1)

    Ar_names = np.array(['fresh', '1e15 Ar$^+$\\,cm$^{-2}$', '3e15 Ar$^+$\\,cm$^{-2}$', '6e15 Ar$^+$\\,cm$^{-2}$',
                            '1e16 Ar$^+$\\,cm$^{-2}$', '2e16 Ar$^+$\\,cm$^{-2}$', '6e16 Ar$^+$\\,cm$^{-2}$',
                            '1e17 Ar$^+$\\,cm$^{-2}$'])
    OL_Ar = predictions[:len(Ar_names), :4]
    PX_Ar = predictions[34:34+len(Ar_names), [0, 1, 2, 5]]
    Ar = np.hstack((OL_Ar, PX_Ar))

    H_names = np.array(['fresh', '1e14 H$^+$\\,cm$^{-2}$', '1e15 H$^+$\\,cm$^{-2}$', '1e16 H$^+$\\,cm$^{-2}$',
                        '1e17 H$^+$\\,cm$^{-2}$', '2e17 H$^+$\\,cm$^{-2}$', '5e17 H$^+$\\,cm$^{-2}$',
                        '1e18 H$^+$\\,cm$^{-2}$'])
    OL_H = predictions[len(Ar_names):len(Ar_names) + len(H_names), :4]
    PX_H_raw = predictions[34+len(Ar_names):34+len(Ar_names)+len(H_names)-2, [0, 1, 2, 5]]

    PX_H = np.zeros((np.shape(OL_H)[0], np.shape(PX_H_raw)[1]))
    PX_H[0] = PX_H_raw[0]
    PX_H[3:] = PX_H_raw[1:]
    H = np.hstack((OL_H, PX_H))

    He_names = np.array(['fresh', '1e16 He$^+$\\,cm$^{-2}$', '3e16 He$^+$\\,cm$^{-2}$', '6e16 He$^+$\\,cm$^{-2}$',
                         '1e17 He$^+$\\,cm$^{-2}$'])
    OL_He = predictions[len(Ar_names) + len(H_names):len(Ar_names) + len(H_names) + len(He_names), :4]
    PX_He = predictions[34+len(Ar_names)+len(H_names)-2:34+len(Ar_names)+len(H_names)-2+len(He_names), [0, 1, 2, 5]]
    He = np.hstack((OL_He, PX_He))

    for what, names in zip([H, He, Ar], [H_names, He_names, Ar_names]):
        rows = len(what) * [0]

        for i, row in enumerate(what):
            tmp = np.array(['{:5.1f}'.format(k1) for k1 in row]).astype("<U100")
            tmp = replace_spaces_with_phantom(tmp.ravel())

            rows[i] = np.hstack((names[i], tmp))

        rows = np.array(rows)

        lines = [" & ".join(line) + " \\\\" for line in rows]
        for line in lines:
            print(line.replace('\phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0', '- & - & - & -'))
        print('%\n\\hdashline\n%')


def kachr_sw_laser(y_pred: np.ndarray) -> None:
    predictions = np.round(y_pred * 100, 1)

    OL_names = np.array(['fresh', '\\phantom{00}1.7 J\\,cm$^{-2}$', '\\phantom{00}2.4 J\\,cm$^{-2}$',
                         '\\phantom{00}3.8 J\\,cm$^{-2}$', '\\phantom{00}4.6 J\\,cm$^{-2}$',
                         '\\phantom{00}6.7 J\\,cm$^{-2}$', '\\phantom{0}10.4 J\\,cm$^{-2}$',
                         '\\phantom{0}15.0 J\\,cm$^{-2}$', '\\phantom{0}23.4 J\\,cm$^{-2}$',
                         '\\phantom{0}30.6 J\\,cm$^{-2}$', '\\phantom{0}60.0 J\\,cm$^{-2}$',
                         '\\phantom{0}93.8 J\\,cm$^{-2}$', '375.0 J\\,cm$^{-2}$'])
    OL = predictions[21:34, :4]

    PX_names = np.array(['fresh', '\\phantom{000}4.5 J\\,cm$^{-2}$', '\\phantom{000}5.6 J\\,cm$^{-2}$',
                         '\\phantom{00}12.5 J\\,cm$^{-2}$', '\\phantom{00}18.0 J\\,cm$^{-2}$',
                         '\\phantom{00}28.1 J\\,cm$^{-2}$', '\\phantom{00}36.7 J\\,cm$^{-2}$',
                         '\\phantom{00}50.0 J\\,cm$^{-2}$', '\\phantom{00}72.0 J\\,cm$^{-2}$',
                         '\\phantom{0}112.5 J\\,cm$^{-2}$', '\\phantom{0}200.0 J\\,cm$^{-2}$',
                         '\\phantom{0}450.0 J\\,cm$^{-2}$', '1800.0 J\\,cm$^{-2}$'])
    PX = predictions[-len(PX_names):, [0, 1, 2, 5]]

    for what, names in zip([OL, PX], [OL_names, PX_names]):
        rows = len(what) * [0]

        for i, row in enumerate(what):
            tmp = np.array(['{:5.1f}'.format(k1) for k1 in row]).astype("<U100")
            tmp = replace_spaces_with_phantom(tmp.ravel())

            rows[i] = np.hstack((names[i], tmp))

        rows = np.array(rows)

        lines = [" & ".join(line) + " \\\\" for line in rows]
        for line in lines:
            print(line.replace('\phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0', '- & - & - & -'))
        print('%\n\\hdashline\n%')


def A_type_properties(y_pred: np.ndarray) -> None:
    print('Print composition of the selected A-type asteroids')

    ind_Fa, ind_Fs = num_minerals, num_minerals + subtypes[0]

    filename = project_dir + '/Datasets/taxonomy/asteroid_spectra-denoised-norm.npz'
    data = np.load(filename, allow_pickle=True)  # to read the file
    ast_nums = data["metadata"][:, 0].astype(np.str)

    nums = ['246', '289', '354', '1951', '5261']
    indices = np.array([np.where(ast_nums == N)[0][0] for N in nums])

    Fa, Fs = y_pred[:, ind_Fa] * 100, y_pred[:, ind_Fs] * 100

    Fa_A, Fs_A = np.round(Fa[indices], 1), np.round(Fs[indices], 1)

    print('Fa:', '[' + ', '.join('{:.1f}'.format(k) for k in Fa_A) + ']')
    print('Fs:', '[' + ', '.join('{:.1f}'.format(k) for k in Fs_A) + ']')


def taxonomy_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    from modules.NN_config_taxonomy import classes
    target_names = classes.keys()
    print(classification_report(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), target_names=target_names))

    text = classification_report(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1),
                                 target_names=target_names).split()

    text_cls = np.array(text[4:-15]).astype("<U100")
    text_tot = np.array(text[-5:]).astype("<U100")
    text_tot[0] = 'total'

    text_cls = np.reshape(text_cls, (np.int(np.round(np.size(text_cls) / 5)), 5))
    text_tot = np.reshape(text_tot, (np.int(np.round(np.size(text_tot) / 5)), 5))

    # add phantom
    order = len(text_tot.ravel()[-1])
    for i, row in enumerate(text_cls):
        text_cls[i, -1] = (order - len(row[-1])) * '\\phantom{0}' + row[-1]

    lines = [" & ".join(row) + ' \\\\' for row in text_cls]
    lines += ['%\n\\hdashline\n%'] + [" & ".join(row) + ' \\\\' for row in text_tot]

    for line in lines:
        print(line)


def taxonomy_class_of_minerals():
    pass
