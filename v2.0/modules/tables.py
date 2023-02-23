import numpy as np
from sklearn.metrics import classification_report
from tensorflow_addons.metrics import CohenKappa

from modules.NN_evaluate import evaluate

from modules.NN_losses_metrics_activations import my_rmse, my_r2, my_sam, my_quantile
from modules.utilities import round_data_with_errors, replace_spaces_with_phantom, flatten_list, stack
from modules.utilities_spectra import find_outliers, return_mineral_position, unique_indices, used_indices

from modules.NN_config import num_minerals, num_labels, endmembers_counts, minerals_used, endmembers_used
from modules.NN_config import mineral_names_short, endmember_names
from modules.NN_config_taxonomy import classes, classes2

from modules._constants import _path_data


def how_many_data(y_data: np.ndarray) -> None:
    # table with specific mixtures
    binary = np.array(y_data[:, :num_minerals] > 0, dtype=int)
    # this keeps the same order of mixtures even if a mineral is not present (each mineral has own base)
    base = np.array([2 ** i for i in reversed(range(len(minerals_used)))])[minerals_used]
    mixtures = np.sum(binary * base, axis=1)

    # PURE MINERALS
    OL = np.sum(mixtures == 8)
    OPX = np.sum(mixtures == 4)
    CPX = np.sum(mixtures == 2)
    PLG = np.sum(mixtures == 1)

    # BINARY MIXTURES
    OL_OPX = np.sum(mixtures == 12)
    OL_CPX = np.sum(mixtures == 10)
    OL_PLG = np.sum(mixtures == 9)
    OPX_CPX = np.sum(mixtures == 6)
    OPX_PLG = np.sum(mixtures == 5)
    CPX_PLG = np.sum(mixtures == 3)

    # TERNARY MIXTURES
    OL_OPX_CPX = np.sum(mixtures == 14)
    OL_OPX_PLG = np.sum(mixtures == 13)
    OL_CPX_PLG = np.sum(mixtures == 11)
    OPX_CPX_PLG = np.sum(mixtures == 7)

    # QUATERNARY MIXTURES
    OL_OPX_CPX_PLG = np.sum(mixtures == 15)

    counts = np.array([[OL, OPX, CPX, OL_OPX, OL_CPX, OPX_CPX, OL_OPX_CPX]])

    rows = np.array([np.array([np.sort(np.round(y_data[mixtures == res, i] * 100, 1))[[0, -1]]
                               for res in [8, 4, 2, 12, 10, 6, 14]]).ravel() for i in [0, 1, 2, 3, 5, 7, 8, 9]])

    print(rows)


def accuracy_table(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print("Print accuracy metrics")

    RMSE = np.round(my_rmse(num_minerals)(y_true, y_pred), 1)
    R2 = np.round(my_r2(num_minerals)(y_true, y_pred), 2)
    SAM = np.round(my_sam(num_minerals)(y_true, y_pred) * 180. / np.pi, 1)

    print("-------------------------- ALL DATA ------------------------------")
    print("RMSE:", "[" + ", ".join("{:4.1f}".format(k) for k in RMSE) + "]")
    print("R2:  ", "[" + ", ".join("{:4.2f}".format(k) for k in R2) + "]")
    print("SAM: ", "[" + ", ".join("{:4.1f}".format(k) for k in SAM) + "]")

    # no outliers

    inds_to_delete = find_outliers(y_true, y_pred, num_minerals)
    y_true, y_pred = np.delete(y_true, inds_to_delete, axis=0), np.delete(y_pred, inds_to_delete, axis=0)

    RMSE = np.round(my_rmse(num_minerals)(y_true, y_pred), 1)
    R2 = np.round(my_r2(num_minerals)(y_true, y_pred), 2)
    SAM = np.round(my_sam(num_minerals)(y_true, y_pred) * 180. / np.pi, 1)

    print("------------------------- NO OUTLIERS ----------------------------")
    print("RMSE:", "[" + ", ".join("{:4.1f}".format(k) for k in RMSE) + "]")
    print("R2:  ", "[" + ", ".join("{:4.2f}".format(k) for k in R2) + "]")
    print("SAM: ", "[" + ", ".join("{:4.1f}".format(k) for k in SAM) + "]")


def quantile_table(y_true: np.ndarray, y_pred: np.ndarray, latex_output: bool = False) -> None:
    print("Print quantile table")
    percentile = np.arange(0, 101, 1)

    quantile = my_quantile(num_minerals, percentile)(y_true, y_pred)
    quantile_all = my_quantile(num_minerals, percentile, True)(y_true, y_pred)

    where_evaluate = [5., 10., 15., 20.]

    if latex_output:
        table = np.zeros((num_labels + 1, len(where_evaluate)), dtype=str)
        table[0] = np.array([percentile[np.where(quantile_all <= where)[0]][-1] for where in where_evaluate])
        table[1:] = np.transpose(np.array([[percentile[np.where(quantile[:, i] <= where)[0]][-1]
                                            for i in range(num_labels)] for where in where_evaluate]))

        # +1 because at 0 coordinate is all data
        table = table[np.insert(unique_indices(minerals_used, endmembers_used, return_digits=True) + 1, 0, 0)]

        names = [[s + " (vol\%)" for s in mineral_names_short]] + endmember_names
        names = flatten_list(names)[unique_indices(minerals_used, endmembers_used, all_minerals=True)]
        names = np.insert(names, 0, "All data")

        """
        # replace spaces with phantom numbers
        for i, tab in enumerate(table):
            table[i] = replace_spaces_with_phantom(tab)
        """

        lines = [" & ".join(np.core.defchararray.add(line, "\\%")) + " \\\\" for line in table]

        for i, line in enumerate(lines):
            print("".join((names[i], " & ", line.replace("\\pm", " $\\pm$ "))))
            if i in [0, 3, 4, 5]:
                print("%\n\\hdashline\n%")
    else:
        print("--------------------------- ALL LABELS ---------------------------")
        print(" 5 percent:", "{:2}".format(percentile[np.where(quantile_all <= 5)[0]][-1]))
        print("10 percent:", "{:2}".format(percentile[np.where(quantile_all <= 10)[0]][-1]))
        print("15 percent:", "{:2}".format(percentile[np.where(quantile_all <= 15)[0]][-1]))
        print("20 percent:", "{:2}".format(percentile[np.where(quantile_all <= 20)[0]][-1]))

        print("----------------------- INDIVIDUAL LABELS ------------------------")
        print(" 5 percent:   ", "[", ", ".join("{:3}".format(k) for k in [percentile[np.where(quantile[:, i] <= 5)[0]][-1]
                                                                          for i in range(num_labels)]) + "]")
        print("10 percent:   ", "[", ", ".join("{:3}".format(k) for k in [percentile[np.where(quantile[:, i] <= 10)[0]][-1]
                                                                          for i in range(num_labels)]) + "]")
        print("15 percent:   ", "[", ", ".join("{:3}".format(k) for k in [percentile[np.where(quantile[:, i] <= 15)[0]][-1]
                                                                          for i in range(num_labels)]) + "]")
        print("20 percent:   ", "[", ", ".join("{:3}".format(k) for k in [percentile[np.where(quantile[:, i] <= 20)[0]][-1]
                                                                          for i in range(num_labels)]) + "]")

    print("1-sigma error (all):", "{:3.1f}".format(np.round(quantile_all[
                                                            np.where(percentile == 68)[0][0], :].numpy(), 1)[0]))
    print("1-sigma error:", "[", ", ".join(
        "{:3.1f}".format(k) for k in np.round(quantile[np.where(percentile == 68)[0][0], :].numpy(), 1)) + "]")


def mean_asteroid_type(y_pred: np.ndarray) -> None:
    print("Print mean composition of each asteroid type")

    predictions = y_pred * 100

    filename = "".join((_path_data, "/taxonomy/asteroid_spectra-denoised-norm.npz"))
    data = np.load(filename, allow_pickle=True)  # to read the file
    types = np.array(data["metadata"][:, 1], dtype=str)

    inds_S = np.array(["S" in ast_type for ast_type in types])
    inds_Q = np.array(["Q" in ast_type for ast_type in types])
    inds_V = np.array(["V" in ast_type for ast_type in types])
    inds_A = np.array(["A" in ast_type for ast_type in types])

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

    tmp = np.array([["".join(("{:4.1f}".format(k1), "\\pm", "{:4.1f}".format(k2))) for k1, k2 in zip(S, sigma_S)]],
                   dtype="<U100")
    tmp = stack((tmp, np.array([["".join(("{:4.1f}".format(k1), "\\pm",  "{:4.1f}".format(k2)))
                                 for k1, k2 in zip(Q, sigma_Q)]])), axis=0)
    tmp = stack((tmp, np.array([["".join(("{:4.1f}".format(k1), "\\pm", "{:4.1f}".format(k2)))
                                 for k1, k2 in zip(V, sigma_V)]])), axis=0)
    tmp = stack((tmp, np.array([["".join(("{:4.1f}".format(k1), "\\pm", "{:4.1f}".format(k2)))
                                 for k1, k2 in zip(A, sigma_A)]])), axis=0)
    tmp = np.transpose(tmp)
    # replace spaces with phantom numbers
    for i, tab in enumerate(tmp):
        tmp[i] = replace_spaces_with_phantom(tab)

    lines = [" & ".join(line) + " \\\\" for line in tmp]

    names = [["".join((s, " (vol\%)")) for s in mineral_names_short]] + endmember_names
    names = flatten_list(names)[used_indices(minerals_used, endmembers_used)]

    for i, line in enumerate(lines):
        print("".join((names[i], " & " + line.replace("\\pm", " $\\pm$ "))))
        if i in [2, 4, 6]:
            print("%\n\\hdashline\n%")


def mean_S_asteroid_type(y_pred: np.ndarray) -> None:
    print("Print mean composition of each asteroid type")

    predictions = y_pred * 100

    filename = "".join((_path_data, "/taxonomy/asteroid_spectra-denoised-norm.npz"))
    data = np.load(filename, allow_pickle=True)  # to read the file
    types = np.array(data["metadata"][:, 1], dtype=str)

    inds_S = np.array(["S" in ast_type for ast_type in types])
    unique, table = np.unique(types[inds_S], return_counts=True)

    inds_order = [0, -1, 2, 4, 5, 6, 7, 8, 1]
    table = table[inds_order]
    table = np.reshape(table, (1, len(table)))

    for c, i in enumerate(inds_order):
        inds = types == unique[i]
        column = np.round(np.mean(predictions[inds, :], axis=0), 1)
        sigma_column = np.round(np.std(predictions[inds, :], axis=0, ddof=1), 1)

        if np.sum(sigma_column) > 0:  # not NaNs
            column, sigma_column = round_data_with_errors(column, sigma_column)

        if c == 0:
            table_tmp = np.array([["".join(("{:4.1f}".format(k1), "\\pm", "{:4.1f}".format(k2)))
                                   for k1, k2 in zip(column, sigma_column)]], dtype="<U100")
        else:
            table_tmp = stack((table_tmp, np.array( [["".join(("{:4.1f}".format(k1), "\\pm", "{:4.1f}".format(k2)))
                                                      for k1, k2 in zip(column, sigma_column)]])), axis=0)

    table = stack((table, np.transpose(table_tmp)), axis=0)

    # replace spaces with phantom numbers
    for i, tab in enumerate(table):
        table[i] = replace_spaces_with_phantom(tab)

    lines = [" & ".join(line) + " \\\\" for line in table]

    names = [["".join((s, " (vol\%)")) for s in mineral_names_short]] + endmember_names
    names = flatten_list(names)[used_indices(minerals_used, endmembers_used)]
    names = np.insert(names, 0, "Number")

    for i, line in enumerate(lines):
        print("".join((names[i], " & ", line.replace("\\pm", " $\\pm$ "))))
        if i in [0, 3, 5, 7]:
            print("%\n\\hdashline\n%")


def chelyabinsk_composition() -> None:
    densities = np.array([4.39, 3.27, 3.95, 3.20, 2.90, 3.95, 3.20, 2.90, 2.73, 2.62, 2.56])
    modal = np.array([56.2, 26.9, 0., 0.])
    modal /= np.sum(modal)
    modal *= 100.

    chemical = np.array([28.6, 71.4, 23.5, 74.9, 1.6, 0, 0, 0, 8.5, 84.5, 7])
    mineral_density = chemical * densities
    ol = np.sum(mineral_density[:2], keepdims=True)
    opx = np.sum(mineral_density[2:5], keepdims=True)
    cpx = np.sum(mineral_density[5:8], keepdims=True)
    plg = np.sum(mineral_density[8:], keepdims=True)

    mineral_density = stack((ol, opx, cpx, plg))
    mineral_density[mineral_density == 0] = 1.

    modal /= mineral_density
    norm = np.sum(modal)
    modal = np.transpose(np.divide(np.transpose(modal), norm)) * 100.
    chemical[2:4] /= np.sum(chemical[2:4])  # normalise Wo (CPX) out, but is still in the array...
    chemical[2:4] *= 100.

    composition = stack((modal, chemical))
    print(np.round(composition, 1))


def kachr_experiment() -> None:
    filename = "Kachr_ol_opx-denoised-norm.npz"
    predictions = evaluate(["20220330113805_CNN.h5"], filename, subfolder_model="composition")

    ol_stop = 24 if "raw" in filename else 34

    ol = predictions[:ol_stop, [0, 1, 2, 3]] * 100.
    opx = predictions[ol_stop:, [0, 1, 2, 5]] * 100.

    ol = np.round(stack((np.array([[100., 0., 0., 9.9]]), ol), axis=0), 1)
    opx = np.round(stack((np.array([[0., 84. / (84. + 5.) * 100., 5. / (84. + 5.) * 100., 33.]]), opx), axis=0), 1)

    print(np.sum(ol[1:, 0] >= 99) / len(opx[1:, 0]))
    print(np.sum(np.sum(opx[1:, 1:3], 1) >= 99) / len(np.sum(opx[1:, 1:3], 1)))

    if "raw" not in filename:
        ol_ar, ol_h, ol_he, ol_laser = ol[1:9], ol[9:17], ol[17:22], ol[22:]
        opx_ar, opx_h, opx_he, opx_laser = opx[1:9], opx[9:15], opx[15:20], opx[20:]


def chelyabinsk_sw(y_pred: np.ndarray) -> None:
    predictions = np.round(y_pred * 100, 1)

    SD_names = np.array(["SD 0\\%", "SD 5\\%", "SD 10\\%", "SD 20\\%", "SD 30\\%", "SD 40\\%", "SD 50\\%",
                         "SD 60\\%", "SD 70\\%", "SD 80\\%", "SD 90\\%", "SD 95\\%", "SD 100\\%"])
    pred_SD = predictions[:len(SD_names), [0, 1, 2, 3, 5, 7, 9]]

    IM_names = np.array(["IM 0\\%", "IM 10\\%", "IM 20\\%", "IM 30\\%", "IM 40\\%", "IM 50\\%",
                         "IM 60\\%", "IM 70\\%", "IM 80\\%", "IM 90\\%", "IM 95\\%", "IM 100\\%"])
    pred_IM = predictions[len(SD_names):len(SD_names) + len(IM_names), [0, 1, 2, 3, 5, 7, 9]]

    SW_names = np.array(["Fresh", "SW 400", "SW 500", "SW 600", "SW 700"])
    pred_SW = predictions[len(SD_names) + len(IM_names):, [0, 1, 2, 3, 5, 7, 9]]

    for what, names in zip([pred_SD, pred_IM, pred_SW], [SD_names, IM_names, SW_names]):
        rows = len(what) * [0]

        for i, row in enumerate(what):
            tmp = np.array(["{:4.1f}".format(k1) for k1 in row], dtype="<U100")
            tmp = replace_spaces_with_phantom(tmp.ravel())

            rows[i] = stack((names[i], tmp))

        rows = np.array(rows)

        lines = [" & ".join(line) + " \\\\" for line in rows]
        for line in lines:
            print(line)


def kachr_sw(y_pred: np.ndarray) -> None:
    predictions = np.round(y_pred * 100, 1)

    Ar_names = np.array(["Fresh", "1e15 Ar$^+$\\,cm$^{-2}$", "3e15 Ar$^+$\\,cm$^{-2}$", "6e15 Ar$^+$\\,cm$^{-2}$",
                         "1e16 Ar$^+$\\,cm$^{-2}$", "2e16 Ar$^+$\\,cm$^{-2}$", "6e16 Ar$^+$\\,cm$^{-2}$",
                         "1e17 Ar$^+$\\,cm$^{-2}$"])
    OL_Ar = predictions[:len(Ar_names), :4]
    PX_Ar = predictions[34:34 + len(Ar_names), [0, 1, 2, 5]]
    Ar = stack((OL_Ar, PX_Ar), axis=1)

    H_names = np.array(["Fresh", "1e14 H$^+$\\,cm$^{-2}$", "1e15 H$^+$\\,cm$^{-2}$", "1e16 H$^+$\\,cm$^{-2}$",
                        "1e17 H$^+$\\,cm$^{-2}$", "2e17 H$^+$\\,cm$^{-2}$", "5e17 H$^+$\\,cm$^{-2}$",
                        "1e18 H$^+$\\,cm$^{-2}$"])
    OL_H = predictions[len(Ar_names):len(Ar_names) + len(H_names), :4]
    PX_H_raw = predictions[34 + len(Ar_names):34 + len(Ar_names) + len(H_names) - 2, [0, 1, 2, 5]]

    PX_H = np.zeros((np.shape(OL_H)[0], np.shape(PX_H_raw)[1]))
    PX_H[0] = PX_H_raw[0]
    PX_H[3:] = PX_H_raw[1:]
    H = stack((OL_H, PX_H), axis=1)

    He_names = np.array(["Fresh", "1e16 He$^+$\\,cm$^{-2}$", "3e16 He$^+$\\,cm$^{-2}$", "6e16 He$^+$\\,cm$^{-2}$",
                         "1e17 He$^+$\\,cm$^{-2}$"])
    OL_He = predictions[len(Ar_names) + len(H_names):len(Ar_names) + len(H_names) + len(He_names), :4]
    PX_He = predictions[34 + len(Ar_names) + len(H_names) - 2:34 + len(Ar_names) + len(H_names) - 2 + len(He_names),
            [0, 1, 2, 5]]
    He = stack((OL_He, PX_He), axis=1)

    for what, names in zip([H, He, Ar], [H_names, He_names, Ar_names]):
        rows = len(what) * [0]

        for i, row in enumerate(what):
            tmp = np.array(["{:5.1f}".format(k1) for k1 in row], dtype="<U100")
            tmp = replace_spaces_with_phantom(tmp.ravel())

            rows[i] = stack((names[i], tmp))

        rows = np.array(rows)

        lines = [" & ".join(line) + " \\\\" for line in rows]
        for line in lines:
            print(line.replace(
                "\phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0",
                "- & - & - & -"))
        print("%\n\\hdashline\n%")


def kachr_sw_laser(y_pred: np.ndarray) -> None:
    predictions = np.round(y_pred * 100, 1)

    OL_names = np.array(["Fresh", "\\phantom{00}1.7 J\\,cm$^{-2}$", "\\phantom{00}2.4 J\\,cm$^{-2}$",
                         "\\phantom{00}3.8 J\\,cm$^{-2}$", "\\phantom{00}4.6 J\\,cm$^{-2}$",
                         "\\phantom{00}6.7 J\\,cm$^{-2}$", "\\phantom{0}10.4 J\\,cm$^{-2}$",
                         "\\phantom{0}15.0 J\\,cm$^{-2}$", "\\phantom{0}23.4 J\\,cm$^{-2}$",
                         "\\phantom{0}30.6 J\\,cm$^{-2}$", "\\phantom{0}60.0 J\\,cm$^{-2}$",
                         "\\phantom{0}93.8 J\\,cm$^{-2}$", "375.0 J\\,cm$^{-2}$"])
    OL = predictions[21:34, :4]

    PX_names = np.array(["Fresh", "\\phantom{000}4.5 J\\,cm$^{-2}$", "\\phantom{000}5.6 J\\,cm$^{-2}$",
                         "\\phantom{00}12.5 J\\,cm$^{-2}$", "\\phantom{00}18.0 J\\,cm$^{-2}$",
                         "\\phantom{00}28.1 J\\,cm$^{-2}$", "\\phantom{00}36.7 J\\,cm$^{-2}$",
                         "\\phantom{00}50.0 J\\,cm$^{-2}$", "\\phantom{00}72.0 J\\,cm$^{-2}$",
                         "\\phantom{0}112.5 J\\,cm$^{-2}$", "\\phantom{0}200.0 J\\,cm$^{-2}$",
                         "\\phantom{0}450.0 J\\,cm$^{-2}$", "1800.0 J\\,cm$^{-2}$"])
    PX = predictions[-len(PX_names):, [0, 1, 2, 5]]

    for what, names in zip([OL, PX], [OL_names, PX_names]):
        rows = len(what) * [0]

        for i, row in enumerate(what):
            tmp = np.array(["{:5.1f}".format(k1) for k1 in row], dtype="<U100")
            tmp = replace_spaces_with_phantom(tmp.ravel())

            rows[i] = stack((names[i], tmp))

        rows = np.array(rows)

        lines = [" & ".join(line) + " \\\\" for line in rows]
        for line in lines:
            print(line.replace(
                "\phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0",
                "- & - & - & -"))
        print("%\n\\hdashline\n%")


def A_type_properties(y_pred: np.ndarray) -> None:
    print("Print composition of the selected A-type asteroids")

    ind_Fa, ind_Fs = num_minerals, num_minerals + endmembers_counts[0]

    filename = "".join((_path_data, "/taxonomy/asteroid_spectra-denoised-norm.npz"))
    data = np.load(filename, allow_pickle=True)  # to read the file
    ast_nums = np.array(data["metadata"][:, 0], dtype=str)

    nums = ["246", "289", "354", "1951", "5261"]
    indices = np.array([np.where(ast_nums == N)[0][0] for N in nums])

    Fa, Fs = y_pred[:, ind_Fa] * 100, y_pred[:, ind_Fs] * 100

    Fa_A, Fs_A = np.round(Fa[indices], 1), np.round(Fs[indices], 1)

    print("Fa:", "[" + ", ".join("{:.1f}".format(k) for k in Fa_A) + "]")
    print("Fs:", "[" + ", ".join("{:.1f}".format(k) for k in Fs_A) + "]")


def taxonomy_metrics(y_true: np.ndarray, y_pred: np.ndarray, latex_output: bool = False) -> None:
    target_names = classes.keys()

    if latex_output:
        text = classification_report(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1),
                                     target_names=target_names).split()

        text_cls = np.array(text[4:-15], dtype="<U100")
        text_tot = np.array(text[-5:], dtype="<U100")
        text_tot[0] = "Total"

        text_cls = np.reshape(text_cls, (int(np.round(np.size(text_cls) / 5.)), 5))
        text_tot = np.reshape(text_tot, (int(np.round(np.size(text_tot) / 5.)), 5))

        """
        # add phantom
        order = len(text_tot.ravel()[-1])
        for i, row in enumerate(text_cls):
            text_cls[i, -1] = (order - len(row[-1])) * "\\phantom{0}" + row[-1]
        """

        lines = [" & ".join(row) + " \\\\" for row in text_cls]
        lines += ["%\n\\hdashline\n%"] + [" & ".join(row) + " \\\\" for row in text_tot]

        for line in lines:
            print(line)
    else:
        print(classification_report(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), target_names=target_names))

    metric = CohenKappa(num_classes=len(classes), sparse_labels=True)
    metric.update_state(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    kappa = metric.result().numpy()

    print("Cohen's Kappa:", "{:.2f}".format(np.round(kappa, 2)))


def taxonomy_class_of_mineral_types(types: list[str], y_true: np.ndarray, y_pred: np.ndarray,
                                    meta: np.ndarray, inds_in_meta: list[int]) -> tuple[dict[str, dict[str, int]], ...]:
    winning_classes = {}
    mean_classes = {}

    for sample_type in types:
        inds = return_mineral_position(sample_type, meta, y_true, inds_in_meta)

        # mean prediction
        mean_taxonomy = np.mean(y_pred[inds], axis=0) * 100.
        inds_to_keep = mean_taxonomy > 0.
        mean_classes[sample_type] = dict(zip(np.array(list(classes.keys()))[inds_to_keep],
                                             np.round(mean_taxonomy[inds_to_keep], 1)))

        # most probable prediction
        pred_taxonomy = np.argmax(y_pred[inds], axis=1)
        pred_taxonomy = np.array([classes2[cls] for cls in pred_taxonomy])
        pred_taxonomy = np.reshape(pred_taxonomy, (len(pred_taxonomy), 1))
        winning_classes[sample_type] = dict(zip(*np.unique(pred_taxonomy, return_counts=True)))

    return mean_classes, winning_classes
