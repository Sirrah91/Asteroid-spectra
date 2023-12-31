import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tensorflow_addons.metrics import CohenKappa

from modules.NN_evaluate import spectrum_error_transfer

from modules.utilities import round_data_with_errors, normalise_array, flatten_list, stack, return_mean_std, safe_arange
from modules.utilities_spectra import find_outliers, return_mineral_position, unique_indices, used_indices, join_data
from modules.utilities_spectra import load_npz, gimme_predicted_class, compute_mean_predictions, is_taxonomical
from modules.utilities_spectra import compute_within, compute_metrics, compute_one_sigma, gimme_bin_code_from_name

from modules.NN_config_parse import gimme_num_minerals, gimme_endmember_counts, bin_to_used

from modules.NN_config_composition import mineral_names_short, endmember_names, mineral_names

from modules._constants import _label_name, _sep_out, _sep_in

# defaults only
from modules.NN_config_composition import minerals_used, endmembers_used, comp_filtering_setup, comp_data_split_setup
from modules.NN_config_taxonomy import classes


def how_many_data(y_data: np.ndarray, used_minerals: np.ndarray | None = None) -> None:
    if used_minerals is None: used_minerals = minerals_used

    # table with specific mixtures
    binary = np.array(y_data[:, :gimme_num_minerals(used_minerals)] > 0, dtype=int)
    # This keeps the same order of mixtures even if a mineral is not present (each mineral has its own base)
    base = np.array([2. ** i for i in reversed(range(len(used_minerals)))], dtype=int)[used_minerals]
    mixtures = np.sum(binary * base, axis=1)

    counts = {
        # PURE MINERALS
        "OL":np.sum(mixtures == 8),
        "OPX": np.sum(mixtures == 4),
        "CPX": np.sum(mixtures == 2),
        "PLG": np.sum(mixtures == 1),
        # BINARY MIXTURES
        "OL_OPX": np.sum(mixtures == 12),
        "OL_CPX": np.sum(mixtures == 10),
        "OL_PLG": np.sum(mixtures == 9),
        "OPX_CPX": np.sum(mixtures == 6),
        "OPX_PLG": np.sum(mixtures == 5),
        "CPX_PLG": np.sum(mixtures == 3),
        # TERNARY MIXTURES
        "OL_OPX_CPX": np.sum(mixtures == 14),
        "OL_OPX_PLG": np.sum(mixtures == 13),
        "OL_CPX_PLG": np.sum(mixtures == 11),
        "OPX_CPX_PLG": np.sum(mixtures == 7),
        # QUATERNARY MIXTURES
        "OL_OPX_CPX_PLG": np.sum(mixtures == 15)
    }

    rows = np.array([np.array([np.sort(np.round(y_data[mixtures == res, i] * 100, 1))[[0, -1]]
                               for res in [8, 4, 2, 12, 10, 6, 14]]).ravel() for i in [0, 1, 2, 3, 5, 7, 8, 9]])

    print(rows)


def accuracy_table(y_true: np.ndarray, y_pred: np.ndarray, used_minerals: np.ndarray | None = None,
                   used_endmembers: list[list[bool]] | None = None) -> None:
    print("Print accuracy metrics")

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    RMSE, R2, SAM = compute_metrics(y_true, y_pred, return_r2=True, return_sam=True, used_minerals=used_minerals,
                                    used_endmembers=used_endmembers)
    RMSE, R2, SAM = np.round(RMSE, 1), np.round(R2, 2), np.round(SAM, 1)

    print("-------------------------- ALL DATA ------------------------------")
    print(f"RMSE: [{', '.join(f'{acc:4.1f}' for acc in RMSE)}]")
    print(f"R2: [{', '.join(f'{acc:4.2f}' for acc in R2)}]")
    print(f"SAM: [{', '.join(f'{acc:4.1f}' for acc in SAM)}]")

    # no OPX-CPX outliers
    inds_to_delete = find_outliers(y_true, y_pred, used_minerals=used_minerals,
                                   used_endmembers=used_endmembers, px_only=True)
    y_true, y_pred = np.delete(y_true, inds_to_delete, axis=0), np.delete(y_pred, inds_to_delete, axis=0)

    RMSE, R2, SAM = compute_metrics(y_true, y_pred, return_r2=True, return_sam=True, used_minerals=used_minerals,
                                    used_endmembers=used_endmembers)
    RMSE, R2, SAM = np.round(RMSE, 1), np.round(R2, 2), np.round(SAM, 1)

    print("------------------------- NO OUTLIERS ----------------------------")
    print(f"RMSE: [{', '.join(f'{acc:4.1f}' for acc in RMSE)}]")
    print(f"R2: [{', '.join(f'{acc:4.2f}' for acc in R2)}]")
    print(f"SAM: [{', '.join(f'{acc:4.1f}' for acc in SAM)}]")


def quantile_table(y_true: np.ndarray, y_pred: np.ndarray,
                   used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                   latex_output: bool = False) -> None:
    print("Print quantile table")

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    error_limit = (5., 10., 15., 20.)

    within = compute_within(y_true, y_pred, error_limit=error_limit, used_minerals=used_minerals,
                            used_endmembers=used_endmembers)
    within = np.round(within, 1)

    within_all = compute_within(y_true, y_pred, error_limit=error_limit, used_minerals=used_minerals,
                                used_endmembers=used_endmembers, all_to_one=True)
    within_all = np.round(within_all, 1)

    one_sigma = compute_one_sigma(y_true, y_pred, used_minerals=used_minerals,
                                  used_endmembers=used_endmembers)
    one_sigma = np.round(one_sigma, 1)

    one_sigma_all = compute_one_sigma(y_true, y_pred, used_minerals=used_minerals,
                                      used_endmembers=used_endmembers, all_to_one=True)
    one_sigma_all = np.round(one_sigma_all, 1)

    if latex_output:
        table = np.zeros((np.shape(y_true)[1] + 1, len(error_limit)), dtype="<U100")
        table[0] = np.transpose(np.array(np.round(within_all), dtype=int))
        table[1:] = np.transpose(np.array(np.round(within), dtype=int))

        # +1 because at 0 coordinate is all data
        table = table[np.insert(unique_indices(used_minerals, used_endmembers, return_digits=True) + 1, 0, 0)]

        names = [[s + " (vol\%)" for s in mineral_names_short]] + endmember_names
        names = flatten_list(names)[unique_indices(used_minerals, used_endmembers, all_minerals=True)]
        names = np.insert(names, 0, "All data")

        """
        # replace spaces with phantom numbers
        for i, tab in enumerate(table):
            table[i] = replace_spaces_with_phantom(tab)
        """

        lines = [" & ".join(np.core.defchararray.add(line, "\\%")) + " \\\\" for line in table]

        for i, line in enumerate(lines):
            print("".join((names[i], " & " + line.replace("\\pm", " $\\pm$ "))))

            if i in [0, 3, 4, 5]:
                print("%\n\\hdashline\n%")

    else:
        print("--------------------------- ALL LABELS ---------------------------")
        for i, limit in enumerate(error_limit):
            print(f"{int(limit):3} percent: [{', '.join(f'{perc_list:5.1f}' for perc_list in within_all[i])}]")

        print("----------------------- INDIVIDUAL LABELS ------------------------")
        for i, limit in enumerate(error_limit):
            print(f"{int(limit):3} percent: [{', '.join(f'{perc_list:5.1f}' for perc_list in within[i])}]")

    print()
    print(f"1-sigma error (all): [{', '.join(f'{acc:3.1f}' for acc in one_sigma_all)}]")
    print(f"1-sigma error: [{', '.join(f'{acc:3.1f}' for acc in one_sigma)}]")


def mean_asteroid_type(y_pred: np.ndarray, used_minerals: np.ndarray | None = None,
                       used_endmembers: list[list[bool]] | None = None) -> None:
    print("Print mean composition of each asteroid type")

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    data = load_npz(f"asteroid{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz", subfolder="taxonomy")
    types = data[_label_name].ravel()

    inds = lambda x: np.array([x in ast_type for ast_type in types])
    inds_S, inds_Q, inds_V, inds_A = inds("S"), inds("Q"), inds("V"), inds("A")

    S, sigma_S = compute_mean_predictions(y_pred[inds_S])
    Q, sigma_Q = compute_mean_predictions(y_pred[inds_Q])
    V, sigma_V = compute_mean_predictions(y_pred[inds_V])
    A, sigma_A = compute_mean_predictions(y_pred[inds_A])

    S, sigma_S = round_data_with_errors(S, sigma_S)
    Q, sigma_Q = round_data_with_errors(Q, sigma_Q)
    V, sigma_V = round_data_with_errors(V, sigma_V)
    A, sigma_A = round_data_with_errors(A, sigma_A)

    val_sigma = lambda val, sigma: ["".join((f"{v:4.1f}", "\\pm", f"{s:4.1f}"))
                                    for v, s in zip(np.round(val, 1), np.round(sigma, 1))]

    tmp = np.array(stack((val_sigma(S, sigma_S),
                          val_sigma(Q, sigma_Q),
                          val_sigma(V, sigma_V),
                          val_sigma(A, sigma_A)), axis=1), dtype="<U100")

    """
    # replace spaces with phantom numbers
    for i, tab in enumerate(tmp):
        tmp[i] = replace_spaces_with_phantom(tab)
    """

    lines = [" & ".join(line) + " \\\\" for line in tmp]

    names = [[f"{name} (vol\%)" for name in mineral_names_short]] + endmember_names
    names = flatten_list(names)[used_indices(used_minerals, used_endmembers)]

    for i, line in enumerate(lines):
        print("".join((names[i], " & " + line.replace("\\pm", " $\\pm$ "))))
        if i in [2, 4, 6]:
            print("%\n\\hdashline\n%")


def mean_S_asteroid_type(y_pred: np.ndarray, used_minerals: np.ndarray | None = None,
                         used_endmembers: list[list[bool]] | None = None) -> None:
    print("Print mean composition of each asteroid type")

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    data = load_npz(f"asteroid{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz", subfolder="taxonomy")
    types = np.ravel(data[_label_name])

    inds_S = np.array(["S" in ast_type for ast_type in types])
    unique, table = np.unique(types[inds_S], return_counts=True)

    inds_order = [0, -1, 2, 4, 5, 6, 7, 8, 1]
    table = table[inds_order]
    table = np.reshape(table, (1, len(table)))

    val_sigma = lambda val, sigma: ["".join((f"{v:4.1f}", "\\pm", f"{s:4.1f}"))
                                    for v, s in zip(np.round(val, 1), np.round(sigma, 1))]

    for c, i in enumerate(inds_order):
        inds = types == unique[i]
        column, sigma_column = compute_mean_predictions(y_pred[inds])
        column, sigma_column = round_data_with_errors(column, sigma_column)

        if c == 0:
            table_tmp = np.array([val_sigma(column, sigma_column)], dtype="<U100")
        else:
            table_tmp = stack((table_tmp, np.array([val_sigma(column, sigma_column)])), axis=0)

    table = stack((table, np.transpose(table_tmp)), axis=0)

    """
    # replace spaces with phantom numbers
    for i, tab in enumerate(table):
        table[i] = replace_spaces_with_phantom(tab)
    """

    lines = [" & ".join(line) + " \\\\" for line in table]

    names = [[f"{name} (vol\%)" for name in mineral_names_short]] + endmember_names
    names = flatten_list(names)[used_indices(used_minerals, used_endmembers)]
    names = np.insert(names, 0, "Number")

    for i, line in enumerate(lines):
        print("".join((names[i], " & ", line.replace("\\pm", " $\\pm$ "))))
        if i in [0, 3, 5, 7]:
            print("%\n\\hdashline\n%")


def chelyabinsk_composition() -> None:
    densities = np.array([4.39, 3.27, 3.95, 3.20, 2.90, 3.95, 3.20, 2.90, 2.73, 2.62, 2.56])
    modal = np.array([56.2, 26.9, 0., 0.])

    chemical = np.array([28.6, 71.4, 23.5, 74.9, 1.6, 0, 0, 0, 8.5, 84.5, 7])
    mineral_density = chemical * densities
    ol = np.sum(mineral_density[:2], keepdims=True)
    opx = np.sum(mineral_density[2:5], keepdims=True)
    cpx = np.sum(mineral_density[5:8], keepdims=True)
    plg = np.sum(mineral_density[8:], keepdims=True)

    mineral_density = stack((ol, opx, cpx, plg))
    mineral_density[mineral_density == 0] = 1.

    modal /= mineral_density
    modal = normalise_array(modal, norm_constant=100.)

    # normalise Wo (CPX) out, but is still in the array...
    chemical[2:4] = normalise_array(chemical[2:4], norm_constant=100.)

    composition = stack((modal, chemical))
    print(np.round(composition, 1))


def kachr_experiment(y_pred: np.ndarray, raw: bool = False) -> None:
    ol_stop = 24 if raw else 34

    ol = y_pred[:ol_stop, [0, 1, 2, 3]] * 100.
    opx = y_pred[ol_stop:, [0, 1, 2, 5]] * 100.

    ol = np.round(stack((np.array([[100., 0., 0., 9.9]]), ol), axis=0), 1)
    opx = np.round(stack((np.array([[0., 84. / (84. + 5.) * 100., 5. / (84. + 5.) * 100., 33.]]), opx), axis=0), 1)

    print(np.sum(ol[1:, 0] >= 99) / len(opx[1:, 0]))
    print(np.sum(np.sum(opx[1:, 1:3], 1) >= 99) / len(np.sum(opx[1:, 1:3], 1)))

    if raw:
        ol_ar, ol_h, ol_he, ol_laser = ol[1:9], ol[9:17], ol[17:22], ol[22:]
        opx_ar, opx_h, opx_he, opx_laser = opx[1:9], opx[9:15], opx[15:20], opx[20:]


def chelyabinsk_sw(y_pred: np.ndarray, used_minerals: np.ndarray | None = None,
                   used_endmembers: list[list[bool]] | None = None) -> None:
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    predictions = np.round(y_pred * 100, 1)
    inds = unique_indices(used_minerals, used_endmembers, return_digits=True)

    SD_names = np.array(["SD 0\\%", "SD 5\\%", "SD 10\\%", "SD 20\\%", "SD 30\\%", "SD 40\\%", "SD 50\\%",
                         "SD 60\\%", "SD 70\\%", "SD 80\\%", "SD 90\\%", "SD 95\\%", "SD 100\\%"])
    pred_SD = predictions[:len(SD_names), inds]

    IM_names = np.array(["IM 0\\%", "IM 10\\%", "IM 20\\%", "IM 30\\%", "IM 40\\%", "IM 50\\%",
                         "IM 60\\%", "IM 70\\%", "IM 80\\%", "IM 90\\%", "IM 95\\%", "IM 100\\%"])
    pred_IM = predictions[len(SD_names):len(SD_names) + len(IM_names), inds]

    SW_names = np.array(["Fresh", "SW 400", "SW 500", "SW 600", "SW 700"])
    pred_SW = predictions[len(SD_names) + len(IM_names):, inds]

    for what, names in zip([pred_SD, pred_IM, pred_SW], [SD_names, IM_names, SW_names]):
        rows = len(what) * [0]

        for i, row in enumerate(np.round(what, 1)):
            tmp = np.array([f"{val:4.1f}" for val in row], dtype="<U100")
            # tmp = replace_spaces_with_phantom(tmp.ravel())

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

        for i, row in enumerate(np.round(what, 1)):
            tmp = np.array([f"{val:5.1f}" for val in row], dtype="<U100")
            # tmp = replace_spaces_with_phantom(tmp.ravel())

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

        for i, row in enumerate(np.round(what, 1)):
            tmp = np.array([f"{val:5.1f}" for val in row], dtype="<U100")
            # tmp = replace_spaces_with_phantom(tmp.ravel())

            rows[i] = stack((names[i], tmp))

        rows = np.array(rows)

        lines = [" & ".join(line) + " \\\\" for line in rows]
        for line in lines:
            print(line.replace(
                "\phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0 & \phantom{0}\phantom{0}0.0",
                "- & - & - & -"))
        print("%\n\\hdashline\n%")


def A_type_properties(y_pred: np.ndarray, used_minerals: np.ndarray | None = None,
                      used_endmembers: list[list[bool]] | None = None) -> None:
    print("Print composition of the selected A-type asteroids")

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    num_minerals = gimme_num_minerals(used_minerals)
    count_endmembers = gimme_endmember_counts(used_endmembers)

    ind_Fa, ind_Fs = num_minerals, num_minerals + count_endmembers[0]

    data = load_npz(f"asteroid{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz", subfolder="taxonomy")

    metadata = join_data(data, "meta")
    ast_nums = np.array(metadata["asteroid number"], dtype=str)

    nums = ["246", "289", "354", "1951", "5261"]
    indices = np.array([np.where(ast_nums == N)[0][0] for N in nums])

    Fa, Fs = y_pred[:, ind_Fa] * 100, y_pred[:, ind_Fs] * 100

    Fa_A, Fs_A = np.round(Fa[indices], 1), np.round(Fs[indices], 1)

    print(f"Fa: [{', '.join(f'{val:.1f}' for val in Fa_A)}]")
    print(f"Fs: [{', '.join(f'{val:.1f}' for val in Fs_A)}]")


def taxonomy_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     used_classes: dict[str, int] | list | np.ndarray | None = None,
                     latex_output: bool = False) -> None:
    if used_classes is None: used_classes = classes

    if isinstance(used_classes, dict):
        labels = np.array(list(used_classes.keys()))
    else:
        labels = np.array(used_classes)

    text = classification_report(gimme_predicted_class(y_true, used_classes=labels, return_index=True),
                                 gimme_predicted_class(y_pred, used_classes=labels, return_index=True),
                                 target_names=labels)

    if latex_output:
        text = text.split()

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
        print(text)

    metric = CohenKappa(num_classes=len(labels), sparse_labels=True)
    metric.update_state(gimme_predicted_class(y_true, used_classes=labels, return_index=True),
                        gimme_predicted_class(y_pred, used_classes=labels, return_index=True))
    kappa = metric.result().numpy()

    print("Cohen's Kappa:", f"{np.round(kappa, 2):.2f}")


def taxonomy_class_of_mineral_types(types: list[str] | np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                                    meta: pd.DataFrame, used_classes: dict[str, int] | list | np.ndarray | None = None
                                    ) -> tuple[dict[str, dict[str, int]], ...]:
    if used_classes is None: used_classes = classes

    if isinstance(used_classes, dict):
        labels = np.array(list(used_classes.keys()))
    else:
        labels = np.array(used_classes)

    winning_classes = {}
    mean_classes = {}

    for sample_type in types:
        inds = return_mineral_position(sample_type, meta, y_true)

        # mean prediction
        mean_taxonomy = np.mean(y_pred[inds], axis=0) * 100.
        inds_to_keep = mean_taxonomy > 0.
        mean_classes[sample_type] = dict(zip(labels[inds_to_keep], np.round(mean_taxonomy[inds_to_keep], 1)))

        # most probable prediction
        pred_taxonomy = gimme_predicted_class(y_pred, used_classes=labels)
        pred_taxonomy = np.reshape(pred_taxonomy, (len(pred_taxonomy), 1))
        winning_classes[sample_type] = dict(zip(*np.unique(pred_taxonomy, return_counts=True)))

    return mean_classes, winning_classes


def print_grid_test_stats_norm_window(x: np.ndarray, y: np.ndarray, used_minerals: np.ndarray | None = None,
                                      used_endmembers: list[list[bool]] | None = None) -> None:
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    names = stack((mineral_names, flatten_list(endmember_names)))
    titles_all = names[unique_indices(used_minerals=used_minerals, used_endmembers=used_endmembers, all_minerals=True)]

    name = "Window" if isinstance(x[0], str) else "Normalisation"

    print(f"{name} minimum: {x[np.argmin(y[0])]}, {np.round(np.min(y[0]), 1)}")
    print(f"{name} maximum: {x[np.argmax(y[0])]}, {np.round(np.max(y[0]), 1)}")
    print(f"{name} mean: {np.round(np.mean(y[0]), 1)}")

    norm = np.mean(y[1:], axis=1)
    print(f"{name} minimum: {titles_all[np.argmin(norm)].upper()}, {np.round(np.min(norm), 1)}")
    print(f"{name} maximum: {titles_all[np.argmax(norm)].upper()}, {np.round(np.max(norm), 1)}")


def print_grid_test_stats_step(y: np.ndarray,used_minerals: np.ndarray | None = None,
                               used_endmembers: list[list[bool]] | None = None) -> None:
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    names = stack((mineral_names, flatten_list(endmember_names)))
    titles_all = names[unique_indices(used_minerals=used_minerals, used_endmembers=used_endmembers, all_minerals=True)]

    curves = np.array(["650--1850 nm", "650--2450 nm", "ASPECT 650--1600 nm", "ASPECT 650--2450 nm"])

    print("No ASPECT minimum", np.round(np.min((y[0][0], y[1][0])), 1))
    print("No ASPECT maximum", np.round(np.max((y[0][0], y[1][0])), 1))
    print("ASPECT minimum", np.round(np.min((y[2][0], y[3][0])), 1))
    print("ASPECT maximum", np.round(np.max((y[2][0], y[3][0])), 1))

    # 1 + i to skip "All"
    trend = np.array([[np.polyfit(np.arange(10, 51, 10), error[1 + i], deg=1)[0] for i in range(8)] for error in y])
    const = np.array([[np.polyfit(np.arange(10, 51, 10), error[1 + i], deg=1)[1] for i in range(8)] for error in y])

    def print_trends(which: str) -> None:
        if which == "max":
            func = np.max
        else:
            func = np.min

        for resolution in ["ideal", "aspect"]:
            if resolution == "ideal":
                trend_eval = trend[:2]
                const_use = const[:2]
                curves_use = curves[:2]
            else:
                trend_eval = trend[2:]
                const_use = const[2:]
                curves_use = curves[2:]

            icurve, ititle = np.where(trend_eval == func(trend_eval))
            quantity = titles_all[ititle][0]
            curve = curves_use[icurve][0]

            print(f"Trend {which}: {quantity}, {curve}, {np.round(func(trend_eval * 10), 2)}")
            print(f"Trend {which} avg: {np.round(np.diff(np.polyval(np.concatenate((trend_eval[icurve, ititle], const_use[icurve, ititle])), [10, 50]))[0], 1)}")

    print_trends("max")
    print_trends("min")


def print_grid_test_stats_range(error_mat: np.ndarray, lim_from: tuple[int, ...] | int, lim_to: tuple[int, ...] | int,
                                start: np.ndarray | None = None, stop: np.ndarray | None = None,
                                name: str = "", quiet: bool = False) -> tuple[float, ...]:

    if start is None: start = safe_arange(450, 2350, 100, endpoint=True, dtype=int)

    if stop is None: stop = safe_arange(550, 2450, 100, endpoint=True, dtype=int)

    if isinstance(lim_from, tuple | list):
        if isinstance(lim_to, tuple | list):
            mat_cut = error_mat[np.where(stop == lim_to[0])[0][0]:np.where(stop == lim_to[1])[0][0] + 1,
                      np.where(start == lim_from[0])[0][0]:np.where(start == lim_from[1])[0][0] + 1]
        else:  # lim_to is bounded from bottom only
            mat_cut = error_mat[np.where(stop == lim_to)[0][0]:,
                      np.where(start == lim_from[0])[0][0]:np.where(start == lim_from[1])[0][0] + 1]
    else:  # lim_from is bounded from above only
        if isinstance(lim_to, tuple | list):
            mat_cut = error_mat[np.where(stop == lim_to[0])[0][0]:np.where(stop == lim_to[1])[0][0] + 1,
                      :np.where(start == lim_from)[0][0] + 1]
        else:  # lim_to is bounded from bottom only
            mat_cut = error_mat[np.where(stop == lim_to)[0][0]:,
                      :np.where(start == lim_from)[0][0] + 1]

    mn, std = return_mean_std(mat_cut)
    if not quiet:
        print(f"{name} mean ± std: {np.round(mn, 1)} ± {np.round(std, 1)}")

    return mn, std


def print_range_test_table(error_mat: np.ndarray, start: int, stop: int):
    mns, _ = list(zip(*[print_grid_test_stats_range(rng, lim_from=(start, start), lim_to=(stop, stop), quiet=True)
                        for rng in error_mat[1:]]))

    print(f"{start}\u2013{stop} nm")
    print(" & ".join((f"{np.round(mn, 1):.1f}" for mn in mns)), "\\\\")


def print_grid_test_significance_table(norm: np.ndarray, error_mat: np.ndarray, window: np.ndarray, step: np.ndarray,
                                       error_type: str, used_minerals: np.ndarray | None = None,
                                       used_endmembers: list[list[bool]] | None = None,
                                       latex_output: bool = False) -> None:
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    mask = unique_indices(used_minerals=used_minerals, used_endmembers=used_endmembers)
    names = stack((mineral_names_short, flatten_list(endmember_names)))
    names = names[unique_indices(used_minerals=used_minerals, used_endmembers=used_endmembers, all_minerals=True)]

    indices = np.array(stack((np.where("OL" == names)[0], np.where("OPX" == names)[0],
                              np.where("Fa" == names)[0], np.where("Fs (OPX)" == names)[0])), dtype=int)

    _, stds = print_model_to_model_variations(quiet=True)
    std_rmse, std_within = stds

    std_model = std_rmse if error_type == "rmse" else std_within

    _, stds = list(zip(*[print_grid_test_stats_range(error_mat[indices[0] + 1], lim_from=(450, 750), lim_to=(1050, 1250), quiet=True),
                         print_grid_test_stats_range(error_mat[indices[1] + 1], lim_from=(450, 750), lim_to=(1050, 1250), quiet=True),
                         print_grid_test_stats_range(error_mat[indices[2] + 1], lim_from=(450, 750), lim_to=(1550, 1750), quiet=True),
                         print_grid_test_stats_range(error_mat[indices[3] + 1], lim_from=(450, 750), lim_to=(1350, 1750), quiet=True)]))

    std_range = np.zeros(len(std_rmse))
    std_range[indices] = stds

    std_model, std_range = std_model[mask], std_range[mask]

    # already masked
    _, stds = list(zip(*[return_mean_std(quantity, axis=1) for i, quantity in enumerate([norm, window, step])]))

    std_norm, std_window, std_step = stds
    std_norm, std_window, std_step = std_norm[1:], std_window[1:], std_step[1:]  # remove "All"

    if not latex_output:
        print("       ", "".join(f'{name.replace("(OPX)", "").replace("(CPX)", ""):5}' for name in names))
        print(f"Norm.:  {', '.join([f'{sigma:.1f}' for sigma in np.round(std_norm / std_model, 1)])}")
        print(f"Range:  {', '.join([f'{sigma:.1f}' for sigma in np.round(std_range / std_model, 1)])}")
        print(f"Window: {', '.join([f'{sigma:.1f}' for sigma in np.round(std_window / std_model, 1)])}")
        print(f"Step:   {', '.join([f'{sigma:.1f}' for sigma in np.round(std_step / std_model, 1)])}")
    else:
        print("&", f"{' & '.join([f'{sigma:.1f}' for sigma in np.round(std_norm / std_model, 1)[indices]])}", "\\\\")
        print("-" * 50)
        print("&", f"{' & '.join([f'{sigma:.1f}' for sigma in np.round(std_range / std_model, 1)[indices]])}", "\\\\")
        print("-" * 50)
        print("&", f"{' & '.join([f'{sigma:.1f}' for sigma in np.round(std_window / std_model, 1)[indices]])}", "\\\\")
        print("-" * 50)
        print("&", f"{' & '.join([f'{sigma:.1f}' for sigma in np.round(std_step / std_model, 1)[indices]])}", "\\\\")


def print_grid_test_stats_table(norm: np.ndarray, error_mat: np.ndarray, window: np.ndarray,
                                step: np.ndarray, used_minerals: np.ndarray | None = None,
                                used_endmembers: list[list[bool]] | None = None,
                                latex_output: bool = False) -> None:
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    names = stack((mineral_names, flatten_list(endmember_names)))
    titles_all = names[unique_indices(used_minerals=used_minerals, used_endmembers=used_endmembers, all_minerals=True)]
    titles_all = stack((["All"], titles_all))

    indices = np.array(stack((np.where("olivine" == titles_all)[0], np.where("orthopyroxene" == titles_all)[0],
                              np.where("Fa" == titles_all)[0], np.where("Fs (OPX)" == titles_all)[0])), dtype=int)

    data_names = ["Normalisation", "Range", "Window", "Step"]
    print("-" * 50)

    mns, stds = list(zip(*[print_grid_test_stats_range(error_mat[indices[0]], lim_from=(650, 650), lim_to=(1850, 1850), quiet=True),
                           print_grid_test_stats_range(error_mat[indices[1]], lim_from=(650, 650), lim_to=(1850, 1850), quiet=True),
                           print_grid_test_stats_range(error_mat[indices[2]], lim_from=(650, 650), lim_to=(1850, 1850), quiet=True),
                           print_grid_test_stats_range(error_mat[indices[3]], lim_from=(650, 650), lim_to=(1850, 1850), quiet=True)]))
    mn_range = np.zeros(len(titles_all))
    std_range = np.zeros(len(titles_all))
    mn_range[indices] = mns
    std_range[indices] = stds

    for i, quantity in enumerate([norm, error_mat, window, step]):
        if quantity is error_mat:
            mn, std = mn_range, std_range
        else:
            mn, std = return_mean_std(quantity, axis=1)

        if not latex_output:
            for index in indices:  # OL, OPX, Fa, Fs (OPX)
                print(f"{data_names[i]} {titles_all[index]:14}: "
                      f"{np.round(mn[index], 1)} ± {np.round(std[index], 1)}")
        else:

            std_to_print = [f"{np.round(s, 1):.1f}" for s in std]
            std_to_print = np.array([s.replace("0.", "").replace(".", "") for s in std_to_print])

            print("&", " & ".join([f"{np.round(mn[index], 1):.1f}({std_to_print[index]})" for index in indices]), "\\\\")
        print("-" * 50)


def print_model_to_model_variations(quiet: bool = False) -> tuple[np.ndarray, ...]:
    filenames = ["/home/dakorda/Python/NN/accuracy_tests/range_test/normalisation/composition_450-2450-10-550_1110-11-110-111-000_20230704150150.npz",
                 "/home/dakorda/Python/NN/accuracy_tests/range_test/range/composition_450-2450-10-550_1110-11-110-111-000_20230709032616.npz",
                 "/home/dakorda/Python/NN/accuracy_tests/composition_450-2450-10-550_1110-11-110-111-000_20231016111014.npz",
                 "/home/dakorda/Python/NN/accuracy_tests/composition_450-2450-10-550_1110-11-110-111-000_20231016111656.npz",
                 "/home/dakorda/Python/NN/accuracy_tests/composition_450-2450-10-550_1110-11-110-111-000_20231016153911.npz",
                 "/home/dakorda/Python/NN/accuracy_tests/composition_450-2450-10-550_1110-11-110-111-000_20231016153728.npz",
                 "/home/dakorda/Python/NN/accuracy_tests/composition_450-2450-10-550_1110-11-110-111-000_20231016193822.npz",
                 "/home/dakorda/Python/NN/accuracy_tests/composition_450-2450-10-550_1110-11-110-111-000_20231016194425.npz",
                 "/home/dakorda/Python/NN/accuracy_tests/composition_450-2450-10-550_1110-11-110-111-000_20231016234622.npz",
                 "/home/dakorda/Python/NN/accuracy_tests/composition_450-2450-10-550_1110-11-110-111-000_20231016235128.npz"]

    used_minerals, used_endmembers = bin_to_used(gimme_bin_code_from_name(filenames[0]))

    names = ["RMSE", "Within"]

    rmse = []
    within = []

    for filename in filenames:
        data = load_npz(filename)
        y_true, y_pred = data["labels_true"], data["labels_predicted"]

        tmp, = compute_metrics(y_true, y_pred, remove_px_outliers=True, used_minerals=used_minerals,
                               used_endmembers=used_endmembers)
        rmse.append(tmp)

        tmp, = compute_within(y_true, y_pred, error_limit=(10.,), remove_px_outliers=True, used_minerals=used_minerals,
                              used_endmembers=used_endmembers)
        within.append(tmp)

    rmse = np.array(rmse)
    within = np.array(within)

    indices = unique_indices(used_minerals, used_endmembers)
    mns, stds = np.zeros((2, len(rmse))), np.zeros((2, len(rmse)))

    for i, quantity in enumerate([rmse, within]):
        mn, std = return_mean_std(quantity, axis=0)
        mns[i], stds[i] = mn, std

        if not quiet:
            mn, std, prec = round_data_with_errors(mn, std, return_precision=True)
            mn, std, prec = mn[indices], std[indices], prec[indices]
            prec[prec < 0] = 0

            std_to_print = np.array(np.round(std * 10**prec), dtype=int)
            print(f"{names[i]} &", " & ".join([f'{m:.{n}f}({s})' for m, s, n in zip(mn, std_to_print, prec)]), "\\\\")

    return mns, stds


def ASPECT_metrics_variations(snrs: tuple[float, ...],
                              filtering_setup: dict | None = None,
                              data_split_setup: dict | None = None) -> None:
    if filtering_setup is None: filtering_setup = comp_filtering_setup
    if data_split_setup is None: data_split_setup = comp_data_split_setup

    n_trials = 200
    rnd_seed = 42

    model_names = ["/home/dakorda/Python/NN/models/composition/ASPECT_vis-nir1-nir2_30/CNN_ASPECT_vis-nir1-nir2_30_1110-11-110-111-000_20231015114247.h5",
                   "/home/dakorda/Python/NN/models/composition/ASPECT_vis-nir1-nir2-swir_30/CNN_ASPECT_vis-nir1-nir2-swir_30_1110-11-110-111-000_20231015114247.h5"]

    if is_taxonomical(model=model_names[0]):
        raise ValueError("Only composition models are allowed at this moment.")

    used_minerals, used_endmembers = bin_to_used(gimme_bin_code_from_name(model_names[0]))

    to_print = []

    indices = unique_indices(used_minerals, used_endmembers)

    for model_name in model_names:
        for snr in snrs:
            res = spectrum_error_transfer(model_name, snr=snr,  filtering_setup=filtering_setup,
                                          data_split_setup=data_split_setup, n_trials=n_trials, rnd_seed=rnd_seed)

            for quantity in ["rmse", "within"]:
                if quantity == "rmse":
                    mn = res["RMSE mean noisy (pp)"] - res["RMSE mean (pp)"]
                    std = res["RMSE std noisy (pp)"]
                    name = "$\\Delta$~RMSE"
                else:
                    mn = res["within 10 pp mean noisy (pp)"] - res["within 10 pp mean (%)"]
                    std = res["within 10 pp std noisy (pp)"]
                    name = "$\\Delta$~Within"

                mn, std, prec = round_data_with_errors(mn, std, return_precision=True)
                mn, std, prec = mn[indices], std[indices], prec[indices]
                prec[prec < 0] = 0

                std_to_print = np.array(np.round(std * 10 ** prec), dtype=int)
                to_print.append(" ".join((f"& {name} &", " & ".join([f'{m:.{n}f}({s})' for m, s, n in zip(mn, std_to_print, prec)]), "\\\\")))

        to_print.append("-" * 100)

    print("\n".join(to_print))
