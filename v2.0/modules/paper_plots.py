from modules.control_plots import *

from typing import Callable
import re
import numpy as np
import matplotlib.patches as patches
import cv2
import os.path

from modules.BAR_BC_method import calc_BAR_BC, calc_composition, filter_data_mask
from modules.NN_data import load_compositional_data as load_data
from modules.NN_data import split_compositional_data_proportional as split_data_proportional
from modules.NN_data import split_meta_proportional

from modules.utilities import check_dir, best_blk, normalise_in_rows, distance, stack
from modules.utilities_spectra import combine_same_range_models, error_estimation_bin_like, cut_error_bars
from modules.utilities_spectra import used_indices, unique_indices

from modules.NN_config import num_minerals, minerals_used, endmembers_counts, endmembers_used
from modules.NN_config_taxonomy import classes, classes2

from modules._constants import _path_data, _path_asteroid_images


def plot_ast_PC1_PC2(y_pred: np.ndarray) -> None:
    print("Plot PC1 vs PC2 for asteroids")

    cmap = "viridis_r"  # cmap of points
    s = 30.  # scaling parameter (marker size)
    vmin, vmax = 0., 100.  # minimum and maximum for colormap in scatter of chemical

    # annotation
    start, end = np.array([0.5, -0.5]), np.array(([0.0, 0.8]))
    shift = 0.03

    filename = "".join((_path_data, "/taxonomy/asteroid_spectra-denoised-norm.npz"))
    data = np.load(filename, allow_pickle=True)  # to read the file
    types = np.array(data["metadata"][:, 1], dtype=str)

    inds_S = np.array(["S" in ast_type for ast_type in types])
    inds_Q = np.array(["Q" in ast_type for ast_type in types])
    inds_A = np.array(["A" in ast_type for ast_type in types])
    inds_V = np.array(["V" in ast_type for ast_type in types])

    inds = inds_S + inds_Q + inds_V + inds_A

    # ignore Sq: since its spectrum is very weird
    inds[types == "Sq:"] = False

    unique, counts = np.unique(types[inds], return_counts=True)

    PCA = np.array(data["metadata"][inds, 3:5], dtype=np.float64)
    predictions = y_pred[inds] * 100.

    labels = np.core.defchararray.add("$", types[inds])
    labels = np.core.defchararray.add(labels, "$")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Why I cannot set marker=labels and do it all as for vectors??
    for pca, label, pred in zip(PCA, labels, predictions):
        if len(label) == 3:
            fact = 1.
        if len(label) == 4:
            fact = 2.5
        if len(label) == 5:
            fact = 5.5

        c = pred[0]
        sp = ax.scatter(pca[1], pca[0], c=c, cmap=cmap, vmin=vmin, vmax=vmax, s=s * fact, marker=label)

    ax.set_xlabel("PC2'")
    ax.set_ylabel("PC1'")

    # annotation
    x, y = np.mean(stack((start, end), axis=0), axis=0) + shift

    arrowprops = dict(arrowstyle="->", color="r", facecolor="black", lw=3)
    text_kwargs = {"horizontalalignment": "center", "verticalalignment": "center", "fontsize": 16,
                   "transform": ax.transData, "c": "r",
                   "rotation": np.degrees(np.arctan((end[1] - start[1]) / (end[0] - start[0]))),
                   "transform_rotates_text": True}

    ax.annotate("", xy=end, xytext=start, zorder=-1, arrowprops=arrowprops)
    ax.text(x=x, y=y, s="Space weathering", **text_kwargs)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(**cbar_kwargs)
    cbar = plt.colorbar(sp, ax=ax, cax=cax)
    cbar.ax.set_ylabel("Olivine abundance (vol\%)")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.draw()

    fig_name = "".join(("PCA_plot.", fig_format))
    fig.savefig("".join((outdir_composition, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_PCA_BAR() -> None:
    cmap = "viridis_r"  # cmap of points
    s = 30.  # scaling parameter (marker size)
    vmin, vmax = 0., 100.  # minimum and maximum for colormap in scatter of chemical

    # annotation
    start, end = np.array([0.5, -0.5]), np.array(([0.0, 0.8]))
    shift = 0.03

    filename_data = "asteroid_spectra-denoised-norm.npz"
    data_file = "".join((_path_data, filename_data))
    data = np.load(data_file, allow_pickle=True)

    meta, wvl, data = data["metadata"], data["wavelengths"], data["spectra"]

    types = np.array(meta[:, 1], dtype=str)

    inds_S = np.array(["S" in ast_type for ast_type in types])
    inds_Q = np.array(["Q" in ast_type for ast_type in types])
    inds_A = np.array(["A" in ast_type for ast_type in types])
    inds_V = np.array(["V" in ast_type for ast_type in types])

    inds = inds_S + inds_Q + inds_V + inds_A

    # ignore Sq: since its spectrum is very weird
    inds[types == "Sq:"] = False

    types = types[inds]

    x_data = data[inds]
    PCA = np.array(meta[inds, 3:5], dtype=np.float64)

    labels = np.core.defchararray.add("$", types)
    labels = np.core.defchararray.add(labels, "$")

    BAR, BIC, BIIC = calc_BAR_BC(wvl, x_data)

    # remove nans
    mask = np.logical_and(np.logical_and(BAR > 0, BIC > 0), BIIC > 0)
    BAR, BIC, BIIC = BAR[mask], BIC[mask], BIIC[mask]
    labels = labels[mask]
    types = types[mask]
    PCA = PCA[mask]

    OL_fraction, Fs, Wo = calc_composition(BAR, BIC, BIIC, types, method="biic")

    # filter the data
    mask = filter_data_mask(OL_fraction, Fs, Wo)
    OL_fraction = OL_fraction[mask]
    labels = labels[mask]
    types = types[mask]
    PCA = PCA[mask]

    unique, counts = np.unique(types, return_counts=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Why I cannot set marker=labels and do it all as for vectors??
    for pca, label, OL_frac in zip(PCA, labels, OL_fraction):
        if len(label) == 3:
            fact = 1
        if len(label) == 4:
            fact = 2.5
        if len(label) == 5:
            fact = 5.5

        c = OL_frac
        sp = ax.scatter(pca[1], pca[0], c=c, cmap=cmap, vmin=vmin, vmax=vmax, s=s * fact, marker=label)

    ax.set_xlabel("PC2'")
    ax.set_ylabel("PC1'")

    # annotation
    x, y = np.mean(stack((start, end), axis=0), axis=0) + shift

    arrowprops = dict(arrowstyle="->", color="r", facecolor="black", lw=3)
    text_kwargs = {"horizontalalignment": "center", "verticalalignment": "center", "fontsize": 16,
                   "transform": ax.transData, "c": "r",
                   "rotation": np.degrees(np.arctan((end[1] - start[1]) / (end[0] - start[0]))),
                   "transform_rotates_text": True}

    ax.annotate("", xy=end, xytext=start, zorder=-1, arrowprops=arrowprops)
    ax.text(x=x, y=y, s="Space weathering", **text_kwargs)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(**cbar_kwargs)
    cbar = plt.colorbar(sp, ax=ax, cax=cax)
    cbar.ax.set_ylabel("Olivine abundance (vol\%)")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.draw()

    fig_name = "".join(("PCA_plot_BAR_BC.", fig_format))
    fig.savefig("".join((outdir_composition, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_Fa_vs_Fs_ast_only() -> None:
    print("Plot Fa vs Fs")

    limx1, limx2 = 15., 35.
    limy1, limy2 = 10., 30.

    shift = 1.  # Control ranges of axes
    s = 50.  # scaling parameter

    xticks, yticks = safe_arange(limx1, limx2, 5., endpoint=True), safe_arange(limy1, limy2, 5., endpoint=True)
    left, right = limx1 - shift, limx2 + shift
    bottom, top = limy1 - shift, limy2 + shift

    #####
    # NEED TO BE SET ACCORDING TO MODEL PREDICTIONS
    Fa_S, Fs_S, sigma_fas, sigma_fss = [20.9], [18.3], 4.6, 4.9
    Fa_Sq, Fs_Sq, sigma_fasq, sigma_fssq = [23.3], [21.0], 4.6, 4.1
    Fa_Sr, Fs_Sr, sigma_fasr, sigma_fssr, = [18.5], [19.2], 5.8, 5.4
    Fa_Sw, Fs_Sw, sigma_fasw, sigma_fssw = [21.6], [14.7], 4.3, 4.3

    Fa_Q, Fs_Q, sigma_faq, sigma_fsq = [26.5], [23.8], 5.5, 5.2
    #####

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # definition of boxes from (for some reasons should be used just once)
    # https://www.researchgate.net/figure/Mineral-compositional-data-for-the-Morokweng-meteoriteDiagram-shows-ferrosilite-Fs_fig3_7093505
    H_rect = patches.Rectangle((16.2, 14.5), 3.8, 3.5, linewidth=1.5, edgecolor="r", facecolor="none")
    L_rect = patches.Rectangle((22.0, 19.0), 4.0, 3.0, linewidth=1.5, edgecolor="g", facecolor="none")
    LL_rect = patches.Rectangle((26.0, 22.0), 6.0, 4.2, linewidth=1.5, edgecolor="b", facecolor="none")

    ax.set_xlabel("Fa")
    ax.set_ylabel("Fs (OPX)")
    ax.tick_params(axis="both")
    ax.axis("square")
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xlim(left=left, right=right)
    ax.set_ylim(bottom=bottom, top=top)

    # add the patches
    ax.add_patch(H_rect)
    ax.add_patch(L_rect)
    ax.add_patch(LL_rect)

    # artificial data to get labels for legend
    ax.plot(120, 120, "rs", label="H", markerfacecolor="none")
    ax.plot(120, 120, "gs", label="L", markerfacecolor="none")
    ax.plot(120, 120, "bs", label="LL", markerfacecolor="none")

    ax.legend(loc="upper left")

    ax.scatter(Fa_S, Fs_S, marker="$S$", c="k", s=s * 2.5, zorder=100)
    ax.errorbar(Fa_S, Fs_S, xerr=sigma_fas, yerr=sigma_fss, c="c", ls="None")
    ax.scatter(Fa_Sq, Fs_Sq, marker="$Sq$", c="k", s=s * 5, zorder=100)
    ax.errorbar(Fa_Sq, Fs_Sq, xerr=sigma_fasq, yerr=sigma_fssq, c="c", ls="None")
    ax.scatter(Fa_Sr, Fs_Sr, marker="$Sr$", c="k", s=s * 5, zorder=100)
    ax.errorbar(Fa_Sr, Fs_Sr, xerr=sigma_fasr, yerr=sigma_fssr, c="c", ls="None")
    ax.scatter(Fa_Sw, Fs_Sw, marker="$Sw$", c="k", s=s * 6, zorder=100)
    ax.errorbar(Fa_Sw, Fs_Sw, xerr=sigma_fasw, yerr=sigma_fssw, c="c", ls="None")

    ax.scatter(Fa_Q, Fs_Q, marker="$Q$", c="k", s=s * 2.5, zorder=100)
    ax.errorbar(Fa_Q, Fs_Q, xerr=sigma_faq, yerr=sigma_fsq, c="c", ls="None")

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(("Fa_vs_Fs.", fig_format))
    fig.savefig("".join((outdir_composition, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_Fa_vs_Fs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print("Plot Fa vs Fs")

    limx1, limx2 = 15., 35.
    limy1, limy2 = 10., 30.

    shift = 3.  # Control ranges of axes
    s = 40.  # scaling parameter

    xticks, yticks = safe_arange(limx1, limx2, 5., endpoint=True), safe_arange(limy1, limy2, 5., endpoint=True)
    left, right = limx1 - shift, limx2 + shift
    bottom, top = limy1 - shift, limy2 + shift

    #####
    # NEED TO BE SET ACCORDING TO MODEL PREDICTIONS
    error_Fa, error_Fs = 5.7, 5.7

    Fa_S, Fs_S = [20.7], [18.3]
    Fa_Sq, Fs_Sq = [23.2], [21.0]
    Fa_Sr, Fs_Sr = [18.3], [19.2]
    Fa_Sw, Fs_Sw = [21.3], [14.7]

    Fa_Q, Fs_Q = [26.2], [23.8]
    #####

    # Indices in which Fa and Fs are
    ind_Fa, ind_Fs = num_minerals, num_minerals + endmembers_counts[0]

    filename = "combined-denoised-norm.npz"
    x_data, y_data, meta = load_data(filename, clean_dataset=True, return_meta=True)

    meta_train, meta_val, meta_test = split_meta_proportional(meta, y_data, "compositional")
    types = meta_test[:, 7]

    Fa_true, Fs_true = y_true[:, ind_Fa] * 100., y_true[:, ind_Fs] * 100.
    Fa_pred, Fs_pred = y_pred[:, ind_Fa] * 100., y_pred[:, ind_Fs] * 100.

    inds_H = np.array([("H" in OC_type) and ("HH" not in OC_type) if len(OC_type) == 2 else False for OC_type in types])
    inds_L = np.array([("L" in OC_type) and ("LL" not in OC_type) if len(OC_type) == 2 else False for OC_type in types])
    inds_LL = np.array(["LL" in OC_type if len(OC_type) == 3 else False for OC_type in types])

    fig, ax = plt.subplots(1, 2, figsize=(4.5 * 2, 6))


    for i in range(2):
        # definition of boxes from (for some reasons should be used just once)
        # https://www.researchgate.net/figure/Mineral-compositional-data-for-the-Morokweng-meteoriteDiagram-shows-ferrosilite-Fs_fig3_7093505
        H_rect = patches.Rectangle((16.2, 14.5), 3.8, 3.5, linewidth=1.5, edgecolor="r", facecolor="none")
        L_rect = patches.Rectangle((22.0, 19.0), 4.0, 3.0, linewidth=1.5, edgecolor="g", facecolor="none")
        LL_rect = patches.Rectangle((26.0, 22.0), 6.0, 4.2, linewidth=1.5, edgecolor="b", facecolor="none")

        if i == 0:
            ax[i].scatter(Fa_true[inds_H], Fs_true[inds_H], c="r", s=s, label="H")
            ax[i].scatter(Fa_true[inds_L], Fs_true[inds_L], c="g", s=s, label="L")
            ax[i].scatter(Fa_true[inds_LL], Fs_true[inds_LL], c="b", s=s, label="LL")
        else:
            ax[i].scatter(Fa_pred[inds_H], Fs_pred[inds_H], c="r", s=s, label="H")
            ax[i].scatter(Fa_pred[inds_L], Fs_pred[inds_L], c="g", s=s, label="L")
            ax[i].scatter(Fa_pred[inds_LL], Fs_pred[inds_LL], c="b", s=s, label="LL")

            ax[i].errorbar(Fa_pred[inds_H], Fs_pred[inds_H], xerr=error_Fa, yerr=error_Fs, c="r", fmt="o")
            ax[i].errorbar(Fa_pred[inds_L], Fs_pred[inds_L], xerr=error_Fa, yerr=error_Fs, c="g", fmt="o")
            ax[i].errorbar(Fa_pred[inds_LL], Fs_pred[inds_LL], xerr=error_Fa, yerr=error_Fs, c="b", fmt="o")

        ax[i].set_xlabel("Fa")
        if i == 0:
            ax[i].set_ylabel("Fs")
            ax[i].set_title("Ordinary chondrites")
        else:
            ax[i].set_title("Predictions")
            ax[i].set_yticklabels([])
        ax[i].tick_params(axis="both")
        ax[i].axis("square")

        ax[i].set_xticks(xticks)
        ax[i].set_yticks(yticks)
        ax[i].set_xlim(left=left, right=right)
        ax[i].set_ylim(bottom=bottom, top=top)

        # add the patches
        ax[i].add_patch(H_rect)
        ax[i].add_patch(L_rect)
        ax[i].add_patch(LL_rect)

        ax[i].legend()

    ax[1].scatter(Fa_S, Fs_S, marker="$S$", c="k", s=s * 2.5)
    ax[1].scatter(Fa_Sq, Fs_Sq, marker="$Sq$", c="k", s=s * 5)
    ax[1].scatter(Fa_Sr, Fs_Sr, marker="$Sr$", c="k", s=s * 5)
    ax[1].scatter(Fa_Sw, Fs_Sw, marker="$Sw$", c="k", s=s * 6)

    ax[1].scatter(Fa_Q, Fs_Q, marker="$Q$", c="k", s=s * 2.5)

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(("Fa_vs_Fs.", fig_format))
    fig.savefig("".join((outdir_composition, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_error_evaluation_ol_opx(suf: str = "pures", quiet: bool = False) -> None:
    if not quiet:
        print("Print quantiles")

    percentile = safe_arange(0., 100., 5., endpoint=True)
    shift = 3.  # Control ranges of axes (from 0 - shift to 100 + shift)

    xticks, yticks = safe_arange(0., 100., 10., endpoint=True), safe_arange(0., 100., 10., endpoint=True)
    left, right = -shift, 100. + shift
    bottom, top = -shift, 100. + shift

    # define the lines
    x_line = safe_arange(-150., 150., endpoint=True)
    y_line_10 = np.ones(np.shape(x_line)) * 10.
    y_line_20 = np.ones(np.shape(x_line)) * 20.
    l10, l20 = "k--", "k--"

    #####
    # NEED TO BE SET ACCORDING TO MODEL PREDICTIONS
    quantile_ol = np.array([1.5258789e-05, 1.8050480e-01, 3.9839172e-01, 7.2644901e-01,
                             1.1257100e+00, 1.4958014e+00, 1.9493690e+00, 2.2068353e+00,
                             2.2957458e+00, 2.3736839e+00, 2.4543171e+00, 2.7094297e+00,
                             3.0693655e+00, 3.6118984e+00, 4.8393745e+00, 6.1240335e+00,
                             7.0013580e+00, 7.4689040e+00, 7.8921270e+00, 8.3154631e+00,
                             8.3744736e+00])

    quantie_opx = np.array([0.2890625, 0.43381977, 0.51078033, 1.5087776, 1.5450859,
                             1.5713272, 1.6243706, 1.8998203, 2.0758362, 2.1846313,
                             3.2700148, 3.7066813, 4.024225, 4.1321206, 4.177703,
                             5.094024, 5.6581955, 6.756565, 7.5293565, 8.264683,
                             9.635145])
    #####

    quantile = stack((quantile_ol, quantie_opx), axis=1)

    titles_all = ["Fa", "Fs (OPX)"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(percentile, quantile, linewidth=2)
    # constant error lines
    ax.plot(x_line, y_line_10, l10)
    ax.plot(x_line, y_line_20, l20)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Absolute error (pp)")

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_ylim(bottom=left, top=top)
    ax.set_xlim(left=left, right=right)

    ax.legend(titles_all, loc="upper left")

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(("quantile_error_plot_", suf, ".", fig_format))
    fig.savefig("".join((outdir_composition, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_scatter_NN_BC() -> None:
    from modules.NN_evaluate import evaluate_test_data

    max_cpx = 0.1

    LW = 1.
    s = 30.  # scaling parameter (marker size)
    x_line, y_line = safe_arange(-150., 150., endpoint=True), safe_arange(-150., 150., endpoint=True)
    y1p_line, y1m_line = y_line + 10., y_line - 10.
    y2p_line, y2m_line = y_line + 20., y_line - 20.
    l0, l10, l20 = "k-", "m-", "c-"
    lab_line0, lab_line10, lab_line20 = "0 pp error", "10 pp error", "20 pp error"

    shift = 3.  # Control ranges of axes (from 0 - shift to 100 + shift)

    xticks, yticks = safe_arange(0., 100., 25., endpoint=True), safe_arange(0., 100., 25., endpoint=True)
    left, right = -shift, 100. + shift
    bottom, top = -shift, 100. + shift

    true_errorbar = np.array([3.])  # pp

    filename_train_data = "combined-denoised-norm.npz"
    x_data, y_data, meta = load_data(filename_train_data, clean_dataset=True, return_meta=True)

    x_train, y_train, x_val, y_val, x_test, y_test = split_data_proportional(x_data, y_data)
    meta_train, meta_val, meta_test = split_meta_proportional(meta, y_data, "compositional")
    types = meta_test[:, 7]

    data = np.load("".join((_path_data, filename_train_data)), allow_pickle=True)
    wvl = data["wavelengths"]

    model_names = ["20220330113805_CNN.h5"]  # [14.4, 5.7, 5.7, 10.7]
    y_pred, accuracy = evaluate_test_data(model_names, x_test, y_test, subfolder_model="compositional")

    # jenom ty, co obrasuhuji OL a OPX (jen mixtures 12 a 14)
    binary = np.array(y_test[:, :num_minerals] > 0, dtype=int)
    base = np.array([8, 4, 2, 1])[minerals_used]
    mixtures = np.sum(binary * base, axis=1)

    mask1 = np.logical_or(mixtures == 12, mixtures == 14)
    # only those with CPX <= 10
    mask2 = y_test[:, 2] <= max_cpx

    mask = np.logical_and(mask1, mask2)

    predicted_errorbar_reduced, actual_errorbar_reduced = error_estimation_bin_like(y_test, y_pred,
                                                                                    num_minerals=num_minerals,
                                                                                    num_labels=num_minerals + np.sum(
                                                                                        endmembers_counts),
                                                                                    actual_error=true_errorbar)

    x_true = x_test[mask]
    y_test = y_test[mask]
    y_pred = y_pred[mask]
    types = types[mask]
    predicted_errorbar_reduced = predicted_errorbar_reduced[:, :, mask]
    actual_errorbar_reduced = actual_errorbar_reduced[:, :, mask]

    inds_HED = np.array(["HED" in t for t in types])
    types[inds_HED] = "V"
    types[~inds_HED] = "S"

    BAR, BIC, BIIC = calc_BAR_BC(wvl, x_true)

    # remove nans
    mask = np.logical_and(np.logical_and(BAR > 0, BIC > 0), BIIC > 0)
    BAR, BIC, BIIC = BAR[mask], BIC[mask], BIIC[mask]
    OL_true = y_test[mask, 0] * 100.
    OL_pred = y_pred[mask, 0] * 100.
    Fs_true = y_test[mask, 5] * 100.
    Fs_pred = y_pred[mask, 5] * 100.
    types = types[mask]
    predicted_errorbar_reduced = predicted_errorbar_reduced[:, :, mask]
    actual_errorbar_reduced = actual_errorbar_reduced[:, :, mask]

    OL_BAR, Fs_BIIC, Wo_BIIC = calc_composition(BAR, BIC, BIIC, types, method="biic")
    _, Fs_BIC, _ = calc_composition(BAR, BIC, BIIC, types, method="bic")

    # filter the data
    mask = filter_data_mask(OL_BAR, Fs_BIIC, Wo_BIIC)
    OL_BAR = OL_BAR[mask]
    Fs_BIIC, Wo_BIIC = Fs_BIIC[mask], Wo_BIIC[mask]
    OL_true = OL_true[mask]
    OL_pred = OL_pred[mask]
    Fs_true = Fs_true[mask]
    Fs_pred = Fs_pred[mask]
    Fs_BIC = Fs_BIC[mask]
    types = types[mask]
    predicted_errorbar_reduced = predicted_errorbar_reduced[:, :, mask]
    actual_errorbar_reduced = actual_errorbar_reduced[:, :, mask]

    pred_errorbar_model = predicted_errorbar_reduced[[0, 5]]
    true_errorbar_model = actual_errorbar_reduced[[0, 5]]

    actual_errorbar_BIC = np.zeros(np.shape(types))
    actual_errorbar_BIC[types == "S"] = 1.4
    actual_errorbar_BIC[types == "V"] = 3.

    actual_errorbar_BIIC = np.zeros(np.shape(Wo_BIIC))
    actual_errorbar_BIIC[Wo_BIIC < 30] = 5.
    actual_errorbar_BIIC[Wo_BIIC >= 30] = 4.

    pred_errorbar_BIC, true_errorbar_BIC = cut_error_bars(Fs_true, true_errorbar, Fs_BIC, actual_errorbar_BIC)
    pred_errorbar_BIIC, true_errorbar_BIIC = cut_error_bars(Fs_true, true_errorbar, Fs_BIIC, actual_errorbar_BIIC)

    fig, ax = plt.subplots(1, 2, figsize=(4.7 * 2. + 1.5, 6))
    # line first in legend
    for i in range(2):
        ax[i].plot(x_line, y_line, l0, label=lab_line0, linewidth=LW)
        ax[i].plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW)
        ax[i].plot(x_line, y1m_line, l10, linewidth=LW)
        ax[i].plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW)
        ax[i].plot(x_line, y2m_line, l20, linewidth=LW)

    ax[0].scatter(OL_true, OL_pred, s=s, label="NN pred.", zorder=100)
    ax[0].scatter(OL_true, OL_BAR, s=s, label="BAR", zorder=99)

    ax[0].errorbar(OL_true, OL_pred, yerr=pred_errorbar_model[0],
                   xerr=true_errorbar_model[0], fmt="tab:blue", ls="", zorder=90)
    ax[0].errorbar(OL_true, OL_BAR, xerr=true_errorbar_model[0], fmt="tab:orange", ls="", zorder=89)

    ax[1].scatter(Fs_true, Fs_pred, s=s, label="NN pred.", zorder=100)
    ax[1].scatter(Fs_true, Fs_BIC, s=s, label="BIC", zorder=99)
    ax[1].scatter(Fs_true, Fs_BIIC, s=s, label="BIIC", zorder=98)

    ax[1].errorbar(Fs_true, Fs_pred, yerr=pred_errorbar_model[1],
                   xerr=true_errorbar_model[1], fmt="tab:blue", ls="", zorder=90)
    ax[1].errorbar(Fs_true, Fs_BIC, yerr=pred_errorbar_BIC,
                   xerr=true_errorbar_BIC, fmt="tab:orange", ls="", zorder=89)
    ax[1].errorbar(Fs_true, Fs_BIIC, yerr=pred_errorbar_BIIC,
                   xerr=true_errorbar_BIIC, fmt="tab:green", ls="", zorder=88)

    ax[0].set_title("Olivine")
    ax[1].set_title("Fs (OPX)")

    ax[0].set_xlabel("Actual (vol\%)")
    ax[0].set_ylabel("Modelled (vol\%)")

    ax[1].set_xlabel("Actual")
    ax[1].set_ylabel("Modelled")

    for i in range(2):
        ax[i].tick_params(axis="both")
        ax[i].axis("square")

        ax[i].set_xticks(xticks)
        ax[i].set_yticks(yticks)
        ax[i].set_xlim(left=left, right=right)
        ax[i].set_ylim(bottom=bottom, top=top)

        """
        if i > 0:
            ax[i].set_yticklabels([])
        """

        ax[i].legend(loc="upper left", frameon=False)

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(("scatter_plot_NN_BAR_BC_met_mix.", fig_format))
    fig.savefig("".join((outdir_composition, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_spectra_1(x_data: np.ndarray, y_data: np.ndarray) -> None:
    # plot spectra of minerals and B-D asteroids

    xticks, yticks = safe_arange(0.5, 2.5, 0.5, endpoint=True), safe_arange(0., 5., endpoint=True)
    left, right = 0.0, 5.0
    bottom, top = 0.4, 2.500
    x = safe_arange(450., 2450., 5., endpoint=True) / 1000.  # um
    titles = ["Olivine", "Orthopyroxene", "Clinopyroxene", "Laboratory mixtures", "Meteorites", "Asteroids"]

    m, n = best_blk(len(titles))

    # asteroid data
    filename = "".join((_path_data, "/taxonomy/asteroid_spectra-denoised-norm.npz"))
    data = np.load(filename, allow_pickle=True)  # to read the file
    types = np.array(data["metadata"][:, 1], dtype=str)

    inds_S = np.array(["S" in ast_type for ast_type in types])
    inds_Q = np.array(["Q" in ast_type for ast_type in types])
    inds_A = np.array(["A" in ast_type for ast_type in types])
    inds_V = np.array(["V" in ast_type for ast_type in types])

    inds = inds_S + inds_Q + inds_V + inds_A

    # ignore Sq: since its spectrum is very weird
    inds[types == "Sq:"] = False

    ast_data = data["spectra"][inds]

    fig, ax = plt.subplots(m, n, figsize=(4.7 * n, 4.7 * m))

    ax = np.reshape(ax, (m, n))

    inds = np.where(np.sum(y_data[:, :3] > 0, axis=1) > 1)[0]
    # urceno z konkretnich dat (90, 63)
    inds_met, inds_mix = inds[:63], inds[63:]

    for j, title in enumerate(titles):
        if j <= 2:
            data = x_data[y_data[:, j] == 1]
        if j == 3:
            data = x_data[inds_mix]
            # data = x_data[109:180]
        if j == 4:
            data = x_data[inds_met]
            # data = x_data[:109]
        if j == 5:
            data = ast_data

        i, k = np.unravel_index(j, (m, n))

        ax[i, k].plot(x, np.transpose(data))

        ax[i, k].tick_params(axis="both")
        ax[i, k].set_title(title)

        ax[i, k].set_xticks(xticks)
        ax[i, k].set_yticks(yticks)
        ax[i, k].set_xlim(left=left, right=right)
        ax[i, k].set_ylim(bottom=bottom, top=top)

        if k > 0:
            ax[i, k].set_yticklabels([])
        else:
            ax[i, k].set_ylabel("Reflectance (normalised)")

        if i == 0:
            ax[i, k].set_xticklabels([])
        else:
            ax[i, k].set_xlabel("Wavelength ($\mu$m)")

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(("spectra_all_1.", big_fig_format))
    fig.savefig("".join((outdir, "/", fig_name)), format=big_fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close()


def plot_spectra_2() -> None:
    # plot spectra of B-D asteroids and Itokawa and Eros

    # figure limits
    bottom, top = 0.0, 1.5
    left, right = 0.4, 2.5

    xticks, yticks = safe_arange(0., right, 0.5, endpoint=True), safe_arange(0., top, 0.5, endpoint=True)

    titles = list(classes.keys())
    titles.append("Eros")
    titles.append("Itokawa")

    m, n = best_blk(len(titles))

    # asteroid data
    filename = "".join((_path_data, "/asteroid_spectra-reduced-denoised-norm.npz"))
    data = np.load(filename, allow_pickle=True)  # to read the file
    types = np.array(data["metadata"][:, 1], dtype=str)

    fig, ax = plt.subplots(m, n, figsize=(6 * n, 8 * m))
    ax = np.reshape(ax, (m, n))

    for k, unique_type in enumerate(np.unique(types)):
        inds_class = np.array([unique_type == ast_type for ast_type in types])
        spectra, wavelengths = data["spectra"][inds_class], data["wavelengths"] / 1000.  # um
        spectra = normalise_in_rows(spectra, np.transpose(spectra[:, wavelengths == 1.500]))  # normalise at 1500 nm
        avg_spectrum = np.mean(spectra, axis=0)

        i, j = np.unravel_index(k, (m, n))

        ax[i, j].plot(wavelengths, np.transpose(spectra), "-k")
        ax[i, j].plot(wavelengths, avg_spectrum, "-r", linewidth=3)

        ax[i, j].tick_params(axis="both")
        ax[i, j].set_title(titles[k])

        ax[i, j].set_xticks(xticks)
        ax[i, j].set_yticks(yticks)
        ax[i, j].set_xlim(left=left, right=right)
        ax[i, j].set_ylim(bottom=bottom, top=top)

        if j > 0:
            ax[i, j].set_yticklabels([])
        else:
            ax[i, j].set_ylabel("Reflectance (normalised)")

        if i == m - 1:
            ax[i, j].set_xlabel("Wavelength ($\mu$m)")

    # Itokawa and Eros data
    j += 1
    filename = "".join((_path_data, "/Eros-denoised-norm.npz"))
    data = np.load(filename, allow_pickle=True)  # to read the file
    spectra, wavelengths = data["spectra"], data["wavelengths"] / 1000.  # um
    avg_spectrum = np.mean(spectra, axis=0)

    ax[i, j].plot(wavelengths, np.transpose(spectra), "-k")
    ax[i, j].plot(wavelengths, avg_spectrum, "-r", linewidth=3)

    ax[i, j].tick_params(axis="both")
    ax[i, j].set_title(titles[k + 1])

    ax[i, j].set_xticks(xticks)
    ax[i, j].set_yticks(yticks)
    ax[i, j].set_xlim(left=left, right=right)
    ax[i, j].set_ylim(bottom=bottom, top=top)

    ax[i, j].set_yticklabels([])
    ax[i, j].set_xlabel("Wavelength ($\mu$m)")

    j += 1
    filename = "".join((_path_data, "/Itokawa-denoised-norm.npz"))
    data = np.load(filename, allow_pickle=True)  # to read the file
    spectra, wavelengths = data["spectra"], data["wavelengths"] / 1000.  # um
    avg_spectrum = np.mean(spectra, axis=0)

    ax[i, j].plot(wavelengths, np.transpose(spectra), "-k")
    ax[i, j].plot(wavelengths, avg_spectrum, "-r", linewidth=3)

    ax[i, j].tick_params(axis="both")
    ax[i, j].set_title(titles[k + 2])

    ax[i, j].set_xticks(xticks)
    ax[i, j].set_yticks(yticks)
    ax[i, j].set_xlim(left=left, right=right)
    ax[i, j].set_ylim(bottom=bottom, top=top)

    ax[i, j].set_yticklabels([])
    ax[i, j].set_xlabel("Wavelength ($\mu$m)")

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(("spectra_all_2.", big_fig_format))
    fig.savefig("".join((outdir, "/", fig_name)), format=big_fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close()


def plot_mineralogy_histogram(y_data: np.ndarray) -> None:
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    LW_plot = 2

    left, right = 0.0, 100.

    i = 0
    mask_ol = y_data[:, i] > 0.
    ol = y_data[mask_ol, i] * 100.
    i = 1
    mask_opx = y_data[:, i] > 0.
    opx = y_data[mask_opx, i] * 100.
    i = 2
    mask_cpx = y_data[:, i] > 0.
    cpx = y_data[mask_cpx, i] * 100.

    i = 3
    fa = y_data[mask_ol, i] * 100.

    i = 5
    fs_opx = y_data[mask_opx, i] * 100.

    i = 7
    fs_cpx = y_data[mask_cpx, i] * 100.
    i = 8
    en_cpx = y_data[mask_cpx, i] * 100.
    i = 9
    wo_cpx = y_data[mask_cpx, i] * 100.

    fig, ax = plt.subplots(1, 4, figsize=(5 * 4, 5))
    # modal
    ax[0].hist(ol, bins=bins, linewidth=LW_plot, edgecolor="c", fill=False, label="Olivine")
    ax[0].hist(opx, bins=bins, linewidth=LW_plot, edgecolor="m", fill=False, label="Orthopyroxene")
    ax[0].hist(cpx, bins=bins, linewidth=LW_plot, edgecolor="y", fill=False, label="Clinopyroxene")
    ax[0].set_xlabel("Modal abundance (vol\%)")
    ax[0].set_ylabel("Counts")

    # olivine
    ax[1].hist(fa, bins=bins, linewidth=LW_plot, edgecolor="c", fill=False, label="Fayalite")
    ax[1].set_xlabel("Olivine")

    # orthopyroxene
    ax[2].hist(fs_opx, bins=bins, linewidth=LW_plot, edgecolor="c", fill=False, label="Ferrosilite")
    ax[2].set_xlabel("Orthopyroxene")

    # clinopyroxene
    ax[3].hist(fs_cpx, bins=bins, linewidth=LW_plot, edgecolor="c", fill=False, label="Ferrosilite")
    ax[3].hist(en_cpx, bins=bins, linewidth=LW_plot, edgecolor="m", fill=False, label="Enstatite")
    ax[3].hist(wo_cpx, bins=bins, linewidth=LW_plot, edgecolor="y", fill=False, label="Wollastonite")
    ax[3].set_xlabel("Clinopyroxene")

    for i in range(4):
        ax[i].set_xlim(left=left, right=right)
        ax[i].tick_params(axis="both")
        ax[i].legend(loc="best")

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(("spectra_min_hist.", fig_format))
    fig.savefig("".join((outdir_composition, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close()


def plot_surface_spectra(y_pred: np.ndarray, filename: str, what_type: Literal["composition", "taxonomy"]) -> None:
    def get_most_probable_classes(limit: float = 1.) -> tuple[int, np.ndarray]:
        # limit: minimum averaged probability
        n_probable_classes = np.sum(mean_of_predictions >= limit)
        most_probable_classes = np.argsort(mean_of_predictions)[-n_probable_classes:][::-1]

        print("\nMost probable classes:")
        for cls in most_probable_classes:
            print("{:4s} {:5.2f}%".format(classes2[cls], round(mean_of_predictions[cls], 2)))

        return n_probable_classes, most_probable_classes

    def get_most_winning_classes(limit: float = 0.) -> tuple[int, np.ndarray]:
        # limit: minimum most probable predictions
        unique, counts = np.unique(y_pred.argmax(axis=1), return_counts=True)
        unique, counts = unique[counts >= limit], counts[counts >= limit]

        n_probable_classes = np.size(unique)
        most_probable_classes = unique[np.argsort(counts)][::-1]

        norm_counts = np.sort(counts)[::-1] / np.sum(counts) * 100.
        print("\nMost winning classes:")
        for icls, cls in enumerate(most_probable_classes):
            print("{:4s} {:5.2f}%".format(classes2[cls], round(norm_counts[icls], 2)))

        return n_probable_classes, most_probable_classes

    font_size_axis = 36

    cmap = "viridis_r"
    vmin, vmax = 0., 100.
    alpha = 0.4
    s = 10.

    xticks, yticks = safe_arange(0., 360., 10., endpoint=True), safe_arange(-90., 90., 10., endpoint=True)
    left, right = 0.0, 360.
    bottom, top = -90., 90.

    cticks, ctickslabel = safe_arange(0., 100., 10., endpoint=True), safe_arange(0., 100., 10., endpoint=True)

    if "Itokawa" in filename:
        background_image = "new_itokawa_mosaic.jpg"
        name = "Itokawa"
    elif "Eros" in filename:
        background_image = "eros_cyl_near.jpg"
        name = "Eros"
    else:
        raise ValueError('"filename" must contain either "Itokawa" or "Eros"')

    indices_file = np.load("".join((_path_data, filename)), allow_pickle=True)
    indices = np.array(indices_file["metadata"][:, :2], dtype=float)

    mean_of_predictions = np.mean(y_pred, axis=0) * 100.

    if what_type == "taxonomy":
        _, most_probable_classes_1 = get_most_probable_classes()
        _, most_probable_classes_2 = get_most_winning_classes()
        most_probable_classes = stack((most_probable_classes_1,
                                       np.setdiff1d(most_probable_classes_2, most_probable_classes_1)))
        n_probable_classes = len(most_probable_classes)

        titles = ["".join((name, " ", classes2[most_probable_classes[i]],
                           "-type predictions")) for i in range(n_probable_classes)]

        labels = [classes2[most_probable_classes[i]] for i in range(n_probable_classes)]

    elif what_type == "composition":
        # set titles (this should work well)

        titles_all = [mineral_names] + endmember_names
        titles_all = flatten_list(titles_all)[used_indices(minerals_used, endmembers_used)]
        # titles_all = flatten_list(titles_all)[unique_indices(minerals_used, endmembers_used, all_minerals=True)]

        most_probable_classes = unique_indices(minerals_used, endmembers_used, return_digits=True)
        labels = titles_all[most_probable_classes]

        n_probable_classes = len(most_probable_classes)

        print("\nSelected mineralogy:")
        for i, cls in enumerate(most_probable_classes):
            print("{:14s} {:5.2f}%".format(labels[i], round(mean_of_predictions[cls], 2)))

        titles = ["".join((name, " ", labels[i], " predictions"))
                  for i in range(n_probable_classes)]

    else:
        raise ValueError('"what_type" must be either "taxonomy" or "composition"')

    # Color code dominant classes / labels
    probability_values = np.transpose(np.array([y_pred[:, most_probable_classes[i]]
                                                for i in range(n_probable_classes)]))

    if "Itokawa" in filename:
        # list of craters from "A survey of possible impact structures on 25143 Itokawa"
        # https://doi.org/10.1016/j.icarus.2008.10.027
        # id (index of array); lon, lat, confidence
        craters = np.zeros((38, 3))
        craters[0], craters[1], craters[2], craters[3] = [348, 25, 4], [175, -10, 2], [275, -2, 4], [128, 0, 3]
        craters[4], craters[5], craters[6], craters[7] = [112, 40, 2], [8, 8, 2], [17, -8, 4], [172, 15, 4]
        craters[8], craters[9], craters[10], craters[11] = [134, 20, 4], [244, -40, 3], [151, -6, 3], [215, 17, 3]
        craters[12], craters[13], craters[14], craters[15] = [269, 34, 1], [145, 4, 3], [102, -10, 1], [205, -18, 2]
        craters[16], craters[17], craters[18], craters[19] = [216, -26, 4], [221, -36, 4], [212, -33, 3], [254, -15, 4]
        craters[20], craters[21], craters[22], craters[23] = [7, -18, 4], [162, 1, 2], [14, -17, 2], [52, 12, 3]
        craters[24], craters[25], craters[26], craters[27] = [183, 17, 4], [169, 24, 4], [345, -17, 3], [277, -13, 4]
        craters[28], craters[29], craters[30], craters[31] = [45, 19, 3], [117, -39, 3], [202, 28, 4], [207, 33, 4]
        craters[32], craters[33], craters[34], craters[35] = [232, -40, 4], [45, -28, 1], [244, 6, 2], [111, -33, 3]
        craters[36], craters[37] = [319, -28, 1], [205, -76, 1]

        craters = stack((np.arange(1, len(craters) + 1), craters), axis=1)  # add IDs

    for i in range(n_probable_classes):
        # Plot the coverage map using latitude and longitude from HB
        img = plt.imread("".join((_path_asteroid_images, background_image)))  # Background image
        fig, ax = plt.subplots(figsize=(30, 25))
        ax.imshow(img, cmap="gray", extent=[0, 360, -90, 90], alpha=1)

        # Draw the predictions map
        im = ax.scatter(indices[:, 0], indices[:, 1], s=s, c=probability_values[:, i] * 100.,
                        marker=",", cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)

        if "Itokawa" in filename:
            # draw craters
            cmap_text = mpl.colors.LinearSegmentedColormap.from_list("", ["red", "red", "pink", "pink"])

            for crater in craters:
                if crater[0] < 10:
                    fact = 1.5
                else:
                    fact = 3.

                ax.scatter(crater[1], crater[2], s=180. * fact, c=crater[3], cmap=cmap_text, vmin=1, vmax=4,
                           marker="${:.0f}$".format(crater[0]), zorder=100)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        plt.xticks(rotation=90., fontsize=font_size_axis - 4)
        plt.yticks(fontsize=font_size_axis - 4)

        ax.grid()

        ax.set_xlabel("Longitude (deg)", fontsize=font_size_axis)  # \N{DEGREE SIGN}
        ax.set_ylabel("Latitude (deg)", fontsize=font_size_axis)
        ax.set_title(titles[i], fontsize=font_size_axis + 4)

        ax.set_xlim(left=left, right=right)
        ax.set_ylim(bottom=bottom, top=top)

        divider = make_axes_locatable(ax)
        # cax = divider.append_axes(**cbar_kwargs)
        # cbar = plt.colorbar(im, cax=cax)
        cax = divider.append_axes("bottom", size="10%", pad=1.35)
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")

        cbar.set_ticks(cticks)
        cbar.set_ticklabels(ctickslabel)
        cbar.ax.tick_params(labelsize=font_size_axis - 4)

        if what_type == "taxonomy":
            cbar.ax.set_xlabel("Similarity (\%)", fontsize=font_size_axis)
        elif i < num_minerals:
            cbar.ax.set_xlabel("Modal abundance (vol\%)", fontsize=font_size_axis)

        plt.draw()
        plt.tight_layout()

        fig_name = "".join((name, "_", labels[i].replace(" ", "_"), ".", big_fig_format))
        fig.savefig("".join((outdir_surfaces, "/", fig_name)), format=big_fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close("all")


def plot_surface_spectra_shapeViewer(asteroid_name: str, what_prediction: str | list[str] = "all") -> None:

    alpha = 0.5

    if what_prediction == "all":
        titles_all = [mineral_names] + endmember_names
        titles_all = flatten_list(titles_all)[used_indices(minerals_used, endmembers_used)]
        # titles_all = flatten_list(titles_all)[unique_indices(minerals_used, endmembers_used, all_minerals=True)]
        what_prediction = list(titles_all) + list(classes.keys())

    path_to_images = "".join((_path_asteroid_images, "/shapeViewer/"))

    filenames = ["".join((asteroid_name, "_y.png")),  # side
                 "".join((asteroid_name, "_x.png")),  # front
                 "".join((asteroid_name, "_Y.png")),  # side
                 "".join((asteroid_name, "_X.png"))]  # back

    colorbar_title = "Similarity (\\%)"

    for which in what_prediction:
        fig, ax = plt.subplots(2, 2, figsize=(12, 6), width_ratios=[2, 1])
        if which in classes.keys():  # taxonomical map
            fig.suptitle("".join((asteroid_name, " ", which, "-type predictions")))
        else:  # compositional map
            fig.suptitle("".join((asteroid_name, " ", which, " predictions")))
        ax = ax.ravel()

        for i, filename in enumerate(filenames):
            # BGR to RGB
            fname = "".join((path_to_images, filename.replace("_", "".join(("_", which.replace(" ", "_"), "_")))))

            if not os.path.isfile(fname):
                warnings.warn("The file {file} doesn't exist.".format(file=fname))
                plt.close(fig)
                break

            im_frame = cv2.imread(fname)[:, :, ::-1]
            im_frame_bg = cv2.imread("".join((path_to_images, filename)))[:, :, ::-1]  # background

            ax[i].imshow(im_frame_bg)
            sc = ax[i].imshow(im_frame, alpha=alpha)

            # cut the image
            lims = np.where(im_frame < 255)
            ax[i].set_xlim([np.min(lims[1]), np.max(lims[1])])
            ax[i].set_ylim([np.max(lims[0]), np.min(lims[0])])

            ax[i].axis("off")

            if asteroid_name == "Itokawa":
                text_kwargs = {"horizontalalignment": "center", "verticalalignment": "center", "fontsize": SMALL_SIZE,
                               "transform": ax[i].transAxes, "c": "r"}
                if i == 0:
                    # ax[i].text(0.74, 0.50, "4", **text_kwargs)
                    # ax[i].text(0.61, 0.83, "5", **text_kwargs)
                    ax[i].text(0.10, 0.64, "6", **text_kwargs)
                    # ax[i].text(0.10, 0.34, "7", **text_kwargs)
                    ax[i].text(0.74, 0.74, "9", **text_kwargs)
                    # ax[i].text(0.93, 0.38, "11", **text_kwargs)
                    # ax[i].text(0.88, 0.52, "14", **text_kwargs)
                    # ax[i].text(0.59, 0.40, "15", **text_kwargs)
                    # ax[i].text(0.03, 0.10, "21", **text_kwargs)
                    ax[i].text(0.985, 0.515, "22", **text_kwargs)
                    # ax[i].text(0.06, 0.13, "23", **text_kwargs)
                    ax[i].text(0.405, 0.59, "24", **text_kwargs)
                    ax[i].text(0.38, 0.66, "29", **text_kwargs)
                    # ax[i].text(0.615, 0.19, "30", **text_kwargs)
                    # ax[i].text(0.43, 0.27, "34", **text_kwargs)
                    # ax[i].text(0.61, 0.23, "36", **text_kwargs)

                elif i == 1:
                    # ax[i].text(0.38, 0.82, "1", **text_kwargs)
                    # ax[i].text(0.04, 0.465, "3", **text_kwargs)
                    # ax[i].text(0.98, 0.46, "4", **text_kwargs)
                    ax[i].text(0.60, 0.65, "6", **text_kwargs)
                    # ax[i].text(0.78, 0.33, "7", **text_kwargs)
                    # ax[i].text(0.92, 0.38, "15", **text_kwargs)
                    # ax[i].text(0.64, 0.1, "21", **text_kwargs)
                    # ax[i].text(0.72, 0.11, "23", **text_kwargs)
                    ax[i].text(0.83, 0.60, "24", **text_kwargs)
                    # ax[i].text(0.30, 0.15, "27", **text_kwargs)
                    # ax[i].text(0.08, 0.35, "28", **text_kwargs)
                    ax[i].text(0.81, 0.67, "29", **text_kwargs)

                elif i == 2:
                    # ax[i].text(0.48, 0.46, "3", **text_kwargs)
                    # ax[i].text(0.385, 0.155, "10", **text_kwargs)
                    ax[i].text(0.215, 0.71, "12", **text_kwargs)
                    # ax[i].text(0.455, 0.77, "13", **text_kwargs)
                    ax[i].text(0.135, 0.21, "16", **text_kwargs)
                    # ax[i].text(0.23, 0.165, "17", **text_kwargs)
                    # ax[i].text(0.295, 0.13, "18", **text_kwargs)
                    # ax[i].text(0.25, 0.12, "19", **text_kwargs)
                    ax[i].text(0.385, 0.33, "20", **text_kwargs)
                    ax[i].text(0.885, 0.17, "27", **text_kwargs)
                    # ax[i].text(0.485, 0.355, "28", **text_kwargs)
                    # ax[i].text(0.22, 0.81, "31", **text_kwargs)
                    # ax[i].text(0.255, 0.83, "32", **text_kwargs)
                    # ax[i].text(0.345, 0.15, "33", **text_kwargs)
                    ax[i].text(0.335, 0.56, "35", **text_kwargs)
                    # ax[i].text(0.60, 0.25, "37", **text_kwargs)

                else:
                    # ax[i].text(0.42, 0.3, "2", **text_kwargs)
                    ax[i].text(0.405, 0.7, "8", **text_kwargs)
                    # ax[i].text(0.1, 0.75, "9", **text_kwargs)
                    # ax[i].text(0.095, 0.37, "11", **text_kwargs)
                    ax[i].text(0.82, 0.69, "12", **text_kwargs)
                    # ax[i].text(0.05, 0.55, "14", **text_kwargs)
                    # ax[i].text(0.765, 0.22, "16", **text_kwargs)
                    # ax[i].text(0.79, 0.17, "17", **text_kwargs)
                    ax[i].text(0.22, 0.5, "22", **text_kwargs)
                    ax[i].text(0.53, 0.72, "25", **text_kwargs)
                    # ax[i].text(0.395, 0.8, "26", **text_kwargs)
                    # ax[i].text(0.67, 0.8, "31", **text_kwargs)
                    # ax[i].text(0.68, 0.84, "32", **text_kwargs)
        else:
            sc.set_cmap("turbo")  # virdis is not implemented in shapeViewer yet
            sc.set_clim(0, 100)

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(sc, cax=cbar_ax)
            cbar.ax.set_ylabel(colorbar_title)

            plt.draw()

            fig_name = "".join((asteroid_name, "_", which.replace(" ", "_"), "_SV.", big_fig_format))
            fig.savefig("".join((outdir_surfaces, "/", fig_name)), format=big_fig_format, **savefig_kwargs, **pil_kwargs)
            plt.close(fig)


def plot_average_surface_spectra(y_pred: np.ndarray, asteroid_name: str) -> None:
    filename_data = "".join((asteroid_name, ".npz"))
    data = np.load("".join((_path_data, filename_data)), allow_pickle=True)

    spectra, wvl = data["spectra"], data["wavelengths"]

    most_probable_classes = np.unique(np.argmax(y_pred, axis=1))

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in most_probable_classes:
        ax.plot(wvl, np.mean(spectra[y_pred.argmax(axis=1) == cls], axis=0), label=classes2[cls])

    ax.set_ylabel("Reflectance")
    ax.set_xlabel("Wavelength (nm)")
    plt.legend()

    plt.draw()
    plt.tight_layout()


def plot_Sq_histogram(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    alpha = 0.5
    nbins = safe_arange(0., 100., 10., endpoint=True)

    xticks = safe_arange(0., 100., 10., endpoint=True)
    left, right = 0.0, 100.

    inds = y_true.ravel() == "Sq"
    probability_Q, probability_S = y_pred[inds, classes["Q"]] * 100., y_pred[inds, classes["S"]] * 100.


    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(probability_S, bins=nbins, alpha=alpha, label="S")
    ax.hist(probability_Q, bins=nbins, alpha=alpha, label="Q")

    ax.set_xticks(xticks)
    ax.set_xlim(left=left, right=right)

    ax.set_xlabel("Similarity (\%)")
    ax.set_ylabel("Counts")
    ax.legend()

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(("Sq_hist", ".", fig_format))
    fig.savefig("".join((outdir_taxonomy, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close("all")


def plot_ast_type_histogram(y_true: np.ndarray, y_pred: np.ndarray, type1: str, type2: str) -> None:
    alpha = 0.5
    nbins = safe_arange(0., 100., 5., endpoint=True)

    xticks = safe_arange(0., 100., 10., endpoint=True)
    left, right = 0.0, 100.

    inds_1, inds_2 = y_true[:, classes[type1]] == 1, y_true[:, classes[type2]] == 1
    probability_11, probability_12 = y_pred[inds_1, classes[type1]] * 100., y_pred[inds_2, classes[type1]] * 100.
    probability_21, probability_22 = y_pred[inds_1, classes[type2]] * 100., y_pred[inds_2, classes[type2]] * 100.


    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(probability_11, bins=nbins, alpha=alpha, label=type1)
    ax[0].hist(probability_12, bins=nbins, alpha=alpha, label=type2)

    ax[0].set_xticks(xticks)
    ax[0].set_xlim(left=left, right=right)

    ax[0].set_xlabel("".join(("Similarity to ", type1, " (\%)")))
    ax[0].set_ylabel("Counts")
    ax[0].legend()

    ax[1].hist(probability_21, bins=nbins, alpha=alpha, label=type1)
    ax[1].hist(probability_22, bins=nbins, alpha=alpha, label=type2)

    ax[1].set_xticks(xticks)
    ax[1].set_xlim(left=left, right=right)

    ax[1].set_xlabel("".join(("Similarity to ", type2, " (\%)")))
    # ax[1].set_ylabel("Counts")
    ax[1].legend()

    plt.draw()
    plt.tight_layout()

    fig_name = "".join((type1, "_", type2, "_hist", ".", fig_format))
    fig.savefig("".join((outdir_taxonomy, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close("all")


def plot_range_histogram(start: np.ndarray, stop: np.ndarray, step: np.ndarray) -> None:
    print("Range histograms")
    bottom = 0.

    x_lambda = safe_arange(3000., endpoint=True)
    y_lambda = np.zeros(len(x_lambda))

    left, right = np.min(x_lambda), np.max(x_lambda)
    min_res, max_res = 0., 20.

    for begin, end in zip(start, stop):
        tmp = np.zeros(y_lambda.shape)
        tmp[np.where((x_lambda >= begin) & (x_lambda <= end))] = 1.
        y_lambda += tmp

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.plot(x_lambda, y_lambda)
    ax2.hist(step, bins="auto")

    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Counts")
    ax1.tick_params(axis="both")
    ax1.set_ylim(bottom=bottom)
    ax1.set_xlim(left=left, right=right)
    ax1.set_title("Histogram of ranges")

    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Counts")
    ax2.tick_params(axis="both")
    ax2.set_ylim(bottom=bottom)
    ax2.set_xlim(left=min_res, right=max_res)
    ax2.set_title("Histogram of resolution")

    plt.draw()

    fig_name = "".join(("hist_range.", fig_format))
    fig.savefig("".join((outdir, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_numbers_histogram(fa: np.ndarray, fs: np.ndarray, wo: np.ndarray) -> None:
    print("Numbers histograms")
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    left, right = 0., 1.

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.hist(fa, bins=bins)
    ax2.hist(fs, bins=bins)
    ax3.hist(wo, bins=bins)

    ax1.set_xlabel("Fa")
    ax2.set_xlabel("Fs")
    ax3.set_xlabel("Wo")

    ax1.set_ylabel("Counts")

    ax1.tick_params(axis="both")
    ax2.tick_params(axis="both")
    ax3.tick_params(axis="both")

    ax1.set_xlim(left=left, right=right)
    ax2.set_xlim(left=left, right=right)
    ax3.set_xlim(left=left, right=right)

    ax1.set_title("Histogram of Fa")
    ax2.set_title("Histogram of Fs")
    ax3.set_title("Histogram of Wo")

    plt.draw()

    fig_name = "".join(("hist_numbers.", fig_format))
    fig.savefig("".join((outdir, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_ast_type_hist(y_pred: np.ndarray) -> None:
    print("Plot histograms of asteroids compositions")

    nbins = 10
    shift = 3.  # Control ylim

    predictions = y_pred * 100.

    filename = "".join((_path_data, "/asteroid_spectra-denoised-norm.npz"))
    data = np.load(filename, allow_pickle=True)
    types = data["metadata"][:, 1]

    inds_S = np.array(["S" in ast_type for ast_type in types])
    inds_Q = np.array(["Q" in ast_type for ast_type in types])
    inds_A = np.array(["A" in ast_type for ast_type in types])
    inds_V = np.array(["V" in ast_type for ast_type in types])

    inds = inds_S + inds_Q + inds_V + inds_A

    # ignore Sq: since its spectrum is very weird
    inds[types == "Sq:"] = False

    color = ["blue", "magenta", "brown", "black"]
    labels = ["S type", "Q type", "A type", "V type"]

    # modal first (expect all four minerals are in the model)
    n_min = 3
    fig, ax = plt.subplots(1, n_min, figsize=(6 * n_min, 5))  # !!!!!

    labelx = ["Olivine fraction (vol\%)", "Orthopyroxene fraction (vol\%)",
              "Clinopyroxene fraction (vol\%)", "Plagioclase fraction (vol\%)"]

    limy = 0

    for j in range(n_min):
        # if j > 1:  # !!!!!
        #    continue
        for ind, c, label in zip(inds, color, labels):
            # for i in [0, 1, 3]:  # !!!!! for i in range(len(inds)):
            #    if j == 0 and i == 3:  # !!!!!
            #        continue
            #    if j == 1 and i == 1:  # !!!!!
            #        continue
            hist, bins = np.histogram(predictions[ind, j], bins=nbins, range=(0, 100))
            hist = np.array(hist, dtype=np.float32)
            ax[j].bar(bins[:-1] + (bins[1] - bins[0]) / 2., hist / np.sum(hist) * 100.,
                      width=(bins[1] - bins[0]), fill=False, edgecolor=c, label=label, linewidth=2)

            if np.max(hist / np.sum(hist) * 100.) > limy:
                limy = np.max(hist / np.sum(hist) * 100.)

        if j > 0:
            ax[j].set_yticklabels([])
        else:
            ax[j].set_ylabel("Normalised counts (\%)")

        ax[j].set_xlabel(labelx[j])
        ax[j].legend(loc="upper left")

    # must be done after the first loop to get limy
    limy = np.min((100., np.round(limy))) + shift
    for j in range(n_min):
        # if j > 1:  # !!!!!
        #    continue
        ax[j].set_ylim(bottom=0, top=limy)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.draw()

    fig_name = "".join(("ast_type_hist_modal.", fig_format))
    fig.savefig("".join((outdir, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    # OL
    n_ol = 2
    fig, ax = plt.subplots(1, n_ol, figsize=(6 * n_ol, 5))

    labelx = ["Fa", "Fo"]

    limy = 0.

    for j in range(n_ol):
        for ind, c, label in zip(inds, color, labels):
            hist, bins = np.histogram(predictions[ind, n_min + j], bins=nbins, range=(0, 100))
            hist = np.array(hist, dtype=np.float32)
            ax[j].bar(bins[:-1] + (bins[1] - bins[0]) / 2., hist / np.sum(hist) * 100.,
                      width=(bins[1] - bins[0]), fill=False, edgecolor=c, label=label, linewidth=2)

            if np.max(hist / np.sum(hist) * 100.) > limy:
                limy = np.max(hist / np.sum(hist) * 100.)

        if j > 0:
            ax[j].set_yticklabels([])
        else:
            ax[j].set_ylabel("Normalised counts (\%)")

        ax[j].set_xlabel(labelx[j])
        ax[j].legend(loc="upper left")

    # must be done after the first loop to get limy
    limy = np.min((100., np.round(limy))) + shift
    for j in range(n_ol):
        ax[j].set_ylim(bottom=0, top=limy)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.draw()

    fig_name = "".join(("ast_type_hist_ol.", fig_format))
    fig.savefig("".join((outdir, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    # OPX
    n_opx = 2
    fig, ax = plt.subplots(1, n_opx, figsize=(6 * n_opx, 5))

    labelx = ["Fs (OPX)", "En (OPX)"]

    limy = 0.

    for j in range(n_opx):
        for ind, c, label in zip(inds, color, labels):
            hist, bins = np.histogram(predictions[ind, n_min + n_ol + j], bins=nbins, range=(0, 100))
            hist = np.array(hist, dtype=np.float32)
            ax[j].bar(bins[:-1] + (bins[1] - bins[0]) / 2., hist / np.sum(hist) * 100.,
                      width=(bins[1] - bins[0]), fill=False, edgecolor=c, label=label, linewidth=2)

            if np.max(hist / np.sum(hist) * 100.) > limy:
                limy = np.max(hist / np.sum(hist) * 100.)

        if j > 0:
            ax[j].set_yticklabels([])
        else:
            ax[j].set_ylabel("Normalised counts (\%)")

        ax[j].set_xlabel(labelx[j])
        ax[j].legend(loc="upper left")

    # must be done after the first loop to get limy
    limy = np.min((100., np.round(limy))) + shift
    for j in range(n_opx):
        ax[j].set_ylim(bottom=0, top=limy)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.draw()

    fig_name = "".join(("ast_type_hist_opx.", fig_format))
    fig.savefig("".join((outdir, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    # CPX
    n_cpx = 3
    fig, ax = plt.subplots(1, n_cpx, figsize=(6 * n_cpx, 5))

    labelx = ["Fs (CPX)", "En (CPX)", "Wo (CPX)"]

    limy = 0.

    for j in range(n_cpx):
        for ind, c, label in zip(inds, color, labels):
            hist, bins = np.histogram(predictions[ind, n_min + n_ol + n_opx + j], bins=nbins, range=(0, 100))
            hist = np.array(hist, dtype=np.float32)
            ax[j].bar(bins[:-1] + (bins[1] - bins[0]) / 2., hist / np.sum(hist) * 100.,
                      width=(bins[1] - bins[0]), fill=False, edgecolor=c, label=label, linewidth=2)

            if np.max(hist / np.sum(hist) * 100.) > limy:
                limy = np.max(hist / np.sum(hist) * 100.)

        if j > 0:
            ax[j].set_yticklabels([])
        else:
            ax[j].set_ylabel("Normalised counts (\%)")

        ax[j].set_xlabel(labelx[j])
        ax[j].legend(loc="upper left")

    # must be done after the first loop to get limy
    limy = np.min((100., np.round(limy))) + shift
    for j in range(n_cpx):
        ax[j].set_ylim(bottom=0, top=limy)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.draw()

    fig_name = "".join(("ast_type_hist_cpx.", fig_format))
    fig.savefig("".join((outdir, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_Tomas_mixtures() -> None:
    bottom = 0.
    left, right = 400., 2500.

    filename = "".join((_path_data, "/ol_opx_mix/OL_OPX-denoised-norm.npz"))
    data = np.load(filename, allow_pickle=True)
    x, spectra_msm = data["wavelengths"], data["spectra"]
    C = np.array([0, 10, 25, 50, 75, 90, 100], dtype=int)

    colours = ["k", "r", "b", "g", "c", "m", "y", "k", "r", "b", "g", "c", "m", "y"]
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    for spectra, c, colour in zip(spectra_msm, C, colours):
        ax.plot(x, spectra, colour, label=c)
    ax.set_ylabel("Reflectance")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylim(bottom=bottom)
    ax.tick_params(axis="both")
    ax.set_xlim(left=left, right=right)

    # ax.legend(loc="upper right")

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(("spectra_mixtures_Tomas.", fig_format))
    fig.savefig("".join((outdir, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close()


def plot_OC_distance(y_pred: np.ndarray) -> None:
    print("plot distance of OC predictions from the corresponding box")

    s = 30.  # scaling parameter (marker size)

    # Indices in which Fa and Fs are
    ind_Fa, ind_Fs = num_minerals, num_minerals + endmembers_counts[0]

    filename_train_data = "combined-denoised-norm.npz"
    x_data, y_data, meta = load_data(filename_train_data, clean_dataset=True, return_meta=True)

    meta_train, meta_val, meta_test = split_meta_proportional(meta, y_data, "compositional")
    types = meta_test[:, 7]


    Fa_pred, Fs_pred = y_pred[:, ind_Fa] * 100., y_pred[:, ind_Fs] * 100.

    inds_H = np.array([("H" in OC_type) and ("HH" not in OC_type) if len(OC_type) == 2 else False for OC_type in types])
    inds_L = np.array([("L" in OC_type) and ("LL" not in OC_type) if len(OC_type) == 2 else False for OC_type in types])
    inds_LL = np.array(["LL" in OC_type if len(OC_type) == 3 else False for OC_type in types])

    # https://www.researchgate.net/figure/Mineral-compositional-data-for-the-Morokweng-meteoriteDiagram-shows-ferrosilite-Fs_fig3_7093505
    rect_H = np.array([[16.2, 16.2 + 3.8], [14.5, 14.5 + 3.5]])
    rect_L = np.array([[22, 22 + 4], [19, 19 + 3]])
    rect_LL = np.array([[26, 26 + 6], [22, 22 + 4.2]])

    p_H = np.array([Fa_pred[inds_H], Fs_pred[inds_H]])
    p_L = np.array([Fa_pred[inds_L], Fs_pred[inds_L]])
    p_LL = np.array([Fa_pred[inds_LL], Fs_pred[inds_LL]])

    distance_H = np.array([distance(rect_H, p_H), distance(rect_L, p_H), distance(rect_LL, p_H)])
    distance_L = np.array([distance(rect_H, p_L), distance(rect_L, p_L), distance(rect_LL, p_L)])
    distance_LL = np.array([distance(rect_H, p_LL), distance(rect_L, p_LL), distance(rect_LL, p_LL)])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.scatter(distance_H, np.tile(np.array([[0, 1, 2]]).transpose(), (1, np.shape(distance_H)[1])), color="r",
               label="H", s=s)
    ax.scatter(distance_L, np.tile(np.array([[0, 1, 2]]).transpose(), (1, np.shape(distance_L)[1])), color="g",
               label="L", s=s)
    ax.scatter(distance_LL, np.tile(np.array([[0, 1, 2]]).transpose(), (1, np.shape(distance_LL)[1])), color="b",
               label="LL", s=s)

    ax.set_xlabel("Distance")
    ax.set_ylabel("Type")

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["H", "L", "LL"])

    ax.legend(loc="center right")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.draw()

    fig_name = "".join(("OC_box_dist.", fig_format))
    fig.savefig("".join((outdir, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_error_range_plots(file_names: list[str], error_type: str = "RMSE_all", applied_function: Callable = np.mean,
                           line_plot_or_matrix: Literal["both", "line", "matrix"] = "both") -> None:
    from modules.NN_config_range_test import constant_range_or_spacing

    if constant_range_or_spacing == "spacing":
        from modules.NN_config_range_test import start, stop, step
    elif constant_range_or_spacing == "window":
        from modules.NN_config_range_test import start, stop, _spacing, window_size, window_spacing

    # load data
    def load_data_for_plot(file_names: list[str], error_type: str) -> tuple[np.ndarray, ...]:
        for i, file_name in enumerate(file_names):
            subfolder = "range_test/"
            filename = "".join((_path_data, subfolder, file_name))
            data = pd.read_csv(filename, sep="\t")  # to read the file

            # select wanted RMSE
            indices = data["index"].to_numpy()
            ranges_all = data["range"].to_numpy()
            spacing_all = data["spacing"].to_numpy()
            rmse_all = data[error_type].to_numpy()

            # remove NaNs
            inds_not_nan = np.argwhere(np.isfinite(rmse_all))
            indices = indices[inds_not_nan]
            ranges_all = ranges_all[inds_not_nan]
            spacing_all = spacing_all[inds_not_nan]
            rmse_all = rmse_all[inds_not_nan]

            # constant range => slightly different spacing for each dataset
            if constant_range_or_spacing == "range":
                spacing, rmse = combine_same_range_models(indices, spacing_all, rmse_all, applied_function)

                if i == 0:
                    x_axis = spacing
                    rmse_all_data = rmse
                else:
                    x_axis = stack((x_axis, spacing), axis=0)
                    rmse_all_data = stack((rmse_all_data, rmse), axis=0)

            else:  # spacing or window do the same
                x_axis, rmse = combine_same_range_models(indices, ranges_all, rmse_all, applied_function)

                if i == 0:
                    rmse_all_data = rmse
                else:
                    rmse_all_data = stack((rmse_all_data, rmse), axis=0)

        if constant_range_or_spacing == "spacing":
            rmse_all_data = np.mean(rmse_all_data, axis=0)

        return x_axis, rmse_all_data

    def prepare_data(ranges_or_spacings: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                                                 np.ndarray | int]:
        if constant_range_or_spacing == "spacing":
            wvl = safe_arange(start, stop, step, endpoint=True)

            # line plot
            x = np.array([[int(x.split()[-1][:-1]) for x in ranges_or_spacings
                           if int(x.split()[0][1:-1]) == w] for w in wvl[:-1]])
            y = np.array([[er for er, x in zip(error, ranges_or_spacings) if int(x.split()[0][1:-1]) == w]
                          for w in wvl[:-1]])

            # mat plot
            res_mat = np.empty((len(wvl), len(wvl)))
            res_mat.fill(np.nan)

            for j, w in enumerate(wvl[:-1]):
                tmp = np.array([er for er, x in zip(error, ranges_or_spacings) if int(x.split()[0][1:-1]) == w])
                res_mat[1 + j:1 + j + np.size(y), j] = tmp

            return x, y, res_mat
        elif constant_range_or_spacing == "range":
            return ranges_or_spacings, error, 0
        else:
            x = np.array(["".join((str(window_start), "\u2013", str(window_start + window_size)))
                          for window_start in safe_arange(start, stop, _spacing)
                          if window_start + window_size <= stop])
            return x, error, 0

    def line_plot(ax, x, y):
        cm = plt.get_cmap("gist_rainbow")
        if constant_range_or_spacing == "range":
            num_colors = int(np.ceil(len(y) / 2.))  # SWIR full lines; no SWIR dashed lines
        else:
            num_colors = len(y)
        ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])

        if constant_range_or_spacing == "spacing":
            wvl = safe_arange(start, stop, step, endpoint=True, dtype=int)

            for j in range(num_colors):
                ax.plot(x[j], y[j], "--o", label="From {:.1f} nm".format(wvl[j]))
            ax.set_xticks(wvl[1:])
            ax.set_xlabel("To wavelength (nm)")

            ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0,
                      ncol=best_blk(len(y))[1])
        elif constant_range_or_spacing == "range":
            # extract ranges from file_names
            pos_under = [[m.start() for m in re.finditer("_", file)] for file in file_names]
            labels = ["".join((file[under[1] + 1:under[2]], "\u2013", file[under[2] + 1:-4]))
                      for file, under in zip(file_names, pos_under)]
            styles = ["-or", "--or", "-ob", "--ob"]
            for x_val, y_val, style, label in zip(x, y, styles, labels):
                ax.plot(x_val, y_val, style, label="".join((label, " nm")))

            ax.set_xlabel("Spacing (nm)")
            ax.legend(ncol=2)
        else:
            ax.plot(y.ravel(), "--o", label="Window spacing {:d}".format(int(window_spacing)))

            ax.set_xticks(safe_arange(len(x)))
            ax.set_xticklabels(x, ha="center", fontsize=10)
            ax.set_xlabel("Window range (nm)")
            ax.legend()

        ax.set_ylabel(error_label)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=False, left=True, right=False)

    def matrix_plot(ax_matrix, res_mat: np.ndarray | int) -> None:
        if "RMSE" in error_type:
            vmin = np.nanmin(res_mat)
            vmax = np.nanmin((np.nanmax(res_mat), np.nanmin(res_mat) + 10))
            cmap = "jet_r"
        else:
            vmin = np.nanmax((np.nanmin(res_mat), np.nanmax(res_mat) - 20))
            vmax = np.nanmax(res_mat)
            cmap = "jet"

        wvl = safe_arange(start, stop, step, endpoint=True, dtype=int)
        im = ax_matrix.imshow(res_mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        ax_matrix.set_xticks(safe_arange(len(wvl)))
        ax_matrix.set_yticks(safe_arange(len(wvl)))

        ax_matrix.set_xticklabels(wvl, rotation=90, ha="center")
        ax_matrix.set_yticklabels(wvl)

        ax_matrix.set_xlabel("From wavelength (nm)")
        ax_matrix.set_ylabel("To wavelength (nm)")
        ax_matrix.xaxis.set_label_position("bottom")
        ax_matrix.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                              bottom=True, top=False, left=True, right=False)

        divider = make_axes_locatable(ax_matrix)
        cax = divider.append_axes(**cbar_kwargs)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(error_label)

    x, error = load_data_for_plot(file_names, error_type)
    x, y, mat = prepare_data(x, error)

    error_type = error_type.replace("_nooutliers", "")
    tmp = error_type.split("_")

    if "RMSE" in error_type:
        error_label = "".join((tmp[0], " (pp; ", " ".join(tmp[1:]), ")"))
    else:
        error_label = "".join((tmp[0], " (", tmp[1], " ", tmp[2], "; ", " ".join(tmp[3:]), ")"))

    # no matrix plot for constant range
    if constant_range_or_spacing != "spacing":
        line_plot_or_matrix = "line"

    if constant_range_or_spacing == "window":
        fig_name_suffix = "".join((constant_range_or_spacing, "_", str(int(window_size)), "_",
                                   error_type, ".", fig_format))
    else:
        fig_name_suffix = "".join((constant_range_or_spacing, "_", error_type, ".", fig_format))

    if "line" in line_plot_or_matrix:
        if constant_range_or_spacing == "window":  # needed bigger window
            fig, ax = plt.subplots(1, 1, figsize=(18, 6))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        line_plot(ax, x, y)
        fig_name = "".join(("range_line_", fig_name_suffix))

    elif "matrix" in line_plot_or_matrix:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        matrix_plot(ax, mat)
        fig_name = "".join(("range_matrix_", fig_name_suffix))

    elif "both" in line_plot_or_matrix:
        fig, ax = plt.subplots(1, 2, figsize=(25, 10), gridspec_kw={"width_ratios": [2.3, 1]})
        line_plot(ax[0], x, y)
        matrix_plot(ax[1], mat)
        fig_name = "".join(("range_plot_", fig_name_suffix))
    else:
        raise ValueError('"line_plot_or_matrix" must be one of "both", "line", "matrix".')

    plt.draw()
    plt.tight_layout()

    outdir_range_tests = "".join((outdir, "/range_test/"))
    check_dir(outdir_range_tests)

    fig.savefig("".join((outdir_range_tests, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_error_spacing(file_name: str, error_type: str = "RMSE_all", applied_function: Callable = np.mean,
                       line_plot_or_matrix: Literal["both", "line", "matrix"] = "both") -> None:
    from modules.NN_config_range_test import start, stop, step

    # load data
    def load_data_for_plot(file_name: str, error_type: str) -> tuple[np.ndarray, ...]:
        subfolder = "range_test/"
        filename = "".join((_path_data, subfolder, file_name))
        data = pd.read_csv(filename, sep="\t")  # to read the file

        # select wanted metric
        indices = data["index"].to_numpy()
        ranges_all = data["range"].to_numpy()
        error_all = data[error_type].to_numpy()

        """
        # remove NaNs
        inds_not_nan = np.argwhere(np.isfinite(error_all))
        indices = indices[inds_not_nan]
        ranges_all = ranges_all[inds_not_nan]
        error_all = error_all[inds_not_nan]
        """

        x_axis, error = combine_same_range_models(indices, ranges_all, error_all, applied_function)

        return x_axis, error

    def prepare_data(x_axis: np.ndarray, error: np.ndarray) -> tuple[np.ndarray, ...]:
        wvl = safe_arange(start, stop, step, endpoint=True)

        # line plot
        x = np.array([[int(x.split()[-1][:-1]) for x in x_axis if int(x.split()[0][1:-1]) == w] for w in wvl[:-1]])
        y = np.array([[er for er, x in zip(error, x_axis) if int(x.split()[0][1:-1]) == w] for w in wvl[:-1]])

        # mat plot
        res_mat = np.empty((len(wvl), len(wvl)))
        res_mat.fill(np.nan)

        for j, w in enumerate(wvl[:-1]):
            tmp = np.array([er for er, x in zip(error, x_axis) if int(x.split()[0][1:-1]) == w])
            res_mat[1 + j:1 + j + np.size(y), j] = tmp

        return x, y, res_mat

    def line_plot(ax, x: np.ndarray, y: np.ndarray) -> None:
        cm = plt.get_cmap("gist_rainbow")
        num_colors = len(y)
        ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])

        wvl = safe_arange(start, stop, step, endpoint=True, dtype=int)

        for j in range(num_colors):
            ax.plot(x[j], y[j], "--o", label="From {:.1f} nm".format(wvl[j]))
        ax.set_xticks(wvl[1:])
        ax.set_xlabel("To wavelength (nm)")

        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0,
                  ncol=best_blk(num_colors)[1])

        ax.set_ylabel(error_label)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=False, left=True, right=False)

    def matrix_plot(ax_matrix, res_mat: np.ndarray) -> None:
        if "RMSE" in error_type:
            vmin = np.nanmin(res_mat)
            vmax = np.nanmin((np.nanmax(res_mat), np.nanmin(res_mat) + 10))
            cmap = "jet_r"
        else:
            vmin = np.nanmax((np.nanmin(res_mat), np.nanmax(res_mat) - 20))
            vmax = np.nanmax(res_mat)
            cmap = "jet"

        wvl = safe_arange(start, stop, step, endpoint=True, dtype=int)
        im = ax_matrix.imshow(res_mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        ax_matrix.set_xticks(safe_arange(len(wvl)))
        ax_matrix.set_yticks(safe_arange(len(wvl)))

        ax_matrix.set_xticklabels(wvl, rotation=90, ha="center")
        ax_matrix.set_yticklabels(wvl)

        ax_matrix.set_xlabel("From wavelength (nm)")
        ax_matrix.set_ylabel("To wavelength (nm)")
        ax_matrix.xaxis.set_label_position("bottom")
        ax_matrix.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                              bottom=True, top=False, left=True, right=False)

        divider = make_axes_locatable(ax_matrix)
        cax = divider.append_axes(**cbar_kwargs)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(error_label)

    x, error = load_data_for_plot(file_name, error_type)
    x, y, mat = prepare_data(x, error)

    error_type = error_type.replace("_nooutliers", "")
    tmp = error_type.split("_")

    if "RMSE" in error_type:
        error_label = "".join((tmp[0], " (pp; ", " ".join(tmp[1:]), ")"))
    else:
        error_label = "".join((tmp[0], " (", tmp[1], " ", tmp[2], "; ", " ".join(tmp[3:]), ")"))

    fig_name_suffix = "".join(("spacing", "_", error_type, ".", fig_format))

    if "line" in line_plot_or_matrix:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        line_plot(ax, x, y)
        fig_name = "".join(("range_line_", fig_name_suffix))

    elif "matrix" in line_plot_or_matrix:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        matrix_plot(ax, mat)
        fig_name = "".join(("range_matrix_", fig_name_suffix))

    elif "both" in line_plot_or_matrix:
        fig, ax = plt.subplots(1, 2, figsize=(25, 10), gridspec_kw={"width_ratios": [2.3, 1]})
        line_plot(ax[0], x, y)
        matrix_plot(ax[1], mat)
        fig_name = "".join(("range_plot_", fig_name_suffix))
    else:
        raise ValueError('"line_plot_or_matrix" must be one of "both", "line", "matrix".')

    plt.draw()
    plt.tight_layout()

    outdir_range_tests = "".join((outdir, "/range_test/"))
    check_dir(outdir_range_tests)

    fig.savefig("".join((outdir_range_tests, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_error_range(file_names: list[str], error_type: str = "RMSE_all", applied_function: Callable = np.mean) -> None:
    # load data
    def load_data_for_plot(file_names: list[str], error_type: str) -> tuple[np.ndarray, ...]:
        subfolder = "range_test/"
        for i, file_name in enumerate(file_names):
            filename = "".join((_path_data, subfolder, file_name))
            data = pd.read_csv(filename, sep="\t")  # to read the file

            # select wanted metrics
            indices = data["index"].to_numpy()
            spacing_all = data["spacing"].to_numpy()
            error_all = data[error_type].to_numpy()

            """
            # remove NaNs
            inds_not_nan = np.argwhere(np.isfinite(error_all))
            indices = indices[inds_not_nan]
            spacing_all = spacing_all[inds_not_nan]
            error_all = error_all[inds_not_nan]
            """

            # constant range => slightly different spacing for each dataset
            spacing, error = combine_same_range_models(indices, spacing_all, error_all, applied_function)

            if i == 0:
                x_axis = spacing
                error_all_data = error
            else:
                x_axis = stack((x_axis, spacing), axis=0)
                error_all_data = stack((error_all_data, error), axis=0)

        return x_axis, error_all_data

    def line_plot(ax, x: np.ndarray, y: np.ndarray) -> None:
        cm = plt.get_cmap("gist_rainbow")
        num_colors = int(np.ceil(len(y) / 2.))  # SWIR full lines; no SWIR dashed lines
        ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])

        # extract ranges from file_names
        pos_under = [[m.start() for m in re.finditer("_", file)] for file in file_names]
        labels = ["".join((file[under[1] + 1:under[2]], "\u2013", file[under[2] + 1:-4]))
                  for file, under in zip(file_names, pos_under)]
        styles = ["-or", "--or", "-ob", "--ob"]
        for x_val, y_val, style, label in zip(x, y, styles, labels):
            ax.plot(x_val, y_val, style, label="".join((label, " nm")))

        ax.set_xlabel("Spacing (nm)")
        ax.legend(ncol=2)

        ax.set_ylabel(error_label)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=False, left=True, right=False)

    x, y = load_data_for_plot(file_names, error_type)

    error_type = error_type.replace("_nooutliers", "")
    tmp = error_type.split("_")

    if "RMSE" in error_type:
        error_label = "".join((tmp[0], " (pp; ", " ".join(tmp[1:]), ")"))
    else:
        error_label = "".join((tmp[0], " (", tmp[1], " ", tmp[2], "; ", " ".join(tmp[3:]), ")"))

    fig_name_suffix = "".join(("range", "_", error_type, ".", fig_format))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    line_plot(ax, x, y)
    fig_name = "".join(("range_line_", fig_name_suffix))

    plt.draw()
    plt.tight_layout()

    outdir_range_tests = "".join((outdir, "/range_test/"))
    check_dir(outdir_range_tests)

    fig.savefig("".join((outdir_range_tests, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_error_window(file_name: str, error_type: str = "RMSE", applied_function: Callable = np.mean) -> None:
    from modules.NN_config_range_test import start, stop, _spacing, window_size

    # load data
    def load_data_for_plot(file_name: str) -> tuple[np.ndarray, ...]:
        subfolder = "range_test/"
        filename ="".join((_path_data, subfolder, file_name))
        data = pd.read_csv(filename, sep="\t")  # to read the file

        # select metrices
        indices = data["index"].to_numpy()
        ranges_all = data["range"].to_numpy()

        if "RMSE" in error_type:
            np.arange(3, 12)
            inds = np.arange(3, 12)
        elif "within_5" in error_type:
            inds = np.arange(21, 30)
        elif "within_10" in error_type:
            inds = np.arange(30, 39)
        elif "within_15" in error_type:
            inds = np.arange(39, 48)
        elif "within_20" in error_type:
            inds = np.arange(48, 57)
        else:
            raise ValueError('Invalid "error_type".')

        error_all = np.array(data.to_numpy()[:, inds], dtype=np.float)
        labels = np.array(data.keys()[inds])

        x_axis = np.array(["".join((str(window_start), "\u2013", str(window_start + window_size)))
                           for window_start in safe_arange(start, stop, _spacing)
                           if window_start + window_size <= stop])

        # this function must be applied column by column
        for i in range(np.shape(error_all)[1]):
            _, error = combine_same_range_models(indices, ranges_all, error_all[:, i], applied_function)
            error = np.reshape(error, (len(error), 1))

            if i == 0:
                error_all_data = error
            else:
                error_all_data = stack((error_all_data, error), axis=1)

        return x_axis, error_all_data, labels

    def line_plot(ax, x: np.ndarray, y: np.ndarray) -> None:
        cm = plt.get_cmap("gist_rainbow")
        num_colors = np.shape(y)[1]
        ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])

        ax.plot(y, "--o")

        ax.set_xticks(safe_arange(len(x)))
        ax.set_xticklabels(x, ha="center", fontsize=10)
        ax.set_xlabel("Window range (nm)")

        labels = np.array(["ALL", "OL", "OPX", "CPX", "Fa", "Fs OPX", "Fs CPX", "En CPX", "Wo CPX"])

        ax.legend(labels, bbox_to_anchor=(0., 1.02, 1., 0.2), loc="lower left", mode="expand", borderaxespad=0.,
                  ncol=best_blk(num_colors)[1])

        ax.set_ylabel(error_label)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=False, left=True, right=False)

    x, y, labels = load_data_for_plot(file_name)

    error_type = error_type.replace("_nooutliers", "")
    tmp = error_type.split("_")

    if "RMSE" in error_type:
        error_label = "".join((tmp[0].capitalize(), " (pp)"))
    else:
        error_label = "".join((tmp[0].capitalize(), " (", tmp[1], " ", tmp[2], ")"))

    fig_name_suffix = "".join(("window", "_", str(int(window_size)), "_", error_type, "_all.", fig_format))

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    line_plot(ax, x, y)
    fig_name = "".join(("range_line_", fig_name_suffix))

    plt.draw()
    plt.tight_layout()

    outdir_range_tests = "".join((outdir, "/range_test/"))
    check_dir(outdir_range_tests)

    fig.savefig("".join((outdir_range_tests, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)
