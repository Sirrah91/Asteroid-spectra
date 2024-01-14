from modules.control_plots import *

from typing import Literal
import numpy as np
from copy import deepcopy
import matplotlib.patches as patches
import cv2
from glob import glob
from os import path
from scipy.interpolate import interp1d
# from matplotlib.ticker import FormatStrFormatter

from modules.BAR_BC_method import calc_BAR_BC, calc_composition, filter_data_mask

from modules.NN_data import numbers_to_classes

from modules.utilities import check_dir, best_blk, distance, stack, normalise_array, is_empty, find_all
from modules.utilities_spectra import (error_estimation_bin_like, cut_error_bars, find_outliers, used_indices,
                                       unique_indices, join_data, load_npz, load_txt, normalise_spectra)
from modules.tables import print_grid_test_stats_range

from modules.NN_config_parse import gimme_num_minerals, gimme_endmember_counts

from modules._constants import _path_data, _path_asteroid_images, _sep_out, _sep_in, _label_true_name, _label_pred_name
from modules._constants import _spectra_name, _wavelengths_name, _label_name, _config_name
from modules._constants import _path_accuracy_tests


def plot_PC1_PC2_NN(y_pred: np.ndarray, offset: float = 0.) -> None:
    print("Plot PC1 vs PC2 for asteroids")

    change_params(offset)

    cmap = "viridis_r"  # cmap of points
    s = 30.  # scaling parameter (marker size)
    vmin, vmax = 0., 100.  # minimum and maximum for colormap in the scatter plots of chemical

    # annotation
    start, end = np.array([0.5, -0.5]), np.array(([0.0, 0.8]))
    shift = 0.03

    data = load_npz(f"asteroid{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz", subfolder="taxonomy")
    metadata = join_data(data, "meta")
    types = np.ravel(data[_label_name])

    inds_S = np.array(["S" in ast_type for ast_type in types])
    inds_Q = np.array(["Q" in ast_type for ast_type in types])
    inds_A = np.array(["A" in ast_type for ast_type in types])
    inds_V = np.array(["V" in ast_type for ast_type in types])

    inds = inds_S + inds_Q + inds_V + inds_A

    # ignore Sq: since its spectrum is very weird
    inds[types == "Sq:"] = False

    # unique, counts = np.unique(types[inds], return_counts=True)
    PCA = np.array(metadata[["PC1", "PC2"]], dtype=float)[inds]
    predictions = y_pred[inds] * 100.

    labels = np.core.defchararray.add("$", types[inds])
    labels = np.core.defchararray.add(labels, "$")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Why I cannot set marker=labels and do it all as for vectors??
    for pca, label, pred in zip(PCA, labels, predictions):
        if len(label) == 3:
            fact = 1.
        if len(label) == 4:
            fact = 2.5
        if len(label) == 5:
            fact = 5.5

        c = pred[0]  # olivine
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

    cax = divider.append_axes(**{'position': 'right', 'size': '5%', 'pad': 1})
    SP = deepcopy(sp)
    SP.set_cmap("viridis")
    SP.set_clim(vmin=vmax, vmax=vmin)
    cbar = plt.colorbar(SP, ax=ax, cax=cax)
    cbar.ax.set_ylabel("Pyroxene abundance (vol\%)")

    fig.tight_layout()  # Otherwise the right y-label is slightly clipped
    plt.draw()

    fig_name = f"PCA_plot.{fig_format}"
    fig.savefig(path.join(outdir_composition, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_PC1_PC2_BAR(offset: float = 0.) -> None:
    change_params(offset)

    cmap = "viridis_r"  # cmap of points
    s = 30.  # scaling parameter (marker size)
    vmin, vmax = 0., 100.  # minimum and maximum for colormap in the scatter plots of chemical

    # annotation
    start, end = np.array([0.5, -0.5]), np.array(([0.0, 0.8]))
    shift = 0.03

    data = load_npz(f"asteroid{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz", subfolder="taxonomy")

    metadata, wvl, types, data = join_data(data, "meta"), data[_wavelengths_name], np.ravel(data[_label_name]), data[_spectra_name]

    inds_S = np.array(["S" in ast_type for ast_type in types])
    inds_Q = np.array(["Q" in ast_type for ast_type in types])
    inds_A = np.array(["A" in ast_type for ast_type in types])
    inds_V = np.array(["V" in ast_type for ast_type in types])

    inds = inds_S + inds_Q + inds_V + inds_A

    # ignore Sq: since its spectrum is very weird
    inds[types == "Sq:"] = False

    types = types[inds]

    x_data = data[inds]

    PCA = np.array(metadata[["PC1", "PC2"]], dtype=float)[inds]

    labels = np.core.defchararray.add("$", types)
    labels = np.core.defchararray.add(labels, "$")

    BAR, BIC, BIIC = calc_BAR_BC(wvl, x_data)

    # remove nans
    mask = np.logical_and.reduce((BAR > 0., BIC > 0., BIIC > 0.))
    BAR, BIC, BIIC = BAR[mask], BIC[mask], BIIC[mask]
    labels = labels[mask]
    types = types[mask]
    PCA = PCA[mask]

    OL_fraction, Fs, Wo = calc_composition(BAR, BIC, BIIC, types, method="bic")

    # filter the data
    mask = filter_data_mask(OL_fraction, Fs, Wo, modal_only=True)
    OL_fraction = OL_fraction[mask]
    labels = labels[mask]
    # types = types[mask]
    PCA = PCA[mask]

    # unique, counts = np.unique(types, return_counts=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

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

    cax = divider.append_axes(**{'position': 'right', 'size': '5%', 'pad': 1})
    SP = deepcopy(sp)
    cmap_reverse = cmap.replace("_r", "") if "_r" in cmap else f"{cmap}_r"
    SP.set_cmap(cmap_reverse)
    SP.set_clim(vmin=vmax, vmax=vmin)
    cbar = plt.colorbar(SP, ax=ax, cax=cax)
    cbar.ax.set_ylabel("Pyroxene abundance (vol\%)")

    fig.tight_layout()  # Otherwise the right y-label is slightly clipped
    plt.draw()

    fig_name = f"PCA_plot_BAR_BC.{fig_format}"
    fig.savefig(path.join(outdir_composition, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_Fa_vs_Fs_ast_only(offset: float = 0.) -> None:
    print("Plot Fa vs Fs")

    change_params(offset)

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

    # definition of boxes from (for some reason should be used just once)
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

    # Add the patches
    ax.add_patch(H_rect)
    ax.add_patch(L_rect)
    ax.add_patch(LL_rect)

    # Artificial data to get labels for legend
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

    fig_name = f"Fa_vs_Fs_ast.{fig_format}"
    fig.savefig(path.join(outdir_composition, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_Fa_vs_Fs(y_true: np.ndarray, y_pred: np.ndarray, meta: pd.DataFrame,
                  used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                  print_asteroids: bool = False, offset: float = 0.) -> None:
    print("Plot Fa vs Fs")

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    change_params(offset)

    limx1, limx2 = 15., 35.
    limy1, limy2 = 10., 30.

    shift = 3.  # Control ranges of axes
    s = 40.  # scaling parameter

    xticks, yticks = safe_arange(limx1, limx2, 5., endpoint=True), safe_arange(limy1, limy2, 5., endpoint=True)
    left, right = limx1 - shift, limx2 + shift
    bottom, top = limy1 - shift, limy2 + shift

    RMSE = compute_metrics(y_true, y_pred, return_r2=False, return_sam=False,
                           used_minerals=used_minerals, used_endmembers=used_endmembers)

    if print_asteroids:
        # NEED TO BE SET ACCORDING TO MODEL PREDICTIONS
        Fa_S, Fs_S = [20.7], [18.3]
        Fa_Sq, Fs_Sq = [23.2], [21.0]
        Fa_Sr, Fs_Sr = [18.3], [19.2]
        Fa_Sw, Fs_Sw = [21.3], [14.7]
        Fa_Q, Fs_Q = [26.2], [23.8]

    # Indices in which Fa and Fs are
    num_minerals = gimme_num_minerals(used_minerals)
    count_endmembers = gimme_endmember_counts(used_endmembers)

    ind_Fa, ind_Fs = num_minerals, num_minerals + count_endmembers[0]

    error_Fa, error_Fs = RMSE[ind_Fa], RMSE[ind_Fs]

    types = np.array(meta["SubType"], dtype=str)

    Fa_true, Fs_true = y_true[:, ind_Fa] * 100., y_true[:, ind_Fs] * 100.
    Fa_pred, Fs_pred = y_pred[:, ind_Fa] * 100., y_pred[:, ind_Fs] * 100.

    inds_H = np.array([("H" in OC_type) and ("HH" not in OC_type) if len(OC_type) == 2 else False for OC_type in types])
    inds_L = np.array([("L" in OC_type) and ("LL" not in OC_type) if len(OC_type) == 2 else False for OC_type in types])
    inds_LL = np.array(["LL" in OC_type if len(OC_type) == 3 else False for OC_type in types])

    fig, ax = plt.subplots(1, 2, figsize=(4.5 * 2, 6), sharex=True)

    for i, axis in enumerate(ax):
        # definition of boxes from (for some reason should be used just once)
        # https://www.researchgate.net/figure/Mineral-compositional-data-for-the-Morokweng-meteoriteDiagram-shows-ferrosilite-Fs_fig3_7093505
        H_rect = patches.Rectangle((16.2, 14.5), 3.8, 3.5, linewidth=1.5, edgecolor="r", facecolor="none")
        L_rect = patches.Rectangle((22.0, 19.0), 4.0, 3.0, linewidth=1.5, edgecolor="g", facecolor="none")
        LL_rect = patches.Rectangle((26.0, 22.0), 6.0, 4.2, linewidth=1.5, edgecolor="b", facecolor="none")

        if i == 0:
            axis.scatter(Fa_true[inds_H], Fs_true[inds_H], c="r", s=s, label="H")
            axis.scatter(Fa_true[inds_L], Fs_true[inds_L], c="g", s=s, label="L")
            axis.scatter(Fa_true[inds_LL], Fs_true[inds_LL], c="b", s=s, label="LL")
        else:
            axis.scatter(Fa_pred[inds_H], Fs_pred[inds_H], c="r", s=s, label="H")
            axis.scatter(Fa_pred[inds_L], Fs_pred[inds_L], c="g", s=s, label="L")
            axis.scatter(Fa_pred[inds_LL], Fs_pred[inds_LL], c="b", s=s, label="LL")

            axis.errorbar(Fa_pred[inds_H], Fs_pred[inds_H], xerr=error_Fa, yerr=error_Fs, c="r", fmt="o")
            axis.errorbar(Fa_pred[inds_L], Fs_pred[inds_L], xerr=error_Fa, yerr=error_Fs, c="g", fmt="o")
            axis.errorbar(Fa_pred[inds_LL], Fs_pred[inds_LL], xerr=error_Fa, yerr=error_Fs, c="b", fmt="o")

        axis.set_xlabel("Fa")
        if i == 0:
            axis.set_ylabel("Fs")
            axis.set_title("Ordinary chondrites")
        else:
            axis.set_title("Predictions")

        axis.tick_params(axis="both")
        axis.axis("square")

        axis.set_xticks(xticks)
        axis.set_yticks(yticks)
        axis.set_xlim(left=left, right=right)
        axis.set_ylim(bottom=bottom, top=top)

        # Add the patches
        axis.add_patch(H_rect)
        axis.add_patch(L_rect)
        axis.add_patch(LL_rect)

        axis.legend()

    if print_asteroids:
        ax[1].scatter(Fa_S, Fs_S, marker="$S$", c="k", s=s * 2.5)
        ax[1].scatter(Fa_Sq, Fs_Sq, marker="$Sq$", c="k", s=s * 5)
        ax[1].scatter(Fa_Sr, Fs_Sr, marker="$Sr$", c="k", s=s * 5)
        ax[1].scatter(Fa_Sw, Fs_Sw, marker="$Sw$", c="k", s=s * 6)

        ax[1].scatter(Fa_Q, Fs_Q, marker="$Q$", c="k", s=s * 2.5)

    plt.draw()
    plt.tight_layout()

    fig_name = f"Fa_vs_Fs.{fig_format}"
    fig.savefig(path.join(outdir_composition, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_error_evaluation_ol_opx(offset:float = 0., suf: str = "pures", quiet: bool = False) -> None:
    if not quiet:
        print("Print quantiles")

    change_params(offset)

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

    fig_name = f"quantile_error_plot_{suf}.{fig_format}"
    fig.savefig(path.join(outdir_composition, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_scatter_NN_BC(x_test: np.ndarray, y_test: np.ndarray, meta_test: pd.DataFrame, y_pred: np.ndarray,
                       wvl: np.ndarray, used_minerals: np.ndarray | None = None,
                       used_endmembers: list[list[bool]] | None = None, offset: float = 0.) -> None:
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    change_params(offset)

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

    types = np.array(meta_test["SubType"], dtype=str)
    # Only those that contain OL and OPX (mixtures 12 and 14)
    binary = np.array(y_test[:, :gimme_num_minerals(used_minerals)] > 0, dtype=int)
    base = np.array([8, 4, 2, 1])[used_minerals]
    mixtures = np.sum(binary * base, axis=1)

    mask1 = np.logical_or(mixtures == 12, mixtures == 14)
    # Only those with CPX <= 10
    mask2 = y_test[:, 2] <= max_cpx

    mask = np.logical_and(mask1, mask2)

    predicted_errorbar_reduced, actual_errorbar_reduced = error_estimation_bin_like(y_test, y_pred,
                                                                                    actual_error=true_errorbar,
                                                                                    used_minerals=used_minerals,
                                                                                    used_endmembers=used_endmembers)

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
    mask = np.logical_and.reduce((BAR > 0., BIC > 0., BIIC > 0.))
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
    for axis in ax:
        axis.plot(x_line, y_line, l0, label=lab_line0, linewidth=LW)
        axis.plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW)
        axis.plot(x_line, y1m_line, l10, linewidth=LW)
        axis.plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW)
        axis.plot(x_line, y2m_line, l20, linewidth=LW)

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

    for axis in ax:
        axis.tick_params(axis="both")
        axis.axis("square")

        axis.set_xticks(xticks)
        axis.set_yticks(yticks)
        axis.set_xlim(left=left, right=right)
        axis.set_ylim(bottom=bottom, top=top)

        axis.legend(loc="upper left", frameon=False)

    plt.draw()
    plt.tight_layout()

    fig_name = f"scatter_plot_NN_BAR_BC_met_mix.{fig_format}"
    fig.savefig(path.join(outdir_composition, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_spectra_1(x_data: np.ndarray, y_data: np.ndarray, offset: float = 0.) -> None:
    # Plot spectra of minerals and B-D asteroids
    change_params(offset)

    xticks, yticks = safe_arange(0.5, 2.5, 0.5, endpoint=True), safe_arange(0., 5., endpoint=True)
    left, right = 0.4, 2.500
    bottom, top = 0.0, 5.0
    x = safe_arange(450., 2450., 5., endpoint=True) / 1000.  # um
    titles = ["Olivine", "Orthopyroxene", "Clinopyroxene", "Laboratory mixtures", "Meteorites", "Asteroids"]

    m, n = best_blk(len(titles))

    # asteroid data
    data = load_npz(f"asteroid{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz", subfolder="taxonomy")
    types = np.ravel(data[_label_name])

    inds_S = np.array(["S" in ast_type for ast_type in types])
    inds_Q = np.array(["Q" in ast_type for ast_type in types])
    inds_A = np.array(["A" in ast_type for ast_type in types])
    inds_V = np.array(["V" in ast_type for ast_type in types])

    inds = inds_S + inds_Q + inds_V + inds_A

    # ignore Sq: since its spectrum is very weird
    inds[types == "Sq:"] = False

    ast_data = data[_spectra_name][inds]

    fig, ax = plt.subplots(m, n, figsize=(4.7 * n, 4.7 * m), sharex=True, sharey=True, squeeze=False)

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

        if k == 0:
            ax[i, k].set_ylabel("Reflectance (normalised)")

        if i > 0:
            ax[i, k].set_xlabel("Wavelength (\\textmu{}m)")

    plt.draw()
    plt.tight_layout()

    fig_name = f"spectra_all_1.{big_fig_format}"
    fig.savefig(path.join(outdir, fig_name), format=big_fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close()

    change_params(offset, reset=True)


def plot_spectra_2(offset: float = 0.) -> None:
    # Plot spectra of B-D asteroids and Itokawa and Eros
    change_params(offset)

    # figure limits
    bottom, top = 0.0, 1.5
    left, right = 0.4, 2.5  # um

    norm_at = 1.5  # um

    # text settings
    text_kwargs = {"x": 0.5, "y": 0.15, "horizontalalignment": "center", "verticalalignment": "center",
                   "fontsize": MEDIUM_SIZE, "c": "k"}
    s = lambda x: f"{len(x)} spectra"  # text string

    xticks, yticks = safe_arange(0., right, 0.5, endpoint=True), safe_arange(0., top, 0.5, endpoint=True)

    # asteroid data
    data = load_npz(f"asteroid{_sep_in}spectra{_sep_out}16{_sep_out}reduced{_sep_out}denoised{_sep_out}norm.npz")

    # _metadata_name cannot be here because it contains original types
    types = np.ravel(np.array(data[_label_name], dtype=str))

    titles = list(np.unique(types))
    titles.append("Eros")
    titles.append("Itokawa")

    m, n = best_blk(len(titles), cols_to_rows=6. / 3.)

    fig, ax = plt.subplots(m, n, figsize=(6 * n, 8 * m), sharey=True, squeeze=False)

    for k, unique_type in enumerate(np.unique(types)):
        inds_class = np.array([unique_type == ast_type for ast_type in types])
        spectra, wavelengths = data[_spectra_name][inds_class], data[_wavelengths_name] / 1000.  # um
        spectra = normalise_spectra(spectra, wavelengths, norm_at, on_pixel=False)  # normalise
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

        if j == 0:
            ax[i, j].set_ylabel("Reflectance (normalised)")

        if i == m - 1:
            ax[i, j].set_xlabel("Wavelength (\\textmu{}m)")

        ax[i, j].text(s=s(spectra), transform=ax[i, j].transAxes ,**text_kwargs)

    # Itokawa and Eros data
    j += 1
    data = load_npz(f"Eros{_sep_out}denoised{_sep_out}norm.npz")
    spectra, wavelengths = data[_spectra_name], data[_wavelengths_name] / 1000.  # um
    spectra = normalise_spectra(spectra, wavelengths, norm_at, on_pixel=False)  # normalise
    avg_spectrum = np.mean(spectra, axis=0)

    ax[i, j].plot(wavelengths, np.transpose(spectra), "-k")
    ax[i, j].plot(wavelengths, avg_spectrum, "-r", linewidth=3)

    ax[i, j].tick_params(axis="both")
    ax[i, j].set_title(titles[k + 1])

    ax[i, j].set_xticks(xticks)
    ax[i, j].set_yticks(yticks)
    ax[i, j].set_xlim(left=left, right=right)
    ax[i, j].set_ylim(bottom=bottom, top=top)

    ax[i, j].set_xlabel("Wavelength (\\textmu{}m)")

    ax[i, j].text(s=s(spectra), transform=ax[i, j].transAxes ,**text_kwargs)

    j += 1
    data = load_npz(f"Itokawa{_sep_out}denoised{_sep_out}norm.npz")
    spectra, wavelengths = data[_spectra_name], data[_wavelengths_name] / 1000.  # um
    spectra = normalise_spectra(spectra, wavelengths, norm_at, on_pixel=False)  # normalise
    avg_spectrum = np.mean(spectra, axis=0)

    ax[i, j].plot(wavelengths, np.transpose(spectra), "-k")
    ax[i, j].plot(wavelengths, avg_spectrum, "-r", linewidth=3)

    ax[i, j].tick_params(axis="both")
    ax[i, j].set_title(titles[k + 2])

    ax[i, j].set_xticks(xticks)
    ax[i, j].set_yticks(yticks)
    ax[i, j].set_xlim(left=left, right=right)
    ax[i, j].set_ylim(bottom=bottom, top=top)

    ax[i, j].set_xlabel("Wavelength (\\textmu{}m)")

    ax[i, j].text(s=s(spectra), transform=ax[i, j].transAxes ,**text_kwargs)

    plt.draw()
    plt.tight_layout()

    fig_name = f"spectra_all_2.{big_fig_format}"
    fig.savefig(path.join(outdir, fig_name), format=big_fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close()

    change_params(offset, reset=True)


def compare_spectra_with_instrument(filename: str,
                                    instrument: str,
                                    classes_to_plot: list[str] | None = None,
                                    subfolder: str = "",
                                    offset: float = 0.) -> None:
    from modules.NN_data import reinterpolate_data

    change_params(offset)

    data = load_npz(filename, subfolder=subfolder)

    wavelengths, spectra, labels = data[_wavelengths_name], data[_spectra_name], np.ravel(data[_label_name])

    if classes_to_plot is None: classes_to_plot = np.unique(labels)

    # compute mean spectra for each taxonomy class
    avg_spectra = np.array([np.mean(spectra[label == labels], axis=0) for label in classes_to_plot])

    # apply instrument transmission function
    wavelengths_intrument, avg_spectra_instrument = reinterpolate_data(avg_spectra, wavelengths, wvl_new_norm=None,
                                                                       instrument=instrument)

    # interpolate avg_spectra to wavelengths_instrument grid
    avg_spectra = interp1d(wavelengths, avg_spectra, kind="cubic")(wavelengths_intrument)

    rows, columns = best_blk(len(classes_to_plot))

    fig, ax = plt.subplots(rows, columns, figsize=(columns * 5.5, rows * 4.2), squeeze=False)
    fig.suptitle(f"{instrument}")

    for i in range(len(classes_to_plot)):
        r, c = np.unravel_index(i, (rows, columns))

        ax[r, c].plot(wavelengths_intrument, avg_spectra[i], 'r-', label="delta transmission")
        ax[r, c].plot(wavelengths_intrument, avg_spectra_instrument[i], 'b-', label=f"{instrument} transmission")

        if c == 0:
            ax[r, c].set_ylabel("Reflectance (normalised)")
        if r == rows - 1:
            ax[r, c].set_xlabel("Wavelengths (nm)")

        ax[r, c].set_title(classes_to_plot[i])
        ax[r, c].legend()

    plt.draw()
    fig.tight_layout()

    fig_name = f"compare_spectra_{instrument}.{fig_format}"
    fig.savefig(path.join(outdir, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close()

    change_params(offset, reset=True)


def plot_mineralogy_histogram(y_data: np.ndarray, offset: float = 0.) -> None:
    change_params(offset)

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

    for axis in ax:
        axis.set_xlim(left=left, right=right)
        axis.tick_params(axis="both")
        axis.legend(loc="best")

    plt.draw()
    plt.tight_layout()

    fig_name = f"spectra_min_hist.{fig_format}"
    fig.savefig(path.join(outdir_composition, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close()

    change_params(offset, reset=True)


def plot_surface_spectra(y_pred: np.ndarray, filename: str, what_type: Literal["composition", "taxonomy"],
                         used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                         used_classes: dict[str, int] | None = None, offset: float = 0.) -> None:
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used
    if used_classes is None: used_classes = classes

    change_params(offset)

    def get_most_probable_classes(limit: float = 1.) -> tuple[int, np.ndarray]:
        # limit: minimum averaged probability
        n_probable_classes = np.sum(mean_of_predictions >= limit)
        most_probable_classes = np.argsort(mean_of_predictions)[-n_probable_classes:][::-1]

        print("\nMost probable classes:")
        for cls in most_probable_classes:
            print(f"{numbers_to_classes(cls, used_classes=used_classes):4} {np.round(mean_of_predictions[cls], 2):5.2f}%")

        return n_probable_classes, most_probable_classes

    def get_most_winning_classes(limit: float = 0.) -> tuple[int, np.ndarray]:
        # limit: minimum most probable predictions
        unique, counts = np.unique(gimme_predicted_class(y_pred, used_classes=used_classes, return_index=True),
                                   return_counts=True)
        unique, counts = unique[counts >= limit], counts[counts >= limit]

        n_probable_classes = np.size(unique)
        most_probable_classes = unique[np.argsort(counts)][::-1]

        norm_counts = np.sort(normalise_array(counts, norm_constant=100.))[::-1]
        print("\nMost winning classes:")
        for icls, cls in enumerate(most_probable_classes):
            print(f"{numbers_to_classes(cls, used_classes=used_classes):4} {np.round(norm_counts[icls], 2):5.2f}%")

        return n_probable_classes, most_probable_classes

    font_size_axis = 36

    cmap = "viridis_r"
    vmin, vmax = 0., 100.
    alpha = 0.4
    s = 10.

    xticks = safe_arange(0., 360., 10., endpoint=True)
    yticks = safe_arange(-90., 90., 10., endpoint=True)
    left, right = 0.0, 360.
    bottom, top = -90., 90.

    cticks = safe_arange(0., 100., 10., endpoint=True)
    ctickslabel = safe_arange(0., 100., 10., endpoint=True, dtype=int)

    if "Itokawa" in filename:
        background_image = "new_itokawa_mosaic.jpg"
        name = "Itokawa"
    elif "Eros" in filename:
        background_image = "eros_cyl_near.jpg"
        name = "Eros"
    else:
        raise ValueError('"filename" must contain either "Itokawa" or "Eros"')

    indices_file = load_npz(filename)
    indices = join_data(indices_file, "meta")
    indices = np.array(indices[["longitude", "latitude"]], dtype=float)

    mean_of_predictions = np.mean(y_pred, axis=0) * 100.

    if what_type == "taxonomy":
        _, most_probable_classes_1 = get_most_probable_classes()
        _, most_probable_classes_2 = get_most_winning_classes()
        most_probable_classes = stack((most_probable_classes_1,
                                       np.setdiff1d(most_probable_classes_2, most_probable_classes_1)))

        labels = numbers_to_classes(most_probable_classes, used_classes=used_classes)
        titles = [f"{name} {label}-type predictions" for label in labels]

    elif what_type == "composition":
        # set titles (this should work well)

        titles_all = [mineral_names] + endmember_names
        titles_all = flatten_list(titles_all)[used_indices(used_minerals, used_endmembers)]
        # titles_all = flatten_list(titles_all)[unique_indices(used_minerals, used_endmembers, all_minerals=True)]

        most_probable_classes = unique_indices(used_minerals, used_endmembers, return_digits=True)
        labels = titles_all[most_probable_classes]

        print("\nSelected mineralogy:")
        for i, cls in enumerate(most_probable_classes):
            print(f"{labels[i]:14} {np.round(mean_of_predictions[cls], 2):5.2f}%")

        titles = [f"{name} {label} predictions" for label in labels]

    else:
        raise ValueError('"what_type" must be either "taxonomy" or "composition"')

    # Color code dominant classes / labels
    probability_values = np.transpose(np.array([y_pred[:, icls] for icls in most_probable_classes]))

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

    for i, label in enumerate(labels):
        # Plot the coverage map using latitude and longitude from HB
        img = plt.imread(path.join(_path_asteroid_images, background_image))  # Background image
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
                           marker=f"${np.round(crater[0], 0):.0f}$", zorder=100)

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
            cbar.ax.set_xlabel("Match score (\%)", fontsize=font_size_axis)
        elif i < gimme_num_minerals(used_minerals):
            cbar.ax.set_xlabel("Modal abundance (vol\%)", fontsize=font_size_axis)

        plt.draw()
        plt.tight_layout()

        fig_name = f"{name}_{label.replace(' ', '_')}.{big_fig_format}"
        fig.savefig(path.join(outdir_surfaces, fig_name), format=big_fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close("all")

    change_params(offset, reset=True)


def plot_surface_spectra_shapeViewer(asteroid_name: str, what_prediction: str | list[str] = "all",
                                     used_minerals: np.ndarray | None = None,
                                     used_endmembers: list[list[bool]] | None = None,
                                     used_classes: dict[str, int] | list | np.ndarray | None = None,
                                     offset: float = 0.) -> None:
    # figure order
    # "y" - east; "Y" - west; "x" - head; "X" - body, "z" - bottom, "Z" - top

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used
    if used_classes is None: used_classes = classes

    if isinstance(used_classes, dict):
        labels = list(used_classes.keys())
    else:
        labels = list(used_classes)

    change_params(offset)

    if asteroid_name == "Itokawa":
        faces = ["y", "x", "Y", "X"]
    else:
        faces = ["z", "x", "Y", "X"]

    rows, columns = best_blk(len(faces))

    alpha = 0.75 if fig_format == "pdf" else 0.5

    if what_prediction == "all":
        titles_all = [mineral_names] + endmember_names
        titles_all = flatten_list(titles_all)[used_indices(used_minerals, used_endmembers)]
        # titles_all = flatten_list(titles_all)[unique_indices(used_minerals, used_endmembers, all_minerals=True)]
        what_prediction = list(titles_all) + labels

        if asteroid_name == "Itokawa":
            what_prediction += ["grav", "slope"]

    elif "comp" in what_prediction:
        titles_all = [mineral_names] + endmember_names
        titles_all = flatten_list(titles_all)[used_indices(used_minerals, used_endmembers)]
        what_prediction = list(titles_all)

    elif "tax" in what_prediction:
        what_prediction = labels

    # Either f"min{_sep_out}max{_sep_out}cbar" or f"0{_sep_in}100{_sep_out}cbar"
    image_type = f"min{_sep_out}max{_sep_out}cbar"
    path_to_images = path.join(_path_asteroid_images, "shapeViewer", image_type)
    path_to_background = path.join(_path_asteroid_images, "shapeViewer")

    filenames = [f"{asteroid_name}{_sep_out}{face}.png" for face in faces]

    for which in what_prediction:

        reversed_caxis = which in ["Q", "olivine"]  # these two have reversed color axis

        fig, ax = plt.subplots(rows, columns, figsize=(16, 8), width_ratios=[2, 1])
        ax = np.ravel(ax)

        if which in labels:  # taxonomy map
            colorbar_title = "".join((which, "-type ", "match score (\\%)"))

        elif which == "grav":
            colorbar_title = " ".join(("Gravity acceleration", "(\\textmu{}m\,s$^{-2}$)"))

        elif which == "slope":
            colorbar_title = f"Gravity {which.lower()} (deg)"

        else:  # composition map
            if which in mineral_names:
                colorbar_title = " ".join((which, "abundance (vol\\%)"))
            else:
                colorbar_title = which

        if asteroid_name == "Itokawa":
            fig.suptitle("(25143) Itokawa", fontweight='bold')
        elif asteroid_name == "Eros":
            fig.suptitle("(433) Eros", fontweight='bold')
        else:
            pass

        colorbar_title = f"{colorbar_title[0].upper()}{colorbar_title[1:]}"

        cmap = "turbo"
        which = which.replace(" ", _sep_in)

        # to adjust colorbar
        if reversed_caxis:
            fname = path.join(_path_data, "shapeViewer", image_type,
                              f"{asteroid_name}{_sep_out}{which}{_sep_out}reversed.dat")
            y_pred = 100. - load_txt(fname, sep=" ", header=None).iloc[:, -1].to_numpy()
            cmap = f"{cmap}_r"
        else:
            fname = path.join(_path_data, "shapeViewer", image_type, f"{asteroid_name}{_sep_out}{which}.dat")
            y_pred = load_txt(fname, sep=" ", header=None).iloc[:, -1].to_numpy()

        for i, axis in enumerate(ax):
            fname = filenames[i].replace(_sep_out, f"{_sep_out}{which}{_sep_out}")
            fname = path.join(path_to_images, fname)

            if not path.isfile(fname):
                warnings.warn(f"The file {fname} doesn't exist.")
                plt.close(fig)
                break

            im_frame = cv2.imread(fname)[:, :, ::-1]  # BGR to RGB
            im_frame_bg = cv2.imread(path.join(path_to_background, filenames[i]))[:, :, ::-1]  # background BGR to RGB

            axis.imshow(im_frame_bg)
            sc = axis.imshow(im_frame, alpha=alpha)

            # cut the image
            lims = np.where(im_frame < 255)
            axis.set_xlim([np.min(lims[1]), np.max(lims[1])])
            axis.set_ylim([np.max(lims[0]), np.min(lims[0])])

            axis.axis("off")

            text_kwargs = {"horizontalalignment": "center", "verticalalignment": "center", "fontsize": MEDIUM_SIZE,
                           "transform": axis.transAxes, "c": "k"}

            if asteroid_name == "Itokawa":
                if i == 0:
                    axis.text(0.25, 0.28, "1", **text_kwargs)
                    axis.text(0.41, 0.59, "2", **text_kwargs)
                    axis.text(0.78, 0.66, "3", **text_kwargs)
                    axis.text(0.57, 0.44, "10", **text_kwargs)
                    axis.text(0.61, 0.81, "12", **text_kwargs)
                    axis.text(0.48, 0.28, "14", **text_kwargs)

                    """
                    # axis.text(0.74, 0.50, "4", **text_kwargs)
                    # axis.text(0.61, 0.83, "5", **text_kwargs)
                    axis.text(0.10, 0.64, "6", **text_kwargs)
                    # axis.text(0.10, 0.34, "7", **text_kwargs)
                    axis.text(0.74, 0.74, "9", **text_kwargs)
                    # axis.text(0.93, 0.38, "11", **text_kwargs)
                    # axis.text(0.88, 0.52, "14", **text_kwargs)
                    # axis.text(0.59, 0.40, "15", **text_kwargs)
                    # axis.text(0.03, 0.10, "21", **text_kwargs)
                    axis.text(0.985, 0.515, "22", **text_kwargs)
                    # axis.text(0.06, 0.13, "23", **text_kwargs)
                    axis.text(0.405, 0.59, "24", **text_kwargs)
                    axis.text(0.38, 0.66, "29", **text_kwargs)
                    # axis.text(0.615, 0.19, "30", **text_kwargs)
                    # axis.text(0.43, 0.27, "34", **text_kwargs)
                    # axis.text(0.61, 0.23, "36", **text_kwargs)
                    """

                elif i == 1:
                    axis.text(0.61, 0.65, "9", **text_kwargs)

                    """
                    # axis.text(0.38, 0.82, "1", **text_kwargs)
                    # axis.text(0.04, 0.465, "3", **text_kwargs)
                    # axis.text(0.98, 0.46, "4", **text_kwargs)
                    axis.text(0.60, 0.65, "6", **text_kwargs)
                    # axis.text(0.78, 0.33, "7", **text_kwargs)
                    # axis.text(0.92, 0.38, "15", **text_kwargs)
                    # axis.text(0.64, 0.1, "21", **text_kwargs)
                    # axis.text(0.72, 0.11, "23", **text_kwargs)
                    axis.text(0.83, 0.60, "24", **text_kwargs)
                    # axis.text(0.30, 0.15, "27", **text_kwargs)
                    # axis.text(0.08, 0.35, "28", **text_kwargs)
                    axis.text(0.81, 0.67, "29", **text_kwargs)
                    """

                elif i == 2:
                    axis.text(0.64, 0.54, "4", **text_kwargs)
                    axis.text(0.78, 0.34, "5", **text_kwargs)
                    axis.text(0.36, 0.48, "6", **text_kwargs)
                    axis.text(0.41, 0.77, "13", **text_kwargs)
                    axis.text(0.47, 0.52, "15", **text_kwargs)

                    """
                    # axis.text(0.48, 0.46, "3", **text_kwargs)
                    # axis.text(0.385, 0.155, "10", **text_kwargs)
                    axis.text(0.215, 0.71, "12", **text_kwargs)
                    # axis.text(0.455, 0.77, "13", **text_kwargs)
                    axis.text(0.135, 0.21, "16", **text_kwargs)
                    # axis.text(0.23, 0.165, "17", **text_kwargs)
                    # axis.text(0.295, 0.13, "18", **text_kwargs)
                    # axis.text(0.25, 0.12, "19", **text_kwargs)
                    axis.text(0.385, 0.33, "20", **text_kwargs)
                    axis.text(0.885, 0.17, "27", **text_kwargs)
                    # axis.text(0.485, 0.355, "28", **text_kwargs)
                    # axis.text(0.22, 0.81, "31", **text_kwargs)
                    # axis.text(0.255, 0.83, "32", **text_kwargs)
                    # axis.text(0.345, 0.15, "33", **text_kwargs)
                    axis.text(0.335, 0.56, "35", **text_kwargs)
                    # axis.text(0.60, 0.25, "37", **text_kwargs)
                    """

                elif i == 3:
                    axis.text(0.75, 0.69, "7", **text_kwargs)
                    axis.text(0.21, 0.52, "8", **text_kwargs)
                    axis.text(0.40, 0.35, "11", **text_kwargs)

                    """
                    # axis.text(0.42, 0.3, "2", **text_kwargs)
                    axis.text(0.405, 0.7, "8", **text_kwargs)
                    # axis.text(0.1, 0.75, "9", **text_kwargs)
                    # axis.text(0.095, 0.37, "11", **text_kwargs)
                    axis.text(0.82, 0.69, "12", **text_kwargs)
                    # axis.text(0.05, 0.55, "14", **text_kwargs)
                    # axis.text(0.765, 0.22, "16", **text_kwargs)
                    # axis.text(0.79, 0.17, "17", **text_kwargs)
                    axis.text(0.22, 0.5, "22", **text_kwargs)
                    axis.text(0.53, 0.72, "25", **text_kwargs)
                    # axis.text(0.395, 0.8, "26", **text_kwargs)
                    # axis.text(0.67, 0.8, "31", **text_kwargs)
                    # axis.text(0.68, 0.84, "32", **text_kwargs)
                    """
            elif asteroid_name == "Eros":
                if i == 0:
                    axis.text(0.86, 0.7, "1", **text_kwargs)
                    axis.text(0.42, 0.4, "2", **text_kwargs)

                elif i == 1:
                    axis.text(0.73, 0.65, "1", **text_kwargs)

                elif i == 2:
                    axis.text(0.45, 0.65, "2", **text_kwargs)

        else:
            sc.set_cmap(cmap)
            sc.set_clim(np.min(y_pred), np.max(y_pred))

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(sc, cax=cbar_ax, format='%d')
            cbar.ax.set_ylabel(colorbar_title)

            plt.draw()

            fig_name = f"{asteroid_name}_{which}_SV.{fig_format}"
            fig_name = fig_name.replace("(", "").replace(")", "")
            fig.savefig(path.join(outdir_surfaces, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
            plt.close(fig)

    change_params(offset, reset=True)


def plot_average_surface_spectra(y_pred: np.ndarray, asteroid_name: str,
                                 used_classes: dict[str, int] | list | np.ndarray | None = None,
                                 offset: float = 0.) -> None:
    if used_classes is None: used_classes = classes

    change_params(offset)

    data = load_npz(f"{asteroid_name}{_sep_out}denoised{_sep_out}norm.npz")

    spectra, wvl = data[_spectra_name], data[_wavelengths_name]

    predicted_classes = gimme_predicted_class(y_pred, used_classes=used_classes)
    unique_probable_classes = np.unique(predicted_classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in unique_probable_classes:
        ax.plot(wvl, np.mean(spectra[predicted_classes == cls], axis=0), label=cls)

    ax.set_ylabel("Reflectance")
    ax.set_xlabel("Wavelength (nm)")
    plt.legend()

    plt.draw()
    plt.tight_layout()

    fig_name = f"{asteroid_name}_average_spectra.{fig_format}"
    fig.savefig(path.join(outdir, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_Sq_histogram(y_true: np.ndarray, y_pred: np.ndarray, used_classes: dict[str, int] | None = None,
                      offset: float = 0.) -> None:
    if used_classes is None: used_classes = classes

    change_params(offset)

    alpha = 0.5
    nbins = safe_arange(0., 100., 10., endpoint=True)

    xticks = safe_arange(0., 100., 10., endpoint=True)
    left, right = 0.0, 100.

    inds = np.ravel(y_true) == "Sq"

    s_class = "S" if "S" in used_classes else "S+"
    q_class = "Q" if "Q" in used_classes else "Q+"
    probability_Q = y_pred[inds, used_classes[q_class]] * 100.
    probability_S = y_pred[inds, used_classes[s_class]] * 100.

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(probability_S, bins=nbins, alpha=alpha, label=s_class)
    ax.hist(probability_Q, bins=nbins, alpha=alpha, label=q_class)

    ax.set_xticks(xticks)
    ax.set_xlim(left=left, right=right)

    ax.set_xlabel("Match score (\%)")
    ax.set_ylabel("Counts")
    ax.legend()

    plt.draw()
    plt.tight_layout()

    fig_name = f"Sq_hist.{fig_format}"
    fig.savefig(path.join(outdir_taxonomy, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_EI_type_hist(y_Eros: np.ndarray, y_Itokawa: np.ndarray, tax_type: str,
                      used_classes: dict[str, int] | None = None, offset: float = 0.) -> None:
    if used_classes is None: used_classes = classes

    change_params(offset)

    nbins = safe_arange(0., 100., 5., endpoint=True)

    xticks = safe_arange(0., 100., 10., endpoint=True)
    left, right = 0.0, 100.

    inds = used_classes[tax_type]
    ei_list = [y_Eros[:, inds] * 100., y_Itokawa[:, inds] * 100.]
    titles = ["Eros", "Itokawa"]

    fig, ax = plt.subplots(1, len(ei_list), figsize=(12, 6))

    for i, data in enumerate(ei_list):
        ax[i].hist(data, bins=nbins, label=tax_type)

        ax[i].set_xticks(xticks)
        ax[i].set_xlim(left=left, right=right)

        ax[i].set_xlabel(f"{tax_type}-type match score (\%)")
        if i == 0:
            ax[i].set_ylabel("Counts")

        ax[i].set_title(titles[i])
        # ax[i].legend()

    plt.draw()
    plt.tight_layout()

    fig_name = f"Eros_Itokawa_{tax_type}_hist.{fig_format}"
    fig.savefig(path.join(outdir_taxonomy, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close("all")

    change_params(offset, reset=True)


def plot_ast_type_histogram(y_true: np.ndarray, y_pred: np.ndarray, type1: str, type2: str,
                            used_classes: dict[str, int] | None = None, offset: float = 0.) -> None:
    if used_classes is None: used_classes = classes

    change_params(offset)

    alpha = 0.5
    nbins = safe_arange(0., 100., 5., endpoint=True)

    xticks = safe_arange(0., 100., 10., endpoint=True)
    left, right = 0.0, 100.

    inds_1, inds_2 = y_true[:, used_classes[type1]] == 1, y_true[:, used_classes[type2]] == 1
    probability_11, probability_12 = y_pred[inds_1, used_classes[type1]] * 100., y_pred[inds_2, used_classes[type1]] * 100.
    probability_21, probability_22 = y_pred[inds_1, used_classes[type2]] * 100., y_pred[inds_2, used_classes[type2]] * 100.

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(probability_11, bins=nbins, alpha=alpha, label=type1)
    ax[0].hist(probability_12, bins=nbins, alpha=alpha, label=type2)

    ax[0].set_xticks(xticks)
    ax[0].set_xlim(left=left, right=right)

    ax[0].set_xlabel(f"Match score to {type1} (\%)")
    ax[0].set_ylabel("Counts")
    ax[0].legend()

    ax[1].hist(probability_21, bins=nbins, alpha=alpha, label=type1)
    ax[1].hist(probability_22, bins=nbins, alpha=alpha, label=type2)

    ax[1].set_xticks(xticks)
    ax[1].set_xlim(left=left, right=right)

    ax[1].set_xlabel(f"Match score to {type2} (\%)")
    # ax[1].set_ylabel("Counts")
    ax[1].legend()

    plt.draw()
    plt.tight_layout()

    fig_name = f"{type1}_{type2}_hist.{fig_format}"
    fig.savefig(path.join(outdir_taxonomy, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close("all")

    change_params(offset, reset=True)


def plot_Eros_spectra(offset: float = 0.) -> None:
    change_params(offset)

    titles = ["NIS", "DeMeo et al. (2009)", "NIS corrected"]

    ls = ["-k", "-r", "-b"]

    # NIS raw
    Eros_dir = path.join("asteroids", "Eros")
    NIS = load_npz(f"Eros{_sep_in}NIS{_sep_out}denoised{_sep_out}norm.npz", subfolder=Eros_dir)
    NIS, wavelengths = NIS[_spectra_name], NIS[_wavelengths_name]

    # DeMeo et al. (2009)
    DM = load_npz(f"asteroid{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz", subfolder="taxonomy")
    meta = join_data(DM, "meta")
    i_eros = "433" == np.array(meta["asteroid number"], dtype=str)
    DM, DM_wvl = DM[_spectra_name][i_eros], DM[_wavelengths_name]
    fun = interp1d(DM_wvl, DM, kind="cubic")
    DM = normalise_spectra(fun(wavelengths), wavelengths, wvl_norm_nm=1300., fun=fun, on_pixel=False)

    # NIS corrected
    NIS_corr = load_npz(f"Eros{_sep_out}denoised{_sep_out}norm.npz")
    NIS_corr = NIS_corr[_spectra_name]

    mask1 = wavelengths >= 1500.
    mask2 = wavelengths <= 1500.

    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    ax[0].plot(wavelengths, NIS.T, ls[0])
    ax[0].plot(wavelengths[mask2], np.mean(NIS, axis=0)[mask2].T, ls[2], linewidth=3)
    ax[0].plot(wavelengths[mask1], np.mean(NIS, axis=0)[mask1].T, ls[1], linewidth=3)


    ax[1].plot(wavelengths, DM.T, ls[0])
    ax[1].plot(wavelengths[mask2], np.mean(DM, axis=0)[mask2].T, ls[2], linewidth=3)
    ax[1].plot(wavelengths[mask1], np.mean(DM, axis=0)[mask1].T, ls[1], linewidth=3)


    ax[2].plot(wavelengths, NIS_corr.T, ls[0])
    ax[2].plot(wavelengths[mask2], np.mean(NIS_corr, axis=0)[mask2].T, ls[2], linewidth=3)
    ax[2].plot(wavelengths[mask1], np.mean(NIS_corr, axis=0)[mask1].T, ls[1], linewidth=3)

    for i, axis in enumerate(ax):
        axis.set_xlabel("Wavelength (nm)")
        axis.set_xlim([np.min(wavelengths) - 50., np.max(wavelengths) + 50.])
        axis.set_ylim([0.7, 1.7])
        axis.set_title(titles[i])

    ax[0].set_ylabel("Reflectance (normalised)")

    plt.draw()
    plt.tight_layout()

    fig_name = f"Eros_correction.{big_fig_format}"
    fig.savefig(path.join(outdir, fig_name), format=big_fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_range_histogram(start: np.ndarray, stop: np.ndarray, step: np.ndarray, offset: float = 0.) -> None:
    print("Range histograms")
    change_params(offset)

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

    fig_name = f"hist_range.{fig_format}"
    fig.savefig(path.join(outdir, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_numbers_histogram(fa: np.ndarray, fs: np.ndarray, wo: np.ndarray, offset: float = 0.) -> None:
    print("Numbers histograms")
    change_params(offset)

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

    fig_name = f"hist_numbers.{fig_format}"
    fig.savefig(path.join(outdir, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_ast_type_hist(y_pred: np.ndarray, offset: float = 0.) -> None:
    print("Plot histograms of asteroids compositions")

    change_params(offset)

    nbins = 10
    shift = 3.  # Control ylim

    predictions = y_pred * 100.

    data = load_npz(f"asteroid{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz", subfolder="taxonomy")
    types = np.ravel(data[_label_name])

    inds_S = np.array(["S" in ast_type for ast_type in types])
    inds_Q = np.array(["Q" in ast_type for ast_type in types])
    inds_A = np.array(["A" in ast_type for ast_type in types])
    inds_V = np.array(["V" in ast_type for ast_type in types])

    inds = inds_S + inds_Q + inds_V + inds_A

    # ignore Sq: since its spectrum is very weird
    inds[types == "Sq:"] = False

    color = ["blue", "magenta", "brown", "black"]
    labels = ["S type", "Q type", "A type", "V type"]

    # Modal first (expect all four minerals to be in the model)
    n_min = 3
    fig, ax = plt.subplots(1, n_min, figsize=(6 * n_min, 5), sharey=True)  # !!!!!

    labelx = ["Olivine fraction (vol\%)", "Orthopyroxene fraction (vol\%)",
              "Clinopyroxene fraction (vol\%)", "Plagioclase fraction (vol\%)"]

    limy = 0.

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
            hist = normalise_array(hist, norm_constant=100.)
            ax[j].bar(bins[:-1] + (bins[1] - bins[0]) / 2., hist,
                      width=(bins[1] - bins[0]), fill=False, edgecolor=c, label=label, linewidth=2)

            limy = np.max((limy, np.max(hist)))  # limy is the highest value in histograms

        if j == 0:
            ax[j].set_ylabel("Normalised counts (\%)")

        ax[j].set_xlabel(labelx[j])
        ax[j].legend(loc="upper left")

    # must be done after the first loop to get limy
    limy = np.min((100., np.round(limy))) + shift
    for j in range(n_min):
        # if j > 1:  # !!!!!
        #    continue
        ax[j].set_ylim(bottom=0, top=limy)

    fig.tight_layout()  # Otherwise the right y-label is slightly clipped

    plt.draw()

    fig_name = f"ast_type_hist_modal.{fig_format}"
    fig.savefig(path.join(outdir, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    # OL
    n_ol = 2
    fig, ax = plt.subplots(1, n_ol, figsize=(6 * n_ol, 5), sharey=True)

    labelx = ["Fa", "Fo"]

    limy = 0.

    for j in range(n_ol):
        for ind, c, label in zip(inds, color, labels):
            hist, bins = np.histogram(predictions[ind, n_min + j], bins=nbins, range=(0, 100))
            hist = normalise_array(hist, norm_constant=100.)
            ax[j].bar(bins[:-1] + (bins[1] - bins[0]) / 2., hist,
                      width=(bins[1] - bins[0]), fill=False, edgecolor=c, label=label, linewidth=2)

            limy = np.max((limy, np.max(hist)))  # limy is the highest value in histograms

        if j == 0:
            ax[j].set_ylabel("Normalised counts (\%)")

        ax[j].set_xlabel(labelx[j])
        ax[j].legend(loc="upper left")

    # must be done after the first loop to get limy
    limy = np.min((100., np.round(limy))) + shift
    for j in range(n_ol):
        ax[j].set_ylim(bottom=0, top=limy)

    fig.tight_layout()  # Otherwise the right y-label is slightly clipped

    plt.draw()

    fig_name = f"ast_type_hist_ol.{fig_format}"
    fig.savefig(path.join(outdir, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    # OPX
    n_opx = 2
    fig, ax = plt.subplots(1, n_opx, figsize=(6 * n_opx, 5), sharey=True)

    labelx = ["Fs (OPX)", "En (OPX)"]

    limy = 0.

    for j in range(n_opx):
        for ind, c, label in zip(inds, color, labels):
            hist, bins = np.histogram(predictions[ind, n_min + n_ol + j], bins=nbins, range=(0, 100))
            hist = normalise_array(hist, norm_constant=100.)
            ax[j].bar(bins[:-1] + (bins[1] - bins[0]) / 2., hist,
                      width=(bins[1] - bins[0]), fill=False, edgecolor=c, label=label, linewidth=2)

            limy = np.max((limy, np.max(hist)))  # limy is the highest value in histograms

        if j == 0:
            ax[j].set_ylabel("Normalised counts (\%)")

        ax[j].set_xlabel(labelx[j])
        ax[j].legend(loc="upper left")

    # must be done after the first loop to get limy
    limy = np.min((100., np.round(limy))) + shift
    for j in range(n_opx):
        ax[j].set_ylim(bottom=0, top=limy)

    fig.tight_layout()  # Otherwise the right y-label is slightly clipped

    plt.draw()

    fig_name = f"ast_type_hist_opx.{fig_format}"
    fig.savefig(path.join(outdir, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    # CPX
    n_cpx = 3
    fig, ax = plt.subplots(1, n_cpx, figsize=(6 * n_cpx, 5), sharey=True)

    labelx = ["Fs (CPX)", "En (CPX)", "Wo (CPX)"]

    limy = 0.

    for j in range(n_cpx):
        for ind, c, label in zip(inds, color, labels):
            hist, bins = np.histogram(predictions[ind, n_min + n_ol + n_opx + j], bins=nbins, range=(0, 100))
            hist = normalise_array(hist, norm_constant=100.)
            ax[j].bar(bins[:-1] + (bins[1] - bins[0]) / 2., hist,
                      width=(bins[1] - bins[0]), fill=False, edgecolor=c, label=label, linewidth=2)

            limy = np.max((limy, np.max(hist)))  # limy is the highest value in histograms

        if j == 0:
            ax[j].set_ylabel("Normalised counts (\%)")

        ax[j].set_xlabel(labelx[j])
        ax[j].legend(loc="upper left")

    # must be done after the first loop to get limy
    limy = np.min((100., np.round(limy))) + shift
    for j in range(n_cpx):
        ax[j].set_ylim(bottom=0, top=limy)

    fig.tight_layout()  # Otherwise the right y-label is slightly clipped

    plt.draw()

    fig_name = f"ast_type_hist_cpx.{fig_format}"
    fig.savefig(path.join(outdir, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_Tomas_mixtures(offset: float = 0.) -> None:
    change_params(offset)

    bottom = 0.
    left, right = 400., 2500.

    data = load_npz(f"ol{_sep_in}opx{_sep_in}mix{_sep_out}denoised{_sep_out}norm.npz",
                    subfolder=f"ol{_sep_in}opx{_sep_in}mix")
    x, spectra_msm = data[_wavelengths_name], data[_spectra_name]
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

    fig_name = f"spectra_mixtures_Tomas.{fig_format}"
    fig.savefig(path.join(outdir, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close()

    change_params(offset, reset=True)


def plot_OC_distance(y_pred: np.ndarray, meta: pd.DataFrame, used_minerals: np.ndarray | None = None,
                     used_endmembers: list[list[bool]] | None = None, offset: float = 0.) -> None:
    print("plot distance of OC predictions from the corresponding box")

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    change_params(offset)

    s = 30.  # scaling parameter (marker size)

    # Indices in which Fa and Fs are
    num_minerals = gimme_num_minerals(used_minerals)
    count_endmembers = gimme_endmember_counts(used_endmembers)

    ind_Fa, ind_Fs = num_minerals, num_minerals + count_endmembers[0]

    types = np.array(meta["SubType"], dtype=str)

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

    fig.tight_layout()  # Otherwise the right y-label is slightly clipped
    plt.draw()

    fig_name = f"OC_box_dist.{fig_format}"
    fig.savefig(path.join(outdir, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_test_range(error_type: str = "RMSE", remove_outliers: bool = False,
                      offset: float = 0.) -> tuple[np.ndarray, ...]:
    change_params(offset)

    if "outliers" in error_type.lower():
        remove_outliers = False

    def process_error_type(error_type: str) -> tuple[str, None | tuple[int]]:
        if error_type.upper() == "RMSE":
            return "RMSE", None

        elif "within" in error_type.lower():
            if len(error_type.split()) == 1:
                return "Within", (10, )
            else:
                error_type, limit = error_type.split()
                limit = int(limit),
                return "Within", limit

        elif "outliers" in error_type.lower():
            return "outliers", None

        else:
            raise ValueError("Unknown error type.")

    # fill error mat
    def fill_mat(start: np.ndarray, stop: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, ...]:
        min_start, max_stop = np.min(start), np.max(stop)

        # here I assume that shift_x == shift_y
        step = np.gcd.reduce(np.unique(stack((start - min_start, max_stop - stop))))
        if step > 0.:
            # wavelength step from minimum start to minimum stop
            wvl_step_minimum = np.mod(stop - start, step)

            if not is_constant(wvl_step_minimum):  # it must be constant, otherwise something weird is happening here
                raise ValueError('I cannot fill the matrix. Check "wvl_step_minimum". It should be a constant array.')

            wvl_step_minimum = wvl_step_minimum[0]

            if wvl_step_minimum == 0:  # if it is zero, the minimum step is just "step
                wvl_step_minimum = np.gcd.reduce(np.unique(stop - start))

            max_start, min_stop = max_stop - wvl_step_minimum, min_start + wvl_step_minimum

            len_start = (max_start - min_start) / step + 1
            len_stop = (max_stop - min_stop) / step + 1
            len_data = int(np.max((len_start, len_stop)))

            error_mat = np.empty((np.shape(value)[1], len_data, len_data)) * np.nan

            irow = np.array(np.round((stop - min_stop) / step), dtype=int)
            icol = np.array(np.round((start - min_start) / step), dtype=int)
            index = (irow, icol)

            for i in range(len(error_mat)):
                error_mat[i][index] = value[:, i]

            start_grid = np.arange(len_data) * step + min_start
            stop_grid = np.arange(len_data) * step + min_stop

        else:  # step == 0. => single-value matrix
            error_mat = np.reshape(value, (-1, 1, 1))
            start_grid, stop_grid = start, stop

        return start_grid, stop_grid, error_mat

    # load data
    def load_data_for_plot(pref: str, error_type: str) -> tuple[
        list[float], list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[str]]:
        error_type, limit = process_error_type(error_type)

        if error_type == "RMSE":
            from modules.utilities_spectra import compute_metrics
        elif error_type == "Within":
            from modules.utilities_spectra import compute_within
        elif error_type == "outliers":
            from modules.utilities_spectra import outliers_frequency
        else:
            raise ValueError("Unknown error type.")

        filenames = glob(pref)

        wvl_start = np.zeros(len(filenames), dtype=int)
        wvl_stop = np.zeros(len(filenames), dtype=int)

        for i, filename in enumerate(filenames):
            data = load_npz(filename)

            y_true, y_pred = data[_label_true_name], data[_label_pred_name]

            wavelengths = data[_wavelengths_name]
            wvl_start[i], wvl_stop[i] = np.min(wavelengths), np.max(wavelengths)

            try:
                used_minerals = data[_config_name][()]["output_setup"]["used_minerals"]
                used_endmembers = data[_config_name][()]["output_setup"]["used_endmembers"]
            except KeyError:  # back compatibility
                try:
                    used_minerals = data["model_info"][()]["used_minerals"]
                    used_endmembers = data["model_info"][()]["used_endmembers"]
                except KeyError:  # back compatibility
                    used_minerals = data["model_info"][()]["used_minerals"]
                    used_endmembers = data["model_info"][()]["used_end-members"]
                used_minerals = used_minerals if np.sum(used_minerals) > 1 else np.array([False] * len(used_minerals))

            if i == 0:  # initialise error
                error = np.zeros((len(filenames), np.sum(unique_indices(used_minerals, used_endmembers)) + 1))

            if remove_outliers:
                inds_outliers = find_outliers(y_true, y_pred, used_minerals=used_minerals,
                                              used_endmembers=used_endmembers, px_only=True)
                y_true = np.delete(y_true, inds_outliers, axis=0)
                y_pred = np.delete(y_pred, inds_outliers, axis=0)

            if error_type == "RMSE":
                calc = lambda x: compute_metrics(y_true, y_pred, used_minerals=used_minerals,
                                                 used_endmembers=used_endmembers,
                                                 cleaning=True, all_to_one=x,
                                                 remove_px_outliers=remove_outliers)[0]
                error[i, 0], error[i, 1:] = calc(True), calc(False)[unique_indices(used_minerals, used_endmembers)]

            elif error_type == "Within":
                calc = lambda x: compute_within(y_true, y_pred, error_limit=limit, used_minerals=used_minerals,
                                                used_endmembers=used_endmembers, all_to_one=x,
                                                remove_px_outliers=remove_outliers)[0]
                error[i, 0], error[i, 1:] = calc(True), calc(False)[unique_indices(used_minerals, used_endmembers)]

            elif error_type == "outliers":
                outliers = outliers_frequency(y_true, y_pred, used_minerals=used_minerals,
                                              used_endmembers=used_endmembers)[unique_indices(used_minerals,
                                                                                              used_endmembers)]
                error[i, 0], error[i, 1:] = np.sum(outliers), outliers

        inds = np.lexsort((wvl_stop, wvl_start))
        wvl_start, wvl_stop, error = wvl_start[inds], wvl_stop[inds], error[inds]

        unique_start, unique_stop = np.unique(wvl_start), np.unique(wvl_stop)
        error_line = [np.transpose(error[wvl_start == start]) for start in unique_start]
        line_stop = [wvl_stop[wvl_start == start] for start in unique_start]

        wvl_start, wvl_stop, error_mat = fill_mat(wvl_start, wvl_stop, error)

        label = [f"From {start} nm" for start in unique_start]

        return line_stop, error_line, wvl_start, wvl_stop, error_mat, label

    def box_properties(image_index: int) -> list[dict]:
        if image_index == 1:  # olivine abundance
            boxes = [[{"x0": (450, 750), "y0": (1050, 1250),  # blue wing of 1-μm band
                       "edgecolor": "b", "shift": 0.1}, "1-\\textmu{}m blue shoulder"],
                     [{"x0": (450, 750), "y0": (1350, 1750),  # whole 1-μm band
                       "edgecolor": "tab:brown", "shift": 0.1}, "1-\\textmu{}m band"],
                     [{"x0": (450, 750), "y0": (2250, 2450),  # both bands
                       "edgecolor": "m", "shift": 0.1}, "1-\\textmu{}m and 2-\\textmu{}m bands"],
                     ]

        elif image_index == 2:  # orthopyroxene abundance
            boxes = [[{"x0": (450, 750), "y0": (1050, 1250),  # blue wing of 1-μm band
                       "edgecolor": "b", "shift": 0.1}, "1-\\textmu{}m blue shoulder"],
                     [{"x0": (450, 750), "y0": (1350, 1750),  # whole 1-μm band
                       "edgecolor": "tab:brown", "shift": 0.1}, "1-\\textmu{}m band"],
                     [{"x0": (850, 950), "y0": (1350, 1750),  # red wing of 1-μm band
                       "edgecolor": "r", "shift": 0.1}, "1-\\textmu{}m red shoulder"],
                     [{"x0": (450, 750), "y0": (2250, 2450),  # both bands
                       "edgecolor": "m", "shift": 0.1}, "1-\\textmu{}m and 2-\\textmu{}m bands"],
                     ]

        elif image_index == 4:  # Fa
            boxes = [[{"x0": (450, 750), "y0": (1550, 1750),  # whole 1-μm band
                       "edgecolor": "tab:brown", "shift": 0.1}, "1-\\textmu{}m band"],
                     [{"x0": (950, 950), "y0": (1250, 1250),  # BIC
                       "edgecolor": "y", "shift": 0.1}, "BIC"],
                     [{"x0": (850, 1150), "y0": (1550, 1750),  # red wing of 1-μm band
                       "edgecolor": "r", "shift": 0.1}, "1-\\textmu{}m red shoulder"],
                     ]

        elif image_index == 5:  # Fs (OPX)
            boxes = [[{"x0": (450, 750), "y0": (1350, 1750),  # whole 1-μm band
                       "edgecolor": "tab:brown", "shift": 0.1}, "1-\\textmu{}m band"],
                     [{"x0": (950, 950), "y0": (1250, 1250),  # BIC
                       "edgecolor": "y", "shift": 0.1}, "BIC"],
                     [{"x0": (450, 750), "y0": (2250, 2450),  # both bands
                       "edgecolor": "m", "shift": 0.1}, "1-\\textmu{}m and 2-\\textmu{}m bands"],
                     [{"x0": (1950, 1950), "y0": (2350, 2350),  # BIIC
                       "edgecolor": "k", "shift": 0.1}, "BIIC"]
                     ]
        else:
            boxes = []

        return boxes

    def add_box(ax, x0: tuple[int, int], y0: tuple[int, int], shift: float = 0., **kwargs) -> tuple[float, float]:
        x = np.where(wvl_start >= x0[0])[0][0] - 0.5
        width = np.where(wvl_start >= x0[1])[0][0] - x + 0.5

        y = np.where(wvl_stop >= y0[0])[0][0] - 0.5
        height = np.where(wvl_stop >= y0[1])[0][0] - y + 0.5

        patch = patches.Rectangle((x + shift, y + shift), width=width - 2 * shift, height=height - 2 * shift,
                                  facecolor="none", **kwargs)
        # Add the patch to the Axes
        ax.add_patch(patch)

        return x + width / 2, y + height / 2  # returns centre of the patch

    pref = path.join(_path_accuracy_tests, "range_test/range/*npz")

    line_stop, error_line, wvl_start, wvl_stop, error_mat, label = load_data_for_plot(pref, error_type)

    error_type, limit = process_error_type(error_type)

    yticks02 = safe_arange(0., 100., 0.2, endpoint=True)
    yticks05 = safe_arange(0., 100., 0.5, endpoint=True)
    yticks1 = safe_arange(0., 100., 1., endpoint=True)
    yticks2 = safe_arange(0., 100., 2., endpoint=True)
    yticks5 = safe_arange(0., 100., 5., endpoint=True)
    yticks10 = safe_arange(0., 500., 10., endpoint=True)

    titles_all = np.array(["all", "olivine", "orthopyroxene", "clinopyroxene", "Fa",
                           "Fs OPX", "Fs CPX", "En CPX", "Wo CPX"], dtype=str)

    num_colors = np.shape(error_mat)[-1]

    outdir_range_tests = path.join(outdir, "range_test")
    check_dir(outdir_range_tests)

    cm = plt.get_cmap("gist_rainbow")
    arrowprops = {"color": "red",
                  "shrink": 0.}
    prec_shift = 2.5

    if error_type == "RMSE":
        unit1, unit2 = "pp", "pp"
    elif error_type == "Within":
        unit1, unit2 = "\%", "pp"
    else:
        unit1, unit2 = "", ""

    for q in range(len(titles_all)):
        fig, axes = plt.subplots(1, 2, figsize=(25, 10), gridspec_kw={"width_ratios": [2.3, 1]})
        axes = np.ravel(axes)

        if error_type == "outliers":
            error_label = f"No. outliers ({titles_all[q]})"
        else:
            if limit is None:
                error_label = f"{error_type} (pp; {titles_all[q]})"
            else:
                error_label = f"{error_type} {limit[0]} pp (\%; {titles_all[q]})"

        for iax, ax in enumerate(axes):
            if iax == 0:  # line plots
                ax.set_prop_cycle(color=cm(np.linspace(0., 1., num_colors)))

                for i in range(len(error_line)):
                    ax.plot(line_stop[i], error_line[i][q, :], marker='o', linestyle="--", label=label[i])

                # ax.axvline(850., color="k", ls="--", zorder=100)
                # ax.axvline(1350., color="k", ls="--", zorder=100)

                # ax.axvline(1650., color="k", ls="-.", zorder=100)
                # ax.axvline(2450., color="k", ls="-.", zorder=100)

                ax.set_xticks(wvl_stop)
                ax.set_xlabel("To wavelength (nm)")
                ax.set_ylabel(error_label)

                lim = ax.get_ylim()
                if lim[1] - lim[0] > 64.:
                    yticks = yticks10
                elif lim[1] - lim[0] > 32.:
                    yticks = yticks5
                elif lim[1] - lim[0] > 16.:
                    yticks = yticks2
                elif lim[1] - lim[0] > 8.:
                    yticks = yticks1
                elif lim[1] - lim[0] > 4.:
                    yticks = yticks05
                else:
                    yticks = yticks02
                yticks = yticks[np.logical_and(yticks >= lim[0], yticks <= lim[1])]
                ax.set_yticks(yticks)
                # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                ax.set_ylim(lim)

                ax.legend(bbox_to_anchor=(0., 1.02, 1., 0.2), loc="lower left", mode="expand", borderaxespad=0.,
                          ncol=best_blk(num_colors)[1])

            else:  # mat plot
                cmap = "viridis"
                if error_type == "RMSE":
                    vmin = np.nanmin(error_mat[q])
                    vmax = np.nanmin((np.nanmax(error_mat[q]), np.nanmin(error_mat[q]) + 10.))
                    cmap += "_r"
                    
                elif error_type == "Within":
                    vmin = np.nanmax((np.nanmin(error_mat[q]), np.nanmax(error_mat[q]) - 20.))
                    vmax = np.nanmax(error_mat[q])
                    
                elif error_type == "outliers":
                    vmin = np.nanmin(error_mat[q])
                    vmax = np.nanmin((np.nanmax(error_mat[q]), np.nanmin(error_mat[q]) + 10.))

                if vmin == vmax:
                    im = ax.imshow(error_mat[q], aspect="auto", cmap=cmap)
                else:
                    im = ax.imshow(error_mat[q], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

                rows, cols = np.shape(error_mat[q])
                ax.set_xticks(range(cols))
                ax.set_yticks(range(rows))

                ax.set_xticklabels(wvl_start, rotation=90, ha="center")
                ax.set_yticklabels(wvl_stop)

                # add_box(ax, (450, 750), (1350, 2450), linewidth=3, linestyle="-", edgecolor="k")
                # add_box(ax, (450, 1550), (2450, 2450), linewidth=3, linestyle="-", edgecolor="k")

                boxes = box_properties(q)

                if boxes:
                    for i, (box_prop, description) in enumerate(boxes):
                        _mn, _std = print_grid_test_stats_range(error_mat[q],
                                                                lim_from=box_prop["x0"], lim_to=box_prop["y0"],
                                                                quiet=True)
                        mn, std = np.round(_mn, 1), np.round(_std, 1)

                        text = f"{mn:.1f} {unit1} $\pm$ {std:.1f} {unit2}" if _std > 0 else f"{mn:.1f} {unit1}"

                        xc, yc = add_box(ax, **box_prop, linewidth=2, linestyle="--")
                        ax.annotate(f"{description} \n {text}", xy=(xc, yc),
                                    xytext=(4 + prec_shift * i, 2 + prec_shift * i),
                                    fontsize=SMALL_SIZE - 3 + offset, color=box_prop["edgecolor"],
                                    arrowprops=arrowprops | {"color": box_prop["edgecolor"]})

                ax.set_xlabel("From wavelength (nm)")
                ax.set_ylabel("To wavelength (nm)")
                ax.xaxis.set_label_position("bottom")
                ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                               bottom=True, top=False, left=True, right=False)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes(**cbar_kwargs)
                cbar = plt.colorbar(im, cax=cax)
                cbar.ax.set_ylabel(error_label)

                lim = (cbar.vmin, cbar.vmax)
                if lim[1] - lim[0] > 32.:
                    yticks = yticks5
                elif lim[1] - lim[0] > 16.:
                    yticks = yticks2
                elif lim[1] - lim[0] > 8.:
                    yticks = yticks1
                elif lim[1] - lim[0] > 4.:
                    yticks = yticks05
                else:
                    yticks = yticks02
                yticks = yticks[np.logical_and(yticks >= lim[0], yticks <= lim[1])]
                if not is_empty(yticks):
                    cbar.set_ticks(yticks)

        if limit is None:
            fig_name = f"range_{error_type}{_sep_out}{titles_all[q].replace(' ', _sep_in)}.{fig_format}"
        else:
            fig_name = f"range_{error_type}{_sep_in}{limit[0]}{_sep_out}{titles_all[q].replace(' ', _sep_in)}.{fig_format}"

        plt.draw()
        plt.tight_layout()

        fig.savefig(path.join(outdir_range_tests, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
        plt.close(fig)

    #### plot parts separately
    for q in [1, 2, 4, 5]:
        fig, ax = plt.subplots(1, 1, figsize=(18, 10))

        if error_type == "outliers":
            error_label = f"No. outliers ({titles_all[q]})"
        else:
            if limit is None:
                error_label = f"{error_type} (pp; {titles_all[q]})"
            else:
                error_label = f"{error_type} {limit[0]} pp (\%; {titles_all[q]})"

        ax.set_prop_cycle(color=cm(np.linspace(0., 1., num_colors)))

        for i in range(len(error_line)):
            ax.plot(line_stop[i], error_line[i][q, :], marker='o', linestyle="--", label=label[i])

        # ax.axvline(850., color="k", ls="--", zorder=100)
        # ax.axvline(1350., color="k", ls="--", zorder=100)

        # ax.axvline(1650., color="k", ls="-.", zorder=100)
        # ax.axvline(2450., color="k", ls="-.", zorder=100)

        ax.set_xticks(wvl_stop)
        ax.set_xlabel("To wavelength (nm)")
        ax.set_ylabel(error_label)

        lim = ax.get_ylim()
        if lim[1] - lim[0] > 64.:
            yticks = yticks10
        elif lim[1] - lim[0] > 32.:
            yticks = yticks5
        elif lim[1] - lim[0] > 16.:
            yticks = yticks2
        elif lim[1] - lim[0] > 8.:
            yticks = yticks1
        elif lim[1] - lim[0] > 4.:
            yticks = yticks05
        else:
            yticks = yticks02
        yticks = yticks[np.logical_and(yticks >= lim[0], yticks <= lim[1])]
        ax.set_yticks(yticks)
        # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylim(lim)

        ax.legend(bbox_to_anchor=(0., 1.02, 1., 0.2), loc="lower left", mode="expand", borderaxespad=0.,
                  ncol=best_blk(num_colors)[1])

        if limit is None:
            fig_name = f"range_{error_type}{_sep_out}{titles_all[q].replace(' ', _sep_in)}_line.{fig_format}"
        else:
            fig_name = f"range_{error_type}{_sep_in}{limit[0]}{_sep_out}{titles_all[q].replace(' ', _sep_in)}_line.{fig_format}"

        plt.draw()
        plt.tight_layout()

        fig.savefig(path.join(outdir_range_tests, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
        plt.close(fig)

        # mat plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        cmap = "viridis"
        if error_type == "RMSE":
            vmin = np.nanmin(error_mat[q])
            vmax = np.nanmin((np.nanmax(error_mat[q]), np.nanmin(error_mat[q]) + 10.))
            cmap += "_r"

        elif error_type == "Within":
            vmin = np.nanmax((np.nanmin(error_mat[q]), np.nanmax(error_mat[q]) - 20.))
            vmax = np.nanmax(error_mat[q])

        elif error_type == "outliers":
            vmin = np.nanmin(error_mat[q])
            vmax = np.nanmin((np.nanmax(error_mat[q]), np.nanmin(error_mat[q]) + 10.))

        if vmin == vmax:
            im = ax.imshow(error_mat[q], aspect="auto", cmap=cmap)
        else:
            im = ax.imshow(error_mat[q], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        rows, cols = np.shape(error_mat[q])
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))

        ax.set_xticklabels(wvl_start, rotation=90, ha="center")
        ax.set_yticklabels(wvl_stop)

        # add_box(ax, (450, 750), (1350, 2450), linewidth=3, linestyle="-", edgecolor="k")
        # add_box(ax, (450, 1550), (2450, 2450), linewidth=3, linestyle="-", edgecolor="k")

        boxes = box_properties(q)

        if boxes:
            for i, (box_prop, description) in enumerate(boxes):
                _mn, _std = print_grid_test_stats_range(error_mat[q],
                                                        lim_from=box_prop["x0"], lim_to=box_prop["y0"],
                                                        quiet=True)
                mn, std = np.round(_mn, 1), np.round(_std, 1)

                text = f"{mn:.1f} {unit1} $\pm$ {std:.1f} {unit2}" if _std > 0 else f"{mn:.1f} {unit1}"

                xc, yc = add_box(ax, **box_prop, linewidth=2, linestyle="--")
                ax.annotate(f"{description} \n {text}", xy=(xc, yc),
                            xytext=(4 + prec_shift * i, 2 + prec_shift * i),
                            fontsize=MEDIUM_SIZE + offset, color=box_prop["edgecolor"],
                            arrowprops=arrowprops | {"color": box_prop["edgecolor"]})

        ax.set_xlabel("From wavelength (nm)")
        ax.set_ylabel("To wavelength (nm)")
        ax.xaxis.set_label_position("bottom")
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=False, left=True, right=False)
        ax.axis("equal")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes(**cbar_kwargs)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(error_label)

        lim = (cbar.vmin, cbar.vmax)
        if lim[1] - lim[0] > 32.:
            yticks = yticks5
        elif lim[1] - lim[0] > 16.:
            yticks = yticks2
        elif lim[1] - lim[0] > 8.:
            yticks = yticks1
        elif lim[1] - lim[0] > 4.:
            yticks = yticks05
        else:
            yticks = yticks02
        yticks = yticks[np.logical_and(yticks >= lim[0], yticks <= lim[1])]
        if not is_empty(yticks):
            cbar.set_ticks(yticks)

        if limit is None:
            fig_name = f"range_{error_type}{_sep_out}{titles_all[q].replace(' ', _sep_in)}_mat.{fig_format}"
        else:
            fig_name = f"range_{error_type}{_sep_in}{limit[0]}{_sep_out}{titles_all[q].replace(' ', _sep_in)}_mat.{fig_format}"

        plt.draw()
        plt.tight_layout()

        fig.savefig(path.join(outdir_range_tests, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
        plt.close(fig)

    change_params(offset, reset=True)

    return wvl_start, wvl_stop, error_mat


def plot_test_step(error_type: str = "RMSE", remove_outliers: bool = False,
                    offset: float = 0.) -> tuple[np.ndarray, ...]:
    change_params(offset)

    if "outliers" in error_type.lower():
        remove_outliers = False

    def process_error_type(error_type: str) -> tuple[str, None | tuple[int]]:
        if error_type.upper() == "RMSE":
            return "RMSE", None

        elif "within" in error_type.lower():
            if len(error_type.split()) == 1:
                return "Within", (10, )
            else:
                error_type, limit = error_type.split()
                limit = int(limit),
                return "Within", limit

        elif "outliers" in error_type.lower():
            return "outliers", None

        else:
            raise ValueError("Unknown error type.")

    # load data
    def load_data_for_plot(pref: str, error_type: str) -> tuple[np.ndarray, np.ndarray, str]:
        error_type, limit = process_error_type(error_type)

        if error_type == "RMSE":
            from modules.utilities_spectra import compute_metrics
        elif error_type == "Within":
            from modules.utilities_spectra import compute_within
        elif error_type == "outliers":
            from modules.utilities_spectra import outliers_frequency
        else:
            raise ValueError("Unknown error type.")

        filenames = glob(pref)

        wvl_step = np.zeros(len(filenames))

        for i, filename in enumerate(filenames):
            data = load_npz(filename)

            y_true, y_pred = data[_label_true_name], data[_label_pred_name]

            wavelengths = data[_wavelengths_name]
            wvl_step[i] = np.mean(np.diff(wavelengths))

            try:
                used_minerals = data[_config_name][()]["output_setup"]["used_minerals"]
                used_endmembers = data[_config_name][()]["output_setup"]["used_endmembers"]
            except KeyError:  # back compatibility
                try:
                    used_minerals = data["model_info"][()]["used_minerals"]
                    used_endmembers = data["model_info"][()]["used_endmembers"]
                except KeyError:  # back compatibility
                    used_minerals = data["model_info"][()]["used_minerals"]
                    used_endmembers = data["model_info"][()]["used_end-members"]
                used_minerals = used_minerals if np.sum(used_minerals) > 1 else np.array([False] * len(used_minerals))

            if remove_outliers:
                inds_outliers = find_outliers(y_true, y_pred, used_minerals=used_minerals,
                                              used_endmembers=used_endmembers, px_only=True)
                y_true = np.delete(y_true, inds_outliers, axis=0)
                y_pred = np.delete(y_pred, inds_outliers, axis=0)

            if i == 0:
                error = np.zeros((len(filenames), np.sum(unique_indices(used_minerals, used_endmembers)) + 1))

            if error_type == "RMSE":
                calc = lambda x: compute_metrics(y_true, y_pred, used_minerals=used_minerals,
                                                 used_endmembers=used_endmembers,
                                                 cleaning=True, all_to_one=x,
                                                 remove_px_outliers=remove_outliers)[0]
                error[i, 0], error[i, 1:] = calc(True), calc(False)[unique_indices(used_minerals, used_endmembers)]

            elif error_type == "Within":
                calc = lambda x: compute_within(y_true, y_pred, error_limit=limit, used_minerals=used_minerals,
                                                used_endmembers=used_endmembers, all_to_one=x,
                                                remove_px_outliers=remove_outliers)[0]
                error[i, 0], error[i, 1:] = calc(True), calc(False)[unique_indices(used_minerals, used_endmembers)]

            elif error_type == "outliers":
                outliers = outliers_frequency(y_true, y_pred, used_minerals=used_minerals,
                                              used_endmembers=used_endmembers)[unique_indices(used_minerals,
                                                                                              used_endmembers)]
                error[i, 0], error[i, 1:] = np.sum(outliers), outliers

        inds = np.argsort(wvl_step)
        wvl_step, error = wvl_step[inds], np.transpose(error[inds])

        if "ASPECT" in filename:  # this is here to get nice limits in labels
            if "swir" in filename:
                label = f"ASPECT {650}\u2013{2450} nm"
            else:
                label = f"ASPECT {650}\u2013{1600} nm"
        elif "HS-H" in filename:
            label = f"HS-H {650}\u2013{950} nm"
        else:
            label = f"{int(np.min(wavelengths))}\u2013{int(np.max(wavelengths))} nm"

        return wvl_step, error, label

    prefs = [
        path.join(_path_accuracy_tests, "range_test/step/composition_650-1850*.npz"),
        path.join(_path_accuracy_tests, "range_test/step/composition_650-2450*.npz"),
        path.join(_path_accuracy_tests, "range_test/step/composition_ASPECT_vis-nir1-nir2_*.npz"),
        path.join(_path_accuracy_tests, "range_test/step/composition_ASPECT_vis-nir1-nir2-swir_*.npz"),
        # path.join(_path_accuracy_tests, "range_test/range/composition_HS-H*.npz")
             ]

    num_colors = len(prefs)

    wvl_step, error, label = zip(*[load_data_for_plot(pref, error_type) for pref in prefs])

    error_type, limit = process_error_type(error_type)

    yticks02 = safe_arange(0., 100., 0.2, endpoint=True)
    yticks05 = safe_arange(0., 100., 0.5, endpoint=True)
    yticks1 = safe_arange(0., 100., 1., endpoint=True)
    yticks2 = safe_arange(0., 100., 2., endpoint=True)
    yticks5 = safe_arange(0., 100., 5., endpoint=True)

    titles_all = np.array(["All", "Olivine", "Orthopyroxene", "Clinopyroxene", "Fa",
                           "Fs (OPX)", "Fs (CPX)", "En (CPX)", "Wo (CPX)"], dtype=str)

    outdir_range_tests = path.join(outdir, "range_test")
    check_dir(outdir_range_tests)

    cm = plt.get_cmap("gist_rainbow")

    fig, axes = plt.subplots(*best_blk(len(error[0])), figsize=(24, 12), sharex=True)
    axes = np.ravel(axes)

    colors = cm(np.linspace(0., 1., num_colors))

    for q, ax in enumerate(axes):
        for i in range(len(error)):
            if q == 0:
                ax.plot(wvl_step[i], error[i][q], marker='o', linestyle="--", label=label[i], color=colors[i])
            else:
                ax.plot(wvl_step[i], error[i][q], marker='o', linestyle="--", color=colors[i])

            if i in [2, 3]:  # highlight ASPECT targeted step
                ax.plot(wvl_step[i][2], error[i][q][2], marker='o', color=colors[i], markersize=13)

        ax.set_title(titles_all[q])

        if q > 5:
            ax.set_xlabel("Step (nm)")

        if error_type == "outliers":
            ax.set_ylabel("No. outliers")
        else:
            if limit is None:
                ax.set_ylabel(f"{error_type} (pp)")
            else:
                ax.set_ylabel(f"{error_type} {limit[0]} pp (\%)")

        lim = ax.get_ylim()
        if lim[1] - lim[0] > 16.:
            yticks = yticks5
        elif lim[1] - lim[0] > 8.:
            yticks = yticks2
        elif lim[1] - lim[0] > 4.:
            yticks = yticks1
        elif lim[1] - lim[0] > 2.:
            yticks = yticks05
        else:
            yticks = yticks02
        yticks = yticks[np.logical_and(yticks >= lim[0], yticks <= lim[1])]

        ax.set_yticks(yticks)
        # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.set_ylim(lim)

    if limit is None:
        fig_name = f"step_{error_type}.{fig_format}"
    else:
        fig_name = f"step_{error_type}{_sep_in}{limit[0]}.{fig_format}"

    plt.draw()
    plt.tight_layout()

    fig.legend(bbox_to_anchor=(axes[0].get_position().intervalx[0], 1.02,
                               axes[best_blk(len(error[0]))[1] - 1].get_position().intervalx[1] -
                               axes[0].get_position().intervalx[0], 0.2),
               loc="lower left", mode="expand", borderaxespad=0.,
               ncol=best_blk(num_colors, 4.)[1])

    fig.savefig(path.join(outdir_range_tests, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)

    return wvl_step, error


def plot_test_normalisation(error_type: str = "RMSE", remove_outliers: bool = False,
                            offset: float = 0.) -> tuple[np.ndarray, ...]:
    change_params(offset)

    if "outliers" in error_type.lower():
        remove_outliers = False

    def process_error_type(error_type: str) -> tuple[str, None | tuple[int]]:
        if error_type.upper() == "RMSE":
            return "RMSE", None

        elif "within" in error_type.lower():
            if len(error_type.split()) == 1:
                return "Within", (10, )
            else:
                error_type, limit = error_type.split()
                limit = int(limit),
                return "Within", limit

        elif "outliers" in error_type.lower():
            return "outliers", None

        else:
            raise ValueError("Unknown error type.")

    # load data
    def load_data_for_plot(pref: str, error_type: str) -> tuple[np.ndarray, np.ndarray]:
        error_type, limit = process_error_type(error_type)

        if error_type == "RMSE":
            from modules.utilities_spectra import compute_metrics
        elif error_type == "Within":
            from modules.utilities_spectra import compute_within
        elif error_type == "outliers":
            from modules.utilities_spectra import outliers_frequency
        else:
            raise ValueError("Unknown error type.")

        filenames = glob(pref)

        wvl_norm = np.zeros(len(filenames))

        for i, filename in enumerate(filenames):
            data = load_npz(filename)

            y_true, y_pred = data[_label_true_name], data[_label_pred_name]

            try:
                wvl_norm[i] = data[_config_name][()]["grid_setup"]["wvl_norm"]
                used_minerals = data[_config_name][()]["output_setup"]["used_minerals"]
                used_endmembers = data[_config_name][()]["output_setup"]["used_endmembers"]
            except KeyError:  # back compatibility
                try:
                    wvl_norm[i] = data["model_info"][()]["wvl_norm"]
                    used_minerals = data["model_info"][()]["used_minerals"]
                    used_endmembers = data["model_info"][()]["used_endmembers"]
                except KeyError:  # back compatibility
                    wvl_norm[i] = data["model_info"][()]["wvl_norm"]
                    used_minerals = data["model_info"][()]["used_minerals"]
                    used_endmembers = data["model_info"][()]["used_end-members"]
                used_minerals = used_minerals if np.sum(used_minerals) > 1 else np.array([False] * len(used_minerals))

            if remove_outliers:
                inds_outliers = find_outliers(y_true, y_pred, used_minerals=used_minerals,
                                              used_endmembers=used_endmembers, px_only=True)
                y_true = np.delete(y_true, inds_outliers, axis=0)
                y_pred = np.delete(y_pred, inds_outliers, axis=0)

            if i == 0:
                error = np.zeros((len(filenames), np.sum(unique_indices(used_minerals, used_endmembers)) + 1))

            if error_type == "RMSE":
                calc = lambda x: compute_metrics(y_true, y_pred, used_minerals=used_minerals,
                                                 used_endmembers=used_endmembers,
                                                 cleaning=True, all_to_one=x,
                                                 remove_px_outliers=remove_outliers)[0]
                error[i, 0], error[i, 1:] = calc(True), calc(False)[unique_indices(used_minerals, used_endmembers)]

            elif error_type == "Within":
                calc = lambda x: compute_within(y_true, y_pred, error_limit=limit, used_minerals=used_minerals,
                                                used_endmembers=used_endmembers, all_to_one=x,
                                                remove_px_outliers=remove_outliers)[0]
                error[i, 0], error[i, 1:] = calc(True), calc(False)[unique_indices(used_minerals, used_endmembers)]

            elif error_type == "outliers":
                outliers = outliers_frequency(y_true, y_pred, used_minerals=used_minerals,
                                              used_endmembers=used_endmembers)[unique_indices(used_minerals,
                                                                                              used_endmembers)]
                error[i, 0], error[i, 1:] = np.sum(outliers), outliers

        inds = np.argsort(wvl_norm)
        wvl_norm, error = wvl_norm[inds], np.transpose(error[inds])

        return wvl_norm, error

    pref = path.join(_path_accuracy_tests, "range_test/normalisation/composition_450-2450*.npz")

    wvl_norm, error = load_data_for_plot(pref, error_type)

    error_type, limit = process_error_type(error_type)

    xticks = safe_arange(np.min(wvl_norm), np.max(wvl_norm), 2 * np.mean(np.diff(wvl_norm)),
                         endpoint=True, linspace_like=False)
    yticks02 = safe_arange(0., 100., 0.2, endpoint=True)
    yticks05 = safe_arange(0., 100., 0.5, endpoint=True)
    yticks1 = safe_arange(0., 100., 1., endpoint=True)
    yticks2 = safe_arange(0., 100., 2., endpoint=True)
    yticks5 = safe_arange(0., 100., 5., endpoint=True)

    titles_all = np.array(["All", "Olivine", "Orthopyroxene", "Clinopyroxene", "Fa",
                           "Fs (OPX)", "Fs (CPX)", "En (CPX)", "Wo (CPX)"], dtype=str)

    num_colors = len(titles_all)

    cm = plt.get_cmap("gist_rainbow")

    outdir_range_tests = path.join(outdir, "range_test")
    check_dir(outdir_range_tests)

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.set_prop_cycle(color=cm(np.linspace(0., 1., num_colors)))

    for i, line in enumerate(error):
        ax.plot(wvl_norm, line, marker='o', linestyle="--", label=titles_all[i])

    ax.set_xticks(xticks)

    lim = ax.get_ylim()
    if lim[1] - lim[0] > 16.:
        yticks = yticks5
    elif lim[1] - lim[0] > 8.:
        yticks = yticks2
    elif lim[1] - lim[0] > 4.:
        yticks = yticks1
    elif lim[1] - lim[0] > 2.:
        yticks = yticks05
    else:
        yticks = yticks02
    yticks = yticks[np.logical_and(yticks >= lim[0], yticks <= lim[1])]

    ax.set_yticks(yticks)
    # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylim(lim)

    ax.set_xlabel("Normalised at (nm)")
    ax.legend(bbox_to_anchor=(0., 1.02, 1., 0.2), loc="lower left", mode="expand", borderaxespad=0.,
              ncol=best_blk(num_colors)[1])

    if error_type == "outliers":
        ax.set_ylabel("No. outliers")
        fig_name = f"normalisation_{error_type}.{fig_format}"
    else:
        if limit is None:
            ax.set_ylabel(f"{error_type} (pp)")
            fig_name = f"normalisation_{error_type}.{fig_format}"
        else:
            ax.set_ylabel(f"{error_type} {limit[0]} pp (\%)")
            fig_name = f"normalisation_{error_type}{_sep_in}{limit[0]}.{fig_format}"

    plt.draw()
    plt.tight_layout()

    fig.savefig(path.join(outdir_range_tests, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)

    return wvl_norm, error


def plot_test_window(error_type: str = "RMSE", remove_outliers: bool = False,
                     offset: float = 0.) -> tuple[np.ndarray, ...]:
    change_params(offset)

    if "outliers" in error_type.lower():
        remove_outliers = False

    def process_error_type(error_type: str) -> tuple[str, None | tuple[int]]:
        if error_type.upper() == "RMSE":
            return "RMSE", None

        elif "within" in error_type.lower():
            if len(error_type.split()) == 1:
                return "Within", (10, )
            else:
                error_type, limit = error_type.split()
                limit = int(limit),
                return "Within", limit

        elif "outliers" in error_type.lower():
            return "outliers", None

        else:
            raise ValueError("Unknown error type.")

    # load data
    def load_data_for_plot(pref: str, error_type: str):
        error_type, limit = process_error_type(error_type)

        if error_type == "RMSE":
            from modules.utilities_spectra import compute_metrics
        elif error_type == "Within":
            from modules.utilities_spectra import compute_within
        elif error_type == "outliers":
            from modules.utilities_spectra import outliers_frequency
        else:
            raise ValueError("Unknown error type.")

        filenames = glob(pref)

        window_range = np.zeros(len(filenames), dtype="<U100")
        window_start = np.zeros(len(filenames), dtype=int)

        for i, filename in enumerate(filenames):
            data = load_npz(filename)

            y_true, y_pred = data[_label_true_name], data[_label_pred_name]

            pos_parentheses = [find_all(filename, "("), find_all(filename, ")")]
            window_start[i] = int(filename[pos_parentheses[0][0] + 1:pos_parentheses[1][0]])
            window_end = int(filename[pos_parentheses[0][1] + 1:pos_parentheses[1][1]])
            window_range[i] = f"{window_start[i]}\u2013{window_end}"

            try:
                used_minerals = data[_config_name][()]["output_setup"]["used_minerals"]
                used_endmembers = data[_config_name][()]["output_setup"]["used_endmembers"]
            except KeyError:  # back compatibility
                try:
                    used_minerals = data["model_info"][()]["used_minerals"]
                    used_endmembers = data["model_info"][()]["used_endmembers"]
                except KeyError:  # back compatibility
                    used_minerals = data["model_info"][()]["used_minerals"]
                    used_endmembers = data["model_info"][()]["used_end-members"]
                used_minerals = used_minerals if np.sum(used_minerals) > 1 else np.array([False] * len(used_minerals))

            if remove_outliers:
                inds_outliers = find_outliers(y_true, y_pred, used_minerals=used_minerals,
                                              used_endmembers=used_endmembers, px_only=True)
                y_true = np.delete(y_true, inds_outliers, axis=0)
                y_pred = np.delete(y_pred, inds_outliers, axis=0)

            if i == 0:
                error = np.zeros((len(filenames), np.sum(unique_indices(used_minerals, used_endmembers)) + 1))

            if error_type == "RMSE":
                calc = lambda x: compute_metrics(y_true, y_pred, used_minerals=used_minerals,
                                                 used_endmembers=used_endmembers,
                                                 cleaning=True, all_to_one=x,
                                                 remove_px_outliers=remove_outliers)[0]
                error[i, 0], error[i, 1:] = calc(True), calc(False)[unique_indices(used_minerals, used_endmembers)]

            elif error_type == "Within":
                calc = lambda x: compute_within(y_true, y_pred, error_limit=limit, used_minerals=used_minerals,
                                                used_endmembers=used_endmembers, all_to_one=x,
                                                remove_px_outliers=remove_outliers)[0]
                error[i, 0], error[i, 1:] = calc(True), calc(False)[unique_indices(used_minerals, used_endmembers)]

            elif error_type == "outliers":
                outliers = outliers_frequency(y_true, y_pred, used_minerals=used_minerals,
                                              used_endmembers=used_endmembers)[unique_indices(used_minerals,
                                                                                              used_endmembers)]
                error[i, 0], error[i, 1:] = np.sum(outliers), outliers

        inds = np.argsort(window_start)
        window_range, error = window_range[inds], np.transpose(error[inds])

        return window_range, error

    pref = path.join(_path_accuracy_tests, "range_test/window/composition_600(*.npz")

    titles_all = np.array(["All", "Olivine", "Orthopyroxene", "Clinopyroxene", "Fa",
                           "Fs (OPX)", "Fs (CPX)", "En (CPX)", "Wo (CPX)"], dtype=str)

    num_colors = len(titles_all)

    window_range, error = load_data_for_plot(pref, error_type)

    error_type, limit = process_error_type(error_type)

    yticks02 = safe_arange(0., 100., 0.2, endpoint=True)
    yticks05 = safe_arange(0., 100., 0.5, endpoint=True)
    yticks1 = safe_arange(0., 100., 1., endpoint=True)
    yticks2 = safe_arange(0., 100., 2., endpoint=True)
    yticks5 = safe_arange(0., 100., 5., endpoint=True)

    outdir_range_tests = path.join(outdir, "range_test")
    check_dir(outdir_range_tests)

    cm = plt.get_cmap("gist_rainbow")

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    ax.set_prop_cycle(color=cm(np.linspace(0., 1., num_colors)))

    for i, line in enumerate(error):
        ax.plot(window_range, line, marker='o', linestyle="--", label=titles_all[i])

    plt.xticks(rotation=-30)

    ax.set_xlabel("Window range (nm)")

    if error_type == "outliers":
        ax.set_ylabel("No. outliers")
    else:
        if limit is None:
            ax.set_ylabel(f"{error_type} (pp)")
        else:
            ax.set_ylabel(f"{error_type} {limit[0]} pp (\%)")

    lim = ax.get_ylim()
    if lim[1] - lim[0] > 16.:
        yticks = yticks5
    elif lim[1] - lim[0] > 8.:
        yticks = yticks2
    elif lim[1] - lim[0] > 4.:
        yticks = yticks1
    elif lim[1] - lim[0] > 2.:
        yticks = yticks05
    else:
        yticks = yticks02

    yticks = yticks[np.logical_and(yticks >= lim[0], yticks <= lim[1])]

    ax.set_yticks(yticks)
    # ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylim(lim)

    ax.legend(bbox_to_anchor=(0., 1.02, 1., 0.2), loc="lower left", mode="expand", borderaxespad=0.,
              ncol=best_blk(num_colors)[1])

    if limit is None:
        fig_name = f"window_{error_type}.{fig_format}"
    else:
        fig_name = f"window_{error_type}{_sep_in}{limit[0]}.{fig_format}"

    plt.draw()
    plt.tight_layout()

    fig.savefig(path.join(outdir_range_tests, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)

    return window_range, error
