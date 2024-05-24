from os import environ, path
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # Import pyplot after setting the backend

import numpy as np
import pandas as pd
import warnings
from datetime import datetime

import seaborn as sns
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras.models import Model

from modules.NN_losses_metrics_activations import my_quantile

from modules.utilities_spectra import error_estimation_bin_like, error_estimation_overall, unique_indices, used_indices
from modules.utilities_spectra import load_npz, load_keras_model, gimme_predicted_class, compute_metrics, is_taxonomical
from modules.utilities_spectra import gimme_model_grid_from_name, gimme_bin_code_from_name

from modules.NN_config_parse import gimme_num_minerals, gimme_endmember_counts, bin_to_cls, bin_to_used

from modules.utilities import check_dir, get_weights_from_model, best_blk, flatten_list, normalise_in_rows, is_constant
from modules.utilities import kernel_density_estimation_1d, kernel_density_estimation_2d, safe_arange
from modules.utilities import split_path, my_polyfit, denoise_array

from modules.NN_config_composition import mineral_names, endmember_names, mineral_names_short

from modules._constants import _path_figures, _path_model, _path_accuracy_tests
from modules._constants import _label_true_name, _label_pred_name, _config_name

# as defaults only
from modules.NN_config_composition import minerals_used, endmembers_used
from modules.NN_config_taxonomy import classes


TEXT_SIZE = 12
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 28

LW = 1

fig_format = "pdf"
big_fig_format = "jpg"  # jpg with higher dpi is sharper and smaller than png with smaller dpi

cbar_kwargs = {"position": "right",
               "size": "5%",
               "pad": 0.1}
savefig_kwargs = {"bbox_inches": "tight",
                  "pad_inches": 0.05,
                  "dpi": 100}
pil_kwargs = {}

plt.rc("font", size=TEXT_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # fontsize of legend
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.rcParams["text.usetex"] = True
plt.rc("text", usetex=True)

plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})
# plt.rc("font", **{"family": "serif", "serif": ["Palatino"]})

outdir = _path_figures
check_dir(outdir)

outdir_composition = path.join(outdir, "composition")
check_dir(outdir_composition)

outdir_taxonomy = path.join(outdir, "taxonomy")
check_dir(outdir_taxonomy)

outdir_surfaces = path.join(outdir, "surfaces")
check_dir(outdir_surfaces)


def change_params(offset: float, reset: bool = False):
    if reset:
        offset = -offset

    plt.rc("font", size=TEXT_SIZE + offset)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE + offset)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE + offset)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE + offset)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE + offset)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE + offset)  # fontsize of legend
    plt.rc("figure", titlesize=BIGGER_SIZE + offset)  # fontsize of the figure title


def plot_scatter_plots(y_true: np.ndarray, y_pred: np.ndarray,
                       used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                       offset: float = 0., suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Scatter plots")

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    change_params(offset)

    LW_scatter = 2.5

    error_estimation_method = "bin"  # "bin" or "overall"

    # limit = 0.25
    shift = 3.  # Control ranges of axes (from 0 - shift to 100 + shift)
    s = 30  # scaling parameter (marker size)
    vmin, vmax = 0., 100.  # bounds for colormap in the scatter plots of chemical
    cmap = "viridis_r"  # cmap of points

    # define the lines
    x_line = safe_arange(-150., 150., endpoint=True)
    y_line = x_line
    y1p_line, y1m_line = y_line + 10., y_line - 10.
    y2p_line, y2m_line = y_line + 20., y_line - 20.
    l0, l10, l20, eb = "k-", "m-", "c-", "r"
    lab_line0, lab_line10, lab_line20 = "0 pp error", "10 pp error", "20 pp error"

    xticks = safe_arange(0., 100., 25., endpoint=True)
    yticks = safe_arange(0., 100., 25., endpoint=True)

    left, right = -shift, 100. + shift
    bottom, top = -shift, 100. + shift

    num_minerals = gimme_num_minerals(used_minerals)

    y_true = y_true[:] * 100.
    y_pred = y_pred[:] * 100.

    mineral_titles = np.array(mineral_names)[used_minerals]
    endmember_titles = [[endmember_names[k][j] for j, endmember in enumerate(endmembers) if endmember]
                        for k, endmembers in enumerate(used_endmembers)]

    RMSE, R2, SAM = compute_metrics(y_true, y_pred, return_r2=True, return_sam=True, used_minerals=used_minerals,
                                    used_endmembers=used_endmembers)
    RMSE, R2, SAM = np.round(RMSE, 1), np.round(R2, 2), np.round(SAM, 1)

    actual_errorbar = np.array([3.])  # pp

    if "bin" in error_estimation_method.lower():
        error_estimation = error_estimation_bin_like
    else:
        error_estimation = error_estimation_overall

    pred_errorbar, true_errorbar = error_estimation(y_true, y_pred,
                                                    actual_error=actual_errorbar,
                                                    used_minerals=used_minerals,
                                                    used_endmembers=used_endmembers)

    # modal first
    start, stop = 0, num_minerals

    if num_minerals > 1:
        x_tmp, y_tmp = y_true[:, start:stop], y_pred[:, start:stop]
        error_pred, error_true = pred_errorbar[start:stop], true_errorbar[start:stop]

        fig, ax = plt.subplots(1, num_minerals, figsize=(4.5 * num_minerals, 6), sharey=True)
        ax = np.ravel(ax)  # to force iterable for the for cycle

        for i, axis in enumerate(ax):
            # lines
            lns1 = axis.plot(x_line, y_line, l0, label=lab_line0, linewidth=LW_scatter, zorder=3)
            lns2 = axis.plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW_scatter, zorder=4)
            axis.plot(x_line, y1m_line, l10, linewidth=LW_scatter, zorder=5)
            lns3 = axis.plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW_scatter, zorder=6)
            axis.plot(x_line, y2m_line, l20, linewidth=LW_scatter, zorder=7)
            # data
            axis.scatter(x_tmp[:, i], y_tmp[:, i], c="black", s=s, zorder=2)
            axis.errorbar(x_tmp[:, i], y_tmp[:, i], yerr=error_pred[i], xerr=error_true[i],
                          fmt=eb, ls="", elinewidth=0.5, zorder=1)

            axis.set_xlabel("Actual (vol\%)")
            axis.tick_params(axis="both")
            axis.axis("square")
            axis.set_title(mineral_titles[i].capitalize())
            axis.set_xticks(xticks)
            axis.set_yticks(yticks)
            axis.set_ylim(bottom=bottom, top=top)
            axis.set_xlim(left=left, right=right)

            axis.text(0.8, 0.15,
                       r"\["  # every line is a separate raw string...
                       r"\begin{split}"  # ...but they are all concatenated by the interpreter :-)
                       r"\mathsf{RMSE} &= " + f"{RMSE[i]:4.1f}" + r"\text{ pp}" + r"\\"
                       r"\mathsf{R}^2 &= " + f"{R2[i]:4.2f}" + r"\\"
                       r"\mathsf{SAM} &= " + f"{SAM[i]:4.1f}" + r"\text{ deg}"
                       r"\end{split}"
                       r"\]",
                       horizontalalignment="center", verticalalignment="center", transform=axis.transAxes)

            lns = lns1 + lns2 + lns3
            labs = [l.get_label() for l in lns]
            axis.legend(lns, labs, loc="upper left", frameon=False)

        ax[0].set_ylabel("Predicted (vol\%)")

        plt.draw()
        plt.tight_layout()

        fig_name = f"scatter_plot_modal{suf}.{fig_format}"
        fig.savefig(path.join(outdir_composition, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
        plt.close(fig)

    # Each mineral on a separate plot
    count_endmembers = gimme_endmember_counts(used_endmembers)
    for i, count_endmember in enumerate(count_endmembers):

        if count_endmember < 2:
            continue

        start, stop = stop, stop + count_endmember

        titles = endmember_titles[i]
        error_pred, error_true = pred_errorbar[start:stop], true_errorbar[start:stop]

        # fig size tuned such that the plots are of approximately the same size
        fig, ax = plt.subplots(1, count_endmember, figsize=(4.4 * count_endmember + 1.5, 6), sharey=True)
        ax = np.ravel(ax)  # to force iterable for the for cycle

        if num_minerals > 1:
            # non-zero modal
            mask = y_true[:, i] > 0
            x_tmp, y_tmp = y_true[mask, start:stop], y_pred[mask, start:stop]
            c = y_true[mask, i]
            error_pred, error_true = error_pred[:, :, mask], error_true[:, :, mask]
        else:
            x_tmp, y_tmp = y_true[:, start:stop], y_pred[:, start:stop]
            c = "black"

        for j, axis in enumerate(ax):
            # lines
            lns1 = axis.plot(x_line, y_line, l0, label=lab_line0, linewidth=LW_scatter, zorder=3)
            lns2 = axis.plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW_scatter, zorder=4)
            axis.plot(x_line, y1m_line, l10, linewidth=LW_scatter, zorder=5)
            lns3 = axis.plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW_scatter, zorder=6)
            axis.plot(x_line, y2m_line, l20, linewidth=LW_scatter, zorder=7)
            # data
            sc = axis.scatter(x_tmp[:, j], y_tmp[:, j], c=c, cmap=cmap, vmin=vmin, vmax=vmax, s=s, zorder=2)
            axis.errorbar(x_tmp[:, j], y_tmp[:, j], yerr=error_pred[j], xerr=error_true[j],
                          fmt=eb, ls="", elinewidth=0.5, zorder=1)

            # if only 1 mineral or if only pure samples (all lines are binary)
            if not (num_minerals <= 1 or is_constant(np.max(y_true[:, :num_minerals], axis=1), constant=100.0)):
                divider = make_axes_locatable(axis)
                cax = divider.append_axes(**cbar_kwargs)
                cbar = plt.colorbar(sc, ax=axis, cax=cax)
                if j == count_endmember - 1:
                    cbar.ax.set_ylabel("Modal abundance (vol\%)")
                else:
                    cbar.remove()

            axis.set_xlabel("Actual")
            axis.tick_params(axis="both")
            axis.axis("square")
            axis.set_title(titles[j])
            axis.set_xticks(xticks)
            axis.set_yticks(yticks)
            axis.set_ylim(bottom=bottom, top=top)
            axis.set_xlim(left=left, right=right)

            axis.text(0.8, 0.15,
                       r"\["  # every line is a separate raw string...
                       r"\begin{split}"  # ...but they are all concatenated by the interpreter :-)
                       r"\mathsf{RMSE} &= " + f"{RMSE[start + j]:4.1f}" + r"\text{ pp}" + r"\\"
                       r"\mathsf{R}^2 &= " + f"{R2[start + j]:4.2f}" + r"\\"
                       r"\mathsf{SAM} &= " + f"{SAM[start + j]:4.1f}" + r"\text{ deg}"
                       r"\end{split}"
                       r"\]",
                       horizontalalignment="center", verticalalignment="center", transform=axis.transAxes)

            lns = lns1 + lns2 + lns3
            labs = [l.get_label() for l in lns]
            axis.legend(lns, labs, loc="upper left", frameon=False)

        ax[0].set_ylabel("Predicted")

        plt.draw()
        plt.tight_layout()

        fig_name = f"scatter_plot_{mineral_names[i]}{suf}.{fig_format}"
        fig.savefig(path.join(outdir_composition, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
        plt.close(fig)

        change_params(offset, reset=True)


def plot_error_evaluation_comp(y_true: np.ndarray, y_pred: np.ndarray,
                               used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                               offset: float = 0., suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Print quantiles")

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    change_params(offset)

    percentile = safe_arange(0., 100., 5., endpoint=True)

    # define the lines
    x_line = safe_arange(-150., 150., endpoint=True)
    y_line_10 = np.ones(np.shape(x_line)) * 10.
    y_line_20 = np.ones(np.shape(x_line)) * 20.
    l10, l20 = "k--", "k--"

    one_sigma = 68.2
    sigma_c, sigma_ls = "k", "--"

    shift = 3.  # Control ranges of axes (from 0 - shift to 100 + shift)

    xticks = safe_arange(0., 100., 10., endpoint=True)
    yticks = safe_arange(0., 100., 10., endpoint=True)

    left, right = -shift, 100. + shift
    bottom, top = -shift, 100. + shift

    all_used_indices = used_indices(used_minerals, used_endmembers)
    num_minerals = gimme_num_minerals(used_minerals)

    quantile = my_quantile(percentile=percentile, used_minerals=used_minerals, used_endmembers=used_endmembers,
                           cleaning=True, all_to_one=False)(y_true, y_pred).numpy()

    titles_all = [mineral_names_short] + endmember_names

    titles_all = np.array(flatten_list(titles_all))[all_used_indices]
    keep_inds = unique_indices(used_minerals, used_endmembers)

    titles_all = titles_all[keep_inds]
    quantile = quantile[:, keep_inds]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if num_minerals > 1:
        n_max = 9
        titles_all = np.insert(titles_all, 0, "All data")
        quantile_all = my_quantile(percentile=percentile, used_minerals=used_minerals, used_endmembers=used_endmembers,
                                   cleaning=True, all_to_one=True)(y_true, y_pred).numpy()

        ax.plot(percentile, quantile_all, linewidth=3, zorder=100)
    else:
        n_max = 10

    if np.shape(quantile)[1] > n_max:
        ax.plot(percentile, quantile[:, :n_max], linewidth=2)
        ax.plot(percentile, quantile[:, n_max:], "--", linewidth=2)
        ncol = 2
    else:
        ax.plot(percentile, quantile, linewidth=2)
        ncol = 1

    # constant error lines
    ax.plot(x_line, y_line_10, l10, zorder=101)
    ax.plot(x_line, y_line_20, l20, zorder=102)

    ax.axvline(one_sigma, color=sigma_c, ls=sigma_ls, zorder=103)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Absolute error (pp)")

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_ylim(bottom=bottom, top=top)
    ax.set_xlim(left=left, right=right)

    ax.legend(titles_all, loc="upper left", ncol=ncol)

    plt.draw()
    plt.tight_layout()

    fig_name = f"quantile_error_plot_composition{suf}.{fig_format}"
    fig.savefig(path.join(outdir_composition, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_error_density_plots(y_true: np.ndarray, y_pred: np.ndarray,
                             used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                             offset: float = 0., suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Print density plots")

    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    change_params(offset)

    nbins = 200
    cmap = "viridis_r"  # cmap of points
    fs = SMALL_SIZE + 6
    error_lim, error_step = 45., 15.

    xticks = safe_arange(0., 100., 25., endpoint=True)
    yticks = safe_arange(-5. * error_step, 5. * error_step, error_step, endpoint=True)

    left, right = 0., 100.
    bottom, top = -error_lim, error_lim

    # define the line styles
    ls_hor, ls_ver = "r--", "k--"

    titles_all = [[f"{name} (vol\%)" for name in mineral_names_short]] + endmember_names
    titles_all = np.array(flatten_list(titles_all))[used_indices(used_minerals, used_endmembers)]
    keep_inds = unique_indices(used_minerals, used_endmembers, return_digits=True)

    titles_all = titles_all[keep_inds]

    fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(22, 12))  # this was set ad hoc; 28, 16

    # 2D density plots first
    for i, inds in enumerate(keep_inds):
        ii, jj = np.unravel_index(i, (3, 4))

        # ad-hoc logic
        if i < 3:  # modal abundances
            y_true_part, y_pred_part = y_true[:, inds], y_pred[:, inds]
        # else filter out samples with w_true = 0
        elif i == 3:  # Fa
            mask = y_true[:, 0] > 0
            y_true_part, y_pred_part = y_true[mask, inds], y_pred[mask, inds]
        elif i == 4:  # Fs (OPX)
            mask = y_true[:, 1] > 0
            y_true_part, y_pred_part = y_true[mask, inds], y_pred[mask, inds]
        elif i > 4:  # CPX chemical
            mask = y_true[:, 2] > 0
            y_true_part, y_pred_part = y_true[mask, inds], y_pred[mask, inds]

        xi, yi, zi = kernel_density_estimation_2d(y_true_part, y_pred_part, nbins=nbins)
        ax[ii, jj].pcolormesh(xi, yi, zi, shading="gouraud", cmap=cmap)
        ax[ii, jj].contour(xi, yi, zi)

        ax[ii, jj].axhline(y=0, linestyle=ls_hor[1:], color=ls_hor[0])

        if jj == 0:
            ax[ii, jj].set_ylabel("Error (pp)", fontsize=fs + 4)

        ax[ii, jj].set_xlabel(titles_all[i], fontsize=fs + 4)

        ax[ii, jj].set_xticks(xticks)
        ax[ii, jj].set_yticks(yticks)
        ax[ii, jj].set_xlim(left=left, right=right)
        ax[ii, jj].set_ylim(bottom=bottom, top=top)

        ax[ii, jj].tick_params(axis="both", labelsize=fs)

    # 1D density plots (stack modal, OL, OPX, CPX)
    ax[2, 0].set_ylabel("Density", fontsize=fs + 4)

    titles_all[:3] = np.array(["OL", "OPX", "CPX"])

    # modal
    ii, jj = 2, 0
    for inds in keep_inds[:3]:
        y_true_part, y_pred_part = y_true[:, inds], y_pred[:, inds]
        xi, zi = kernel_density_estimation_1d(y_true_part, y_pred_part, nbins=nbins)

        ax[ii, jj].plot(xi, zi, label=titles_all[i])

    # Fa
    ii, jj = 2, 1
    i = 3
    mask = y_true[:, 0] > 0
    y_true_part, y_pred_part = y_true[mask, keep_inds[i]], y_pred[mask, keep_inds[i]]
    xi, zi = kernel_density_estimation_1d(y_true_part, y_pred_part, nbins=nbins)

    ax[ii, jj].plot(xi, zi, label=titles_all[i])

    # Fs (OPX)
    ii, jj = 2, 2
    i = 4
    mask = y_true[:, 1] > 0
    y_true_part, y_pred_part = y_true[mask, keep_inds[i]], y_pred[mask, keep_inds[i]]
    xi, zi = kernel_density_estimation_1d(y_true_part, y_pred_part, nbins=nbins)

    ax[ii, jj].plot(xi, zi, label=titles_all[i])

    # chemical CPX
    ii, jj = 2, 3
    mask = y_true[:, 2] > 0
    for i in range(5, len(keep_inds)):
        y_true_part, y_pred_part = y_true[mask, keep_inds[i]], y_pred[mask, keep_inds[i]]
        xi, zi = kernel_density_estimation_1d(y_true_part, y_pred_part, nbins=nbins)

        ax[ii, jj].plot(xi, zi, label=titles_all[i])

    ii = 2
    for jj in range(4):
        ax[ii, jj].axvline(x=0, linestyle=ls_ver[1:], color=ls_ver[0])

        ax[ii, jj].set_xticks(yticks)
        ax[ii, jj].set_xlim(left=bottom, right=top)
        ax[ii, jj].tick_params(axis="both", labelsize=fs)
        ax[ii, jj].set_xlabel("Error (pp)", fontsize=fs + 4)
        ax[ii, jj].legend(loc="upper right", fontsize=SMALL_SIZE)

        _, end = ax[ii, jj].get_ylim()
        stepsize = 0.04 if end > 0.15 else 0.02
        ax[ii, jj].set_yticks(safe_arange(0, end, stepsize, endpoint=True))

    plt.draw()
    plt.tight_layout()

    fig_name = f"density_error_plot{suf}.{big_fig_format}"
    fig.savefig(path.join(outdir_composition, fig_name), format=big_fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          labels: dict[str, int] | list |np.ndarray | None = None,
                          offset: float = 0., suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Confusion matrix")

    if labels is None: labels = classes
    if isinstance(labels, dict):
        labels = list(labels.keys())
    labels = np.array(labels, dtype=str)

    change_params(offset)

    true_classes = gimme_predicted_class(y_true, used_classes=labels, return_index=True)
    pred_classes = gimme_predicted_class(y_pred, used_classes=labels, return_index=True)

    array = confusion_matrix(true_classes, pred_classes)

    sum_cols = np.sum(array, axis=0)  # true positive + false positive
    sum_rows = np.sum(array, axis=1)  # true positive + false negative

    precision = np.round(np.diag(array) / sum_cols * 100.)  # in %; here might be zero division
    recall = np.round(np.diag(array) / sum_rows * 100.)  # in %

    precision_str = np.array(np.array(precision, dtype=int), dtype=str)
    recall_str = np.array(np.array(recall, dtype=int), dtype=str)

    # remove NaNs
    precision_str[np.where(sum_cols == 0.)[0]] = "0"
    recall_str[np.where(sum_rows == 0.)[0]] = "0"

    # dim = list(range(y_true.shape[1]))
    # dim = np.unique(gimme_predicted_class(y_true, return_index=True))  # true values in the confusion matrix
    dim = np.unique(np.union1d(true_classes, pred_classes))
    # normalise colours to the sum of rows
    df_cm = pd.DataFrame(normalise_in_rows(array), dim, dim)
    df_cm.iloc[sum_rows == 0.] = 0.

    labels_to_use = np.array(labels)[dim]

    fig = plt.figure("Confusion Matrix", figsize=(18, 15))
    ax1 = sns.heatmap(df_cm, annot=array, fmt="d", annot_kws={"size": SMALL_SIZE}, cmap="Blues", cbar=False)

    # Plot diagonal line
    ax1.plot([0, np.max(dim) + 1], [0, np.max(dim) + 1], "k--")

    ax2 = ax1.twinx()
    ax3 = ax1.twiny()

    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    ax1.tick_params(length=0)
    ax2.tick_params(length=0)
    ax3.tick_params(length=0)

    ax1.xaxis.tick_top()  # x-axis on top
    ax1.xaxis.set_label_position("top")

    ax3.xaxis.tick_bottom()  # x-axis on bottom
    ax3.xaxis.set_label_position("bottom")

    ax1.set_xlabel("Predicted taxonomy class")
    ax1.set_ylabel("Actual taxonomy class")
    ax3.set_xlabel("Precision (\%)")
    ax2.set_ylabel("Recall (\%)")

    ax1.set_xticklabels(labels_to_use)
    ax1.set_yticklabels(labels_to_use)

    ax2.set_yticks(ax1.get_yticks())
    ax2.set_yticklabels(recall_str[::-1], ha="right")
    ax2.set_ylim([0, ax1.get_ylim()[0]])

    yax = ax2.get_yaxis()
    yax.set_tick_params(pad=27)
    # ax2.yaxis.set_label_coords(1.048, 0.5)

    ax3.set_xticks(ax1.get_xticks())
    ax3.set_xticklabels(precision_str)
    ax3.set_xlim([0, ax1.get_xlim()[1]])

    ax1.axhline(y=0, color="k", linewidth=10)
    ax1.axhline(y=df_cm.shape[1], color="k", linewidth=10)
    ax1.axvline(x=0, color="k", linewidth=10)
    ax1.axvline(x=df_cm.shape[0], color="k", linewidth=10)

    """
    nx, ny = np.shape(array)
    for ix in range(nx):
        for iy in range(ny):
            ax.text(iy + 0.5, ix + 0.5, int(array[ix, iy]), ha="center", va="center", color="r")
    """

    plt.draw()
    plt.tight_layout()

    fig_name = f"confusion_matrix{suf}.{fig_format}"

    fig.savefig(path.join(outdir_taxonomy, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_error_evaluation_class(y_true: np.ndarray, y_pred: np.ndarray,
                                labels: dict[str, int] | list |np.ndarray | None = None,
                                offset: float = 0., suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Print quantiles")

    if labels is None: labels = classes
    if isinstance(labels, dict):
        labels = list(labels.keys())
    labels = np.array(labels, dtype=str)

    change_params(offset)

    labels = np.array(labels, dtype="<U8")  # This must be here if you want to add "All data" to the labels

    percentile = safe_arange(0., 100., 5., endpoint=True)

    # define the lines
    x_line = safe_arange(-150., 150., endpoint=True)
    y_line_30 = np.ones(np.shape(x_line)) * 30.
    y_line_50 = np.ones(np.shape(x_line)) * 50.
    l30, l50 = "k--", "k--"

    one_sigma = 68.2
    sigma_c, sigma_ls = "k", "--"

    shift = 3.  # Control ranges of axes (from 0 - shift to 100 + shift)

    xticks, yticks = safe_arange(0., 100., 10., endpoint=True), safe_arange(0., 100., 10., endpoint=True)
    left, right = -shift, 100. + shift
    bottom, top = -shift, 100. + shift

    mask = np.where(y_true == 1)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    quantile = np.zeros((len(percentile), len(labels)))

    for j in range(len(labels)):
        y_true_part = y_true[mask[1] == j]
        y_pred_part = y_pred[mask[1] == j]

        quantile[:, j] = my_quantile(percentile=percentile, used_minerals=None, used_endmembers=None, cleaning=False,
                                     all_to_one=False)(y_true_part, y_pred_part).numpy()

    keep_inds = np.all(np.isfinite(quantile), axis=0)
    quantile, titles_all = quantile[:, keep_inds], labels[keep_inds]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    n_max = 9
    titles_all = np.insert(titles_all, 0, "All data")
    quantile_all = my_quantile(percentile=percentile, used_minerals=None, used_endmembers=None, cleaning=False,
                               all_to_one=True)(y_true, y_pred).numpy()

    ax.plot(percentile, quantile_all, linewidth=3, zorder=100)

    if np.shape(quantile)[1] > n_max:
        ax.plot(percentile, quantile[:, :n_max], linewidth=2)
        ax.plot(percentile, quantile[:, n_max:], "--", linewidth=2)
        ncol = 2
    else:
        ax.plot(percentile, quantile, linewidth=2)
        ncol = 1

    # constant error lines
    ax.plot(x_line, y_line_30, l30, zorder=101)
    ax.plot(x_line, y_line_50, l50, zorder=102)

    ax.axvline(one_sigma, color=sigma_c, ls=sigma_ls, zorder=103)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Absolute error (pp)")

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_ylim(bottom=bottom, top=top)
    ax.set_xlim(left=left, right=right)

    ax.legend(titles_all, loc="upper left", ncol=ncol)

    plt.draw()
    plt.tight_layout()

    fig_name = f"quantile_error_plot_taxonomy{suf}.{fig_format}"
    fig.savefig(path.join(outdir_taxonomy, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_model_history(model: Model, offset: float = 0., quiet: bool = False) -> None:
    if not quiet:
        print("Model history")

    change_params(offset)

    save_all = False  # set to True to not overwrite the plot

    left, right = 0., model.history.params["epochs"]
    bottom = 0.

    color1, color2 = "tab:red", "tab:blue"

    sigma = 5.

    history = model.history.history

    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    plot1 = denoise_array(history["loss"], sigma=sigma)
    lns1 = ax1.plot(plot1, color=color1, linestyle="-", label="Loss - training")

    if len(model.metrics_names) == 1:  # no metrics were used
        metric_name = ""
    else:
        metric_name = model.metrics_names[1]

    if metric_name:
        metrics = history[metric_name]
        if metric_name == "mse":  # MSE to RMSE
            metrics = np.sqrt(metrics)
            labely = "RMSE"

        elif metric_name in ["mse", "mae", "rmse"]:
            labely = metric_name.upper()

        else:
            labely = metric_name.capitalize()

        plot3 = denoise_array(metrics, sigma=sigma)
        labely = str(np.char.replace(labely, "_", " "))

        ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
        lns3 = ax2.plot(plot3, color=color2, linestyle="-", label=f"{labely} - training")

    if "val_loss" in history.keys():
        plot2 = denoise_array(history["val_loss"], sigma=sigma)
        lns2 = ax1.plot(plot2, color=color1, linestyle=":", label="Loss - validation")

        if metric_name:
            metrics = history[f"val_{metric_name}"]

            if metric_name == "mse":  # MSE to RMSE
                metrics = np.sqrt(metrics)

            plot4 = denoise_array(metrics, sigma=sigma)
            lns4 = ax2.plot(plot4, color=color2, linestyle=":", label=f"{labely} - validation")

    if "val_loss" in history.keys():
        if metric_name:
            lns = lns1 + lns2 + lns3 + lns4
        else:
            lns = lns1 + lns2
    else:
        if metric_name:
            lns = lns1 + lns3
        else:
            lns = lns1

    ax1.set_xlabel("Epoch")
    ax1.tick_params(axis="x")
    ax1.set_ylabel("Loss", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(bottom=bottom)
    ax1.set_xlim(left=left, right=right)
    ax1.grid(False)

    if metric_name:
        ax2.set_ylabel(labely, color=color2)  # we already handled the x-label with ax1
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(bottom=bottom)
        ax2.grid(False)

    labs = [l.get_label() for l in lns]
    if metric_name:
        if plot3[0] > plot3[-1]:  # the metric decreases if quality increases
            loc = "upper right"
        else:
            loc = "center right"
    else:
        loc = "upper right"
    ax1.legend(lns, labs, loc=loc)

    fig.tight_layout()  # Otherwise the right y-label is slightly clipped
    # plt.title("Model history")

    plt.draw()

    if save_all:
        dt_string = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        fig_name = f"{dt_string}_model_history.{fig_format}"
    else:
        fig_name = f"model_history.{fig_format}"

    fig.savefig(path.join(outdir, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_model_layer(model_name: str, subfolder_model: str = "", layer: str = "Conv1D",
                     offset: float = 0., suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print(f"Plotting {layer} layer")

    change_params(offset)

    shift = 0.25

    model = load_keras_model(model_name, subfolder=subfolder_model)
    layers = get_weights_from_model(model)

    try:
        layer_name = next(obj for obj in layers.keys() if layer in obj)
    except StopIteration:
        warnings.warn(f"No {layer} layer in the model.")
        return

    weights = layers[layer_name][0]  # [0] to get only weights, not biases
    weights = np.reshape(weights, (np.shape(weights)[0], -1))  # optimised for conv1d layers

    # center the x values
    x = safe_arange(len(weights))
    x -= np.floor(len(weights) / 2.)

    left, right = np.min(x) - shift, np.max(x) + shift

    # sort weights from higher amplitude to lowest
    sorted_inds = np.argsort(np.max(np.abs(weights), 0))[::-1]

    blk = best_blk(np.shape(weights)[1])

    fig, ax = plt.subplots(blk[0], blk[1], figsize=(4 * blk[0], 4 * blk[1]), sharex=True, squeeze=False)
    ax = np.reshape(ax, (blk[0], blk[1]))  # force dimensions for the for cycle

    c = 0
    for row in range(blk[0]):
        for column in range(blk[1]):
            if c < np.shape(weights)[1]:
                if "1D" in layer:
                    ax[row, column].plot(x, weights[:, sorted_inds[c]])
                else:  # "2D"
                    pass
                    # ax[row, column].imshow(weights[:, sorted_inds[c]], extent=[np.min(x), np.max(x), np.min(x), np.max(x)], aspect="auto")
                if row == blk[0] - 1:  # last row
                    ax[row, column].set_xlabel("$\Delta x$")
                ax[row, column].set_xticks(x)
                ax[row, column].set_xlim(left=left, right=right)
                c += 1
        ax[row, 0].set_ylabel("Weight")

    plt.draw()
    plt.tight_layout()

    if path.isdir(path.join(_path_model, subfolder_model, model_name)) or path.isdir(model_name):
        model_name = model_name.rstrip(path.sep)
        _, name, ext = split_path(model_name)
        fig_name = f"{name}_{ext}" if ext else name
    else:
        fig_name = f"{split_path(model_name)[1]}"

    fig_name = fig_name.replace(".", "_")
    fig_name += f"{suf}.{fig_format}"

    outdir_conv_plot = path.join(outdir, subfolder_model)
    check_dir(outdir_conv_plot)

    fig.savefig(path.join(outdir_conv_plot, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_corr_matrix(labels: np.ndarray, corr_matrix: pd.DataFrame,
                     offset: float = 0., suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Correlation matrix")

    change_params(offset)

    # polynom to adjust font size for various corr_matrix sizes
    x, y = [0., 16., 40.], [0.5, 1., 2.]
    fs_multiply = np.polyval(my_polyfit(x, y, 2), len(labels))

    fs_text = SMALL_SIZE * fs_multiply
    fs_small = SMALL_SIZE * fs_multiply
    fs_med = MEDIUM_SIZE * fs_multiply

    xticks, yticks = safe_arange(len(labels)), safe_arange(len(labels))

    fig, ax = plt.subplots(1, 1, figsize=np.shape(corr_matrix))
    im = ax.matshow(corr_matrix, vmin=-1, vmax=1, cmap="seismic")

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    if plt.rcParams["text.usetex"]:
        labels = np.char.replace(labels, "_", "\_")
    ax.set_xticklabels(labels, rotation=90, fontsize=fs_small)
    ax.set_yticklabels(labels, fontsize=fs_small)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(**cbar_kwargs)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=fs_small)
    cbar.ax.set_ylabel("Correlation coefficient", fontsize=fs_med)

    corr_matrix = np.round(corr_matrix.to_numpy(), 2)

    color = np.full(np.shape(corr_matrix), "w")
    color[np.logical_or(np.abs(corr_matrix) < 0.25, ~np.isfinite(corr_matrix))] = "k"

    for ix in range(len(labels)):
        for iy in range(len(labels)):
            ax.text(iy, ix, f"{corr_matrix[ix, iy]:.2f}",
                    ha="center", va="center", color=color[ix, iy], fontsize=fs_text)

    plt.draw()
    plt.tight_layout()

    fig_name = f"correlation_matrix{suf}.{fig_format}"
    fig.savefig(path.join(outdir, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def composition_plots(y_true: np.ndarray, y_pred: np.ndarray,
                      used_minerals: np.ndarray | None = None, used_endmembers: list[list[bool]] | None = None,
                      density_plot: bool = False, suf: str = "", quiet: bool = True) -> None:
    if used_minerals is None: used_minerals = minerals_used
    if used_endmembers is None: used_endmembers = endmembers_used

    plot_scatter_plots(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers,
                       suf=suf, quiet=quiet)
    plot_error_evaluation_comp(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers,
                               suf=suf, quiet=quiet)
    if density_plot:
        plot_error_density_plots(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers,
                                 suf=suf, quiet=quiet)


def taxonomy_plots(y_true: np.ndarray, y_pred: np.ndarray,
                   used_classes: dict[str, int] | list | np.ndarray | None = None,
                   suf: str = "", quiet: bool = True) -> None:
    if used_classes is None: used_classes = classes

    if isinstance(used_classes, dict):
        used_classes = list(used_classes.keys())
    used_classes = np.array(used_classes, dtype=str)

    plot_confusion_matrix(y_true, y_pred, labels=used_classes, suf=suf, quiet=quiet)
    plot_error_evaluation_class(y_true, y_pred, labels=used_classes, suf=suf, quiet=quiet)


def result_plots(y_true: np.ndarray, y_pred: np.ndarray, bin_code: str, density_plot: bool = False,
                 suf: str = "", quiet: bool = True) -> None:
    if is_taxonomical(bin_code=bin_code):
        used_classes = bin_to_cls(bin_code=bin_code)

        taxonomy_plots(y_true, y_pred, used_classes=used_classes, suf=suf, quiet=quiet)

    else:
        used_minerals, used_endmembers = bin_to_used(bin_code=bin_code)

        composition_plots(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers,
                          density_plot=density_plot, suf=suf, quiet=quiet)


def plot_acc_test_results(filename: str) -> None:
    full_path = path.join(_path_accuracy_tests, filename)
    data = load_npz(full_path)

    y_true, y_pred = data[_label_true_name], data[_label_pred_name]

    try:
        config = data[_config_name][()]
        model_grid = config["grid_setup"]["model_grid"]
        bin_code = config["output_setup"]["bin_code"]
    except KeyError:
        model_grid = gimme_model_grid_from_name(full_path)
        bin_code = gimme_bin_code_from_name(full_path)

    suf = f"_{model_grid}_accuracy_test"

    result_plots(y_true, y_pred, bin_code=bin_code, density_plot=True, suf=suf)
