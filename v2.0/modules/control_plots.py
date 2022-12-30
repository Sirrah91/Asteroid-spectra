from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib as mpl
import numpy as np
import pandas as pd
from typing import Literal
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.models import Functional
from scipy.ndimage import gaussian_filter1d
import re

from modules.NN_losses_metrics_activations import my_rmse, my_sam, my_r2, my_quantile, custom_objects

from modules.utilities_spectra import error_estimation_bin_like, error_estimation_overall, unique_indices, used_indices
from modules.utilities import check_dir, get_weights_from_model, best_blk, flatten_list, normalise_in_rows
from modules.utilities import kernel_density_estimation_1d, kernel_density_estimation_2d, safe_arange

from modules.NN_config import mineral_names, endmember_names, mineral_names_short

from modules._constants import _path_figures, _path_model, _path_accuracy_test

mpl.use("Agg")

size_offset = 0
TEXT_SIZE = 12 + size_offset
SMALL_SIZE = 16 + size_offset
MEDIUM_SIZE = 20 + size_offset
BIGGER_SIZE = 24 + size_offset

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

outdir_composition = "".join((outdir, "/compositional/"))
check_dir(outdir_composition)

outdir_taxonomy = "".join((outdir, "/taxonomical/"))
check_dir(outdir_taxonomy)

outdir_surfaces = "".join((outdir, "/surfaces/"))
check_dir(outdir_surfaces)


def plot_scatter_plots(y_true: np.ndarray, y_pred: np.ndarray, used_minerals: np.ndarray, used_endmembers: np.ndarray,
                       pure_only: bool, mix_of_the_pure_ones: bool, suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Scatter plots")

    LW_scatter = 2.5

    # limit = 0.25
    shift = 3.  # Control ranges of axes (from 0 - shift to 100 + shift)
    s = 30  # scaling parameter (marker size)
    vmin, vmax = 0., 100.  # bounds for colormap in scatter of chemical
    cmap = "viridis_r"  # cmap of points

    # define the lines
    x_line = safe_arange(-150., 150., endpoint=True)
    y_line = x_line
    y1p_line, y1m_line = y_line + 10., y_line - 10.
    y2p_line, y2m_line = y_line + 20., y_line - 20.
    l0, l10, l20, eb = "k-", "m-", "c-", "r"

    xticks, yticks = safe_arange(0., 100., 25., endpoint=True), safe_arange(0., 100., 25., endpoint=True)
    left, right = -shift, 100. + shift
    bottom, top = -shift, 100. + shift

    number_of_minerals = int(np.sum(used_minerals))

    y_true = y_true[:] * 100.
    y_pred = y_pred[:] * 100.

    mineral_titles = np.array(mineral_names)[used_minerals]
    endmember_titles = [[endmember_names[k][j] for j, endmember in enumerate(endmembers) if endmember]
                        for k, endmembers in enumerate(used_endmembers)]

    lab_line0, lab_line10, lab_line20 = "0 pp error", "10 pp error", "20 pp error"

    RMSE = my_rmse(number_of_minerals)(y_true, y_pred).numpy() / 100.  # is multiplied with 100 in the code
    R2 = my_r2(number_of_minerals)(y_true, y_pred).numpy()
    SAM = my_sam(number_of_minerals)(y_true, y_pred).numpy() * 180. / np.pi

    actual_errorbar = np.array([3])  # pp

    """
    pred_errorbar, true_errorbar = error_estimation_overall(y_true, y_pred, num_minerals=number_of_minerals, 
                                                            actual_error=actual_errorbar)
    """
    pred_errorbar, true_errorbar = error_estimation_bin_like(y_true, y_pred, num_minerals=number_of_minerals,
                                                             num_labels=len(
                                                                 flatten_list(endmember_titles)) + number_of_minerals,
                                                             actual_error=actual_errorbar)

    pred_errorbar[pred_errorbar < 0] = 0
    true_errorbar[true_errorbar < 0] = 0

    # modal first
    start, stop = 0, number_of_minerals

    if number_of_minerals > 1:
        x_tmp, y_tmp = y_true[:, start:stop], y_pred[:, start:stop]
        error_pred, error_true = pred_errorbar[start:stop], true_errorbar[start:stop]

        fig, ax = plt.subplots(1, number_of_minerals, figsize=(4.5 * number_of_minerals, 6))
        for i, axis in enumerate(ax):
            # lines
            lns1 = axis.plot(x_line, y_line, l0, label=lab_line0, linewidth=LW_scatter, zorder=3)
            lns2 = axis.plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW_scatter, zorder=4)
            axis.plot(x_line, y1m_line, l10, linewidth=LW_scatter, zorder=5)
            lns3 = axis.plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW_scatter, zorder=6)
            axis.plot(x_line, y2m_line, l20, linewidth=LW_scatter, zorder=7)
            # data
            axis.scatter(x_tmp[:, i], y_tmp[:, i], c="black", s=s, zorder=2)
            axis.errorbar(x_tmp[:, i], y_tmp[:, i], yerr=error_pred[i],
                           xerr=error_true[i], fmt=eb, ls="", zorder=1)

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
                       r"\mathsf{RMSE} &= " + "{:4.1f}".format(RMSE[i]) + r"\text{ pp}" + r"\\"
                                                                                          r"\mathsf{R}^2 &= " + "{:4.2f}".format(
                           R2[i]) + r"\\"
                                    r"\mathsf{SAM} &= " + "{:4.1f}".format(SAM[i]) + r"\text{ deg}"
                                                                                     r"\end{split}"
                                                                                     r"\]",
                       horizontalalignment="center", verticalalignment="center", transform=axis.transAxes)

            if i > 0:
                axis.set_yticklabels([])

            lns = lns1 + lns2 + lns3
            labs = [l.get_label() for l in lns]
            axis.legend(lns, labs, loc="upper left", frameon=False)

        ax[0].set_ylabel("Predicted (vol\%)")

        plt.draw()
        plt.tight_layout()

        fig_name = "".join(("scatter_plot_modal", suf, ".", fig_format))
        fig.savefig("".join((outdir_composition, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
        plt.close(fig)

    # each mineral on separate plot
    for i in range(len(used_endmembers)):
        num_endmembers = int(np.sum(used_endmembers[i]))

        if num_endmembers < 2:
            continue

        start, stop = stop, stop + num_endmembers

        titles = endmember_titles[i]
        error_pred, error_true = pred_errorbar[start:stop], true_errorbar[start:stop]

        # fig size tuned such that the plots are approximately of the same size
        fig, ax = plt.subplots(1, num_endmembers, figsize=(4.4 * num_endmembers + 1.5, 6))

        if number_of_minerals > 1:
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
            axis.errorbar(x_tmp[:, j], y_tmp[:, j], yerr=error_pred[j],
                           xerr=error_true[j], fmt=eb, ls="", zorder=1)

            # if only 1 mineral or if only pure samples, colorbar is not needed
            if not (number_of_minerals <= 1 or (pure_only and not mix_of_the_pure_ones)):
                divider = make_axes_locatable(axis)
                cax = divider.append_axes(**cbar_kwargs)
                cbar = plt.colorbar(sc, ax=axis, cax=cax)
                if j == num_endmembers - 1:
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
                       r"\mathsf{RMSE} &= " + "{:4.1f}".format(RMSE[start + j]) + r"\text{ pp}" + r"\\"
                                                                                                  r"\mathsf{R}^2 &= " + "{:4.2f}".format(
                           R2[start + j]) + r"\\"
                                            r"\mathsf{SAM} &= " + "{:4.1f}".format(SAM[start + j]) + r"\text{ deg}"
                                                                                                     r"\end{split}"
                                                                                                     r"\]",
                       horizontalalignment="center", verticalalignment="center", transform=axis.transAxes)
            if j > 0:
                axis.set_yticklabels([])

            lns = lns1 + lns2 + lns3
            labs = [l.get_label() for l in lns]
            axis.legend(lns, labs, loc="upper left", frameon=False)

        ax[0].set_ylabel("Predicted")

        plt.draw()
        plt.tight_layout()

        fig_name = "".join(("scatter_plot_", mineral_names[i], suf, ".", fig_format))
        fig.savefig("".join((outdir_composition, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
        plt.close(fig)


def plot_error_evaluation(y_true: np.ndarray, y_pred: np.ndarray, used_minerals: np.ndarray,
                          used_endmembers: list[list[bool]], suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Print quantiles")

    percentile = safe_arange(0., 100., 5., endpoint=True)

    # define the lines
    x_line = safe_arange(-150., 150., endpoint=True)
    y_line_10 = np.ones(np.shape(x_line)) * 10.
    y_line_20 = np.ones(np.shape(x_line)) * 20.
    l10, l20 = "k--", "k--"

    shift = 3.  # Control ranges of axes (from 0 - shift to 100 + shift)

    xticks, yticks = safe_arange(0., 100., 10., endpoint=True), safe_arange(0., 100., 10., endpoint=True)
    left, right = -shift, 100. + shift
    bottom, top = -shift, 100. + shift

    all_used_indices = used_indices(used_minerals, used_endmembers)
    number_of_minerals = int(np.sum(used_minerals))

    quantile = my_quantile(number_of_minerals, percentile)(y_true, y_pred).numpy()

    titles_all = [mineral_names_short] + endmember_names

    titles_all = np.array(flatten_list(titles_all))[all_used_indices]
    keep_inds = unique_indices(used_minerals, used_endmembers)

    titles_all = titles_all[keep_inds]
    quantile = quantile[:, keep_inds]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if number_of_minerals > 1:
        n_max = 9
        titles_all = np.insert(titles_all, 0, "All data")
        quantile_all = my_quantile(number_of_minerals, percentile, True)(y_true, y_pred).numpy()

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

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Absolute error (pp)")

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_ylim(bottom=bottom, top=top)
    ax.set_xlim(left=left, right=right)

    ax.legend(titles_all, loc="upper left", ncol=ncol)

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(("quantile_error_plot", suf, ".", fig_format))
    fig.savefig("".join((outdir_composition, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_error_density_plots(y_true: np.ndarray, y_pred: np.ndarray, used_minerals: np.ndarray,
                             used_endmembers: list[list[bool]], suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Print density plots")

    nbins = 200
    cmap = "viridis_r"  # cmap of points
    fs = SMALL_SIZE + 6
    error_lim, error_step = 45., 15.

    xticks, yticks = safe_arange(0., 100., 25., endpoint=True), safe_arange(-5. * error_step, 5. * error_step,
                                                                            error_step, endpoint=True)
    left, right = 0., 100.
    bottom, top = -error_lim, error_lim

    # define the line styles
    ls_hor, ls_ver = "r--", "k--"

    titles_all = [["".join((s, " (vol\%)")) for s in mineral_names_short]] + endmember_names
    titles_all = np.array(flatten_list(titles_all))[used_indices(used_minerals, used_endmembers)]
    keep_inds = unique_indices(used_minerals, used_endmembers, return_digits=True)

    titles_all = titles_all[keep_inds]

    fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(22, 12))  # this was set ad hoc; 28, 16

    # 2D density plots first
    for i, inds in enumerate(keep_inds):
        ii, jj = np.unravel_index(i, (3, 4))

        # ad hoc logic
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
        ax[ii, jj].pcolormesh(xi, yi, zi.reshape(xi.shape), shading="gouraud", cmap=cmap)
        ax[ii, jj].contour(xi, yi, zi.reshape(xi.shape))

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

    fig_name = "".join(("density_error_plot", suf, ".", big_fig_format))
    fig.savefig("".join((outdir_composition, "/", fig_name)), format=big_fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_conv_kernels(model_name: str, subfolder_model: str, layer: str = "Conv1D", quiet: bool = False) -> None:
    from keras.models import load_model

    if quiet:
        print("Plotting convolution kernels")

    shift = 0.25

    model = load_model("".join((_path_model, subfolder_model, "/", model_name)),
                       custom_objects=custom_objects)

    layers = get_weights_from_model(model)

    try:
        layer_name = next(obj for obj in layers.keys() if layer in obj)
    except StopIteration:
        warnings.warn("No {layer:s} layer in the model.".format(layer=layer))
        return

    weights = np.squeeze(layers[layer_name][0])  # [0] to get only weights, not biases

    x = safe_arange(len(weights))
    x = x - np.floor(len(x) / 2.)

    left, right = np.min(x) - shift, np.max(x) + shift

    # sort weights from higher amplitude to lowest
    sorted_inds = np.argsort(np.max(np.abs(weights), 0))[::-1]

    blk = best_blk(np.shape(weights)[1])

    fig, ax = plt.subplots(blk[0], blk[1], figsize=(4 * blk[0], 4 * blk[1]))

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

    fig_name = "".join(("conv_kernels_", model_name, "_.", fig_format))
    fig.savefig("".join((outdir, "/", subfolder_model, "/", fig_name)), format=fig_format,
                **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_model_history(model: Functional, quiet: bool = False) -> None:
    if not quiet:
        print("Model history")

    left, right = 0., model.history.params["epochs"]
    bottom = 0.

    color1, color2 = "tab:red", "tab:blue"

    sigma = 5.

    history = model.history.history

    fig = plt.figure("Loss and accuracy", figsize=(8, 6))
    ax1 = fig.add_subplot(111)

    correction = gaussian_filter1d(np.ones((len(history["loss"]),)), sigma=sigma, mode="constant")
    plot1 = gaussian_filter1d(history["loss"], sigma=sigma, mode="constant") / correction

    if model.metrics_names[1] == "mse":  # MSE to RMSE
        plot3 = gaussian_filter1d(np.sqrt(history[model.metrics_names[1]]), sigma=sigma, mode="constant") / correction
        labely = "Rmse"
    else:
        plot3 = gaussian_filter1d(history[model.metrics_names[1]], sigma=sigma, mode="constant") / correction
        labely = model.metrics_names[1].capitalize()

    labely = str(np.char.replace(labely, "_", " "))

    lns1 = ax1.plot(plot1, color=color1, linestyle="-", label="Loss - training")
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    lns3 = ax2.plot(plot3, color=color2, linestyle="-", label="".join((labely, " - training")))

    if "val_loss" in history.keys():
        plot2 = gaussian_filter1d(history["val_loss"], sigma=sigma, mode="constant") / correction
        if model.metrics_names[1] == "mse":  # MSE to RMSE
            plot4 = gaussian_filter1d(np.sqrt(history["".join(("val_", model.metrics_names[1]))]),
                                      sigma=sigma, mode="constant") / correction
        else:
            plot4 = gaussian_filter1d(history["".join(("val_", model.metrics_names[1]))],
                                      sigma=sigma, mode="constant") / correction
        lns2 = ax1.plot(plot2, color=color1, linestyle=":", label="loss - validation")
        lns4 = ax2.plot(plot4, color=color2, linestyle=":", label="".join((labely, " - validation")))

        lns = lns1 + lns2 + lns3 + lns4
    else:
        lns = lns1 + lns3

    ax1.set_xlabel("Epoch")
    ax1.tick_params(axis="x")
    ax1.set_ylabel("Loss", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(bottom=bottom)
    ax1.set_xlim(left=left, right=right)
    ax1.grid(False)

    ax2.set_ylabel(labely, color=color2)  # we already handled the x-label with ax1
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(bottom=bottom)
    ax2.grid(False)

    labs = [l.get_label() for l in lns]
    if plot3[0] > plot3[-1]:  # the metric decreases if quality increases
        loc = "upper right"
    else:
        loc = "center right"
    ax1.legend(lns, labs, loc=loc)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title("Model history")

    plt.draw()
    """
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    fig_name = "".join((dt_string, "_", "model_history.", fig_format)) 
    """
    fig_name = "".join(("model_history.", fig_format))
    fig.savefig("".join((outdir, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray,
                          suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Confusion matrix")

    array = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    sum_cols = np.sum(array, axis=0)  # true positive + false positive
    sum_rows = np.sum(array, axis=1)  # true positive + false negative

    precision = np.round(np.diag(array) / sum_cols * 100.)  # in %; here might be zero division
    recall = np.round(np.diag(array) / sum_rows * 100.)  # in %

    precision_str = np.array(np.array(precision, dtype=int), dtype=str)
    recall_str = np.array(np.array(recall, dtype=int), dtype=str)

    # remove NaNs
    precision_str[np.where(sum_cols == 0)[0]] = "0"
    recall_str[np.where(sum_rows == 0)[0]] = "0"  # this should never happen

    dim = np.unique(y_true.argmax(axis=1))  # true values in the confusion matrix  # list(range(y_true.shape[1]))
    # normalise colours to sum of rows
    df_cm = pd.DataFrame(normalise_in_rows(array, sum_rows), dim, dim)

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

    ax1.xaxis.tick_top()  # x axis on top
    ax1.xaxis.set_label_position("top")

    ax3.xaxis.tick_bottom()  # x axis on bottom
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

    fig_name = "".join(("confusion_matrix", suf, ".", fig_format))

    fig.savefig("".join((outdir_taxonomy, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_corr_matrix(labels: np.ndarray, corr_matrix: pd.DataFrame, suf: str, quiet: bool = False) -> None:
    if not quiet:
        print("Correlation matrix")

    xticks, yticks = safe_arange(len(labels)), safe_arange(len(labels))

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.matshow(corr_matrix, vmin=-1, vmax=1)

    ax.set_xticks(xticks, rotation=90)
    ax.set_yticks(yticks)

    if plt.rcParams["text.usetex"]:
        labels = np.char.replace(labels, "_", "\_")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(**cbar_kwargs)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.set_ylabel("Correlation")

    for ix in range(len(labels)):
        for iy in range(len(labels)):
            ax.text(iy, ix, "".join("{:.2f}".format(corr_matrix.to_numpy()[ix, iy])),
                    ha="center", va="center", color="r", fontsize=SMALL_SIZE)

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(("correlation_matrix", suf, ".", fig_format))
    fig.savefig("".join((outdir, "/", fig_name)), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)


def plot_acc_test_results(filename: Literal["compositional", "taxonomical"]) -> None:
    full_path = "".join((_path_accuracy_test, filename))
    data = np.load(full_path, allow_pickle=True)

    y_true, y_pred, model_info = data["labels true"], data["labels predicted"], data["model info"][()]

    pos_dash = [m.start() for m in re.finditer("_", filename)]
    model_grid = filename[pos_dash[1] + 1:pos_dash[2]]

    suf = "".join(("_", model_grid, "_accuracy_test"))

    if "compositional" in filename:
        used_minerals, used_endmembers = model_info["used minerals"], model_info["used end-members"]
        pure_only, mix_of_the_pure_ones = model_info["pure only"], model_info["pure and their mixtures"]

        plot_scatter_plots(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers,
                           pure_only=pure_only, mix_of_the_pure_ones=mix_of_the_pure_ones, suf=suf)
        plot_error_evaluation(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers, suf=suf)
        plot_error_density_plots(y_true, y_pred, used_minerals=used_minerals, used_endmembers=used_endmembers, suf=suf)

    elif "taxonomical" in filename:
        labels = np.array(list(model_info["used classes"].keys()))
        plot_confusion_matrix(y_true, y_pred, labels=labels, suf=suf)
    else:
        raise ValueError("Invalid filename.")
