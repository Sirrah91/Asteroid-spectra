from os import path
import matplotlib as mpl
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras.models import Model

from modules.NN_losses_metrics_activations import my_quantile

# from modules.utilities_data import error_estimation_bin_like, error_estimation_overall
from modules.utilities_data import load_npz, compute_metrics, used_indices

from modules.utilities import check_dir, kernel_density_estimation_2d, safe_arange, my_polyfit, denoise_array

from modules._constants import _path_figures, _path_accuracy_tests, _rnd_seed
from modules._constants import _label_true_name, _label_pred_name, _config_name, _observations_name

from modules.NN_config import quantity_names, quantity_names_short_latex

# defaults only
from modules.NN_config import conf_output_setup

mpl.use("Agg")

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


change_params(offset=0.)  # default params

plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rc("font", **{"family": "sans-serif", "sans-serif": ["Arial"]})

outdir = _path_figures
check_dir(outdir)

outdir_HMI_to_SOT = path.join(outdir, "HMI_to_SOT")
check_dir(outdir_HMI_to_SOT)


def plot_quantity_maps(y_pred: np.ndarray, y_true: np.ndarray | None = None, x_true: np.ndarray | None = None,
                       used_quantities: np.ndarray | None = None,
                       merge_rows: int = 1, merge_cols: int = 1, num_plots: int = 1,
                       rnd_seed: int | None = _rnd_seed, offset: float = 0.,
                       suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Quantity maps")
    change_params(offset)

    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    rng = np.random.default_rng(seed=rnd_seed)

    used_names = np.array(quantity_names_short_latex[used_indices(used_quantities)], dtype="<U20")
    used_names[0] = np.core.defchararray.add(used_names[0], " (arb. unit)")
    used_names[1:] = np.core.defchararray.add(used_names[1:], " (G)")

    used_columns = np.array(["SDO/HMI", "Hinode/SOT", "Predictions", r"$\text{Predictions} - \text{Hinode/SOT}$"])
    used_columns = used_columns[[x_true is not None, y_true is not None, True, y_true is not None]]

    num_quantities = len(used_names)
    num_columns = len(used_columns)

    cmap = "viridis"  # cmap of points

    def join_data(data: np.ndarray | None, indices: np.ndarray) -> np.ndarray | None:
        if data is None:
            return None

        _, ny, nx, nz = np.shape(data)

        x = np.zeros((merge_rows * ny, merge_cols * nx, num_quantities))

        for ir in range(merge_rows):
            for ic in range(merge_cols):
                data_index = np.ravel_multi_index((ir, ic), (merge_rows, merge_cols))
                data_index = indices[data_index]
                x[ir * ny: (1 + ir) * ny, ic * nx: (1 + ic) * nx] = data[data_index]

        return x

    num_plots = np.min((num_plots, len(y_pred) // (merge_rows * merge_cols)))
    data_indices = rng.choice(len(y_pred), (num_plots, merge_rows * merge_cols), replace=False)

    for ifig in range(num_plots):
        y_pred_plot = join_data(y_pred, data_indices[ifig])
        y_true_plot = join_data(y_true, data_indices[ifig])
        x_true_plot = join_data(x_true, data_indices[ifig])

        if x_true_plot is None:
            if y_true_plot is None:
                col0, col1, col2, col3 = y_pred_plot, None, None, None
                vmins = np.min(col0, axis=(0, 1))
                vmaxs = np.max(col0, axis=(0, 1))
            else:
                col0, col1, col2, col3 = y_true_plot, y_pred_plot, y_pred_plot - y_true_plot, None
                vmins = np.min((np.min(col0, axis=(0, 1)), np.min(col1, axis=(0, 1))), axis=0)
                vmaxs = np.max((np.max(col0, axis=(0, 1)), np.max(col1, axis=(0, 1))), axis=0)
        else:
            if y_true_plot is None:
                col0, col1, col2, col3 = x_true_plot, y_pred_plot, None, None
                vmins = np.min((np.min(col0, axis=(0, 1)), np.min(col1, axis=(0, 1))), axis=0)
                vmaxs = np.max((np.max(col0, axis=(0, 1)), np.max(col1, axis=(0, 1))), axis=0)
            else:
                col0, col1, col2, col3 = x_true_plot, y_true_plot, y_pred_plot, y_pred_plot - y_true_plot
                vmins = np.min((np.min(col0, axis=(0, 1)), np.min(col1, axis=(0, 1)), np.min(col2, axis=(0, 1))), axis=0)
                vmaxs = np.max((np.max(col0, axis=(0, 1)), np.max(col1, axis=(0, 1)), np.max(col2, axis=(0, 1))), axis=0)
        cols = {"col0": col0, "col1": col1, "col2": col2, "col3": col3}

        fig, ax = plt.subplots(num_quantities, num_columns, figsize=(7 * num_columns, 4 * num_quantities))
        ax = np.reshape(ax, (num_quantities, num_columns))  # force dimensions for the for cycle

        for irow in range(num_quantities):
            for icol in range(num_columns):
                if y_true_plot is not None and icol == num_columns - 1:  # diff panel with different vmin, vmax
                    sp = ax[irow, icol].imshow(cols[f"col{icol}"][:, :, irow], aspect="equal", cmap=cmap)
                else:
                    sp = ax[irow, icol].imshow(cols[f"col{icol}"][:, :, irow], aspect="equal", cmap=cmap,
                                               vmin=vmins[irow], vmax=vmaxs[irow])
                ax[irow, icol].set_xticks([])
                ax[irow, icol].set_yticks([])
                ax[irow, icol].set_yticklabels([])
                ax[irow, icol].set_xticklabels([])

                if irow == 0:
                    ax[irow, icol].set_title(used_columns[icol])

                # add common colorbars
                if icol < num_columns - int(y_true_plot is not None):
                    divider = make_axes_locatable(ax[irow, icol])
                    cax = divider.append_axes(**cbar_kwargs)
                    cbar = plt.colorbar(sp, cax=cax)
                    cbar.ax.set_ylabel(used_names[irow])

                # add colorbar to prediction - SOT column
                if y_true_plot is not None and icol == num_columns - 1:
                    divider = make_axes_locatable(ax[irow, icol])
                    cax = divider.append_axes(**cbar_kwargs)
                    cbar = plt.colorbar(sp, cax=cax)
                    cbar.ax.set_ylabel(used_names[irow])

        plt.draw()
        plt.tight_layout()

        if num_plots == 1:
            fig_name = f"quantity_map{suf}.{fig_format}"
        else:
            fig_name = f"quantity_map_{ifig}{suf}.{fig_format}"
        fig.savefig(path.join(outdir_HMI_to_SOT, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
        plt.close(fig)

    change_params(offset, reset=True)


def plot_scatter_plots(y_true: np.ndarray, y_pred: np.ndarray,
                       used_quantities: np.ndarray | None = None,
                       offset: float = 0., suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Scatter plots")
    change_params(offset)

    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    used_names = quantity_names[used_indices(used_quantities)]
    num_quantities = len(used_names)

    LW_scatter = 2.5

    error_estimation_method = "bin"  # "bin" or "overall"

    # limit = 0.25
    s = 30  # scaling parameter (marker size)

    # xticks = safe_arange(0., 100., 25., endpoint=True)
    # yticks = safe_arange(0., 100., 25., endpoint=True)

    # define the lines
    m, M = np.min((y_pred, y_true), axis=(0, 1, 2, 3)), np.max((y_pred, y_true), axis=(0, 1, 2, 3))
    x_line = np.array([np.min(m) - 0.1 * np.abs(np.min(m)), np.max(M) * 1.1])
    y_line = x_line
    y1p_line, y1m_line = y_line * 1.1, y_line * 0.9
    y2p_line, y2m_line = y_line * 1.2, y_line * 0.8
    l0, l10, l20, eb = "y-", "m-", "c-", "r"
    lab_line0, lab_line10, lab_line20 = r"0\% error", r"10\% error", r"20\% error"

    left, right = m - 0.1 * np.abs(m), M * 1.1
    bottom, top = left, right

    RMSE, R2, SAM = compute_metrics(y_true, y_pred, return_r2=True, return_sam=True, all_to_one=False)
    RMSE, R2, SAM = np.round(RMSE, 1), np.round(R2, 2), np.round(SAM, 1)

    """
    actual_errorbar = np.array([0.])  # pp

    if "bin" in error_estimation_method.lower():
        error_estimation = error_estimation_bin_like
    else:
        error_estimation = error_estimation_overall

    pred_errorbar, true_errorbar = error_estimation(y_true, y_pred, actual_error=actual_errorbar)
    pred_errorbar = np.reshape(pred_errorbar, (-1, num_quantities))
    """

    fig, ax = plt.subplots(1, num_quantities, figsize=(4.5 * num_quantities, 6))
    ax = np.reshape(ax, (-1,))  # to force iterable for the for cycle

    for i, axis in enumerate(ax):
        # lines
        lns1 = axis.plot(x_line, y_line, l0, label=lab_line0, linewidth=LW_scatter, zorder=3)
        lns2 = axis.plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW_scatter, zorder=4)
        axis.plot(x_line, y1m_line, l10, linewidth=LW_scatter, zorder=5)
        lns3 = axis.plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW_scatter, zorder=6)
        axis.plot(x_line, y2m_line, l20, linewidth=LW_scatter, zorder=7)
        # data
        axis.scatter(np.reshape(y_true, (-1, num_quantities))[:, i],
                     np.reshape(y_pred, (-1, num_quantities))[:, i],
                     c="black", s=s, zorder=2)
        # axis.errorbar(y_true[:, i], y_pred[:, i], yerr=error_pred[i], xerr=error_true[i], fmt=eb, ls="", elinewidth=0.5, zorder=1)

        axis.set_xlabel("Actual")
        axis.set_ylabel("Predicted")
        axis.tick_params(axis="both")
        axis.axis("square")
        axis.set_title(used_names[i].capitalize())
        # axis.set_xticks(xticks)
        # axis.set_yticks(yticks)
        axis.set_ylim(bottom=bottom[i], top=top[i])
        axis.set_xlim(left=left[i], right=right[i])

        axis.text(0.8, 0.15,
                   r"\["  # every line is a separate raw string...
                   r"\begin{split}"  # ...but they are all concatenated by the interpreter :-)
                   r"\mathsf{RMSE} &= " + f"{RMSE[i]:4.1f}" + r"\text{ }" + r"\\"
                   r"\mathsf{R}^2 &= " + f"{R2[i]:4.2f}" + r"\\"
                   r"\mathsf{SAM} &= " + f"{SAM[i]:4.1f}" + r"\text{ deg}"
                   r"\end{split}"
                   r"\]",
                   horizontalalignment="center", verticalalignment="center", transform=axis.transAxes)

        lns = lns1 + lns2 + lns3
        labs = [l.get_label() for l in lns]
        axis.legend(lns, labs, loc="upper left", frameon=False)

    plt.draw()
    plt.tight_layout()

    fig_name = f"scatter_plot{suf}.{big_fig_format}"
    fig.savefig(path.join(outdir_HMI_to_SOT, fig_name), format=big_fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_error_evaluation(y_true: np.ndarray, y_pred: np.ndarray,
                          used_quantities: np.ndarray | None = None,
                          offset: float = 0., suf: str = "", quiet: bool = False) -> None:
    if not quiet:
        print("Print quantiles")
    change_params(offset)

    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    used_names = quantity_names[used_indices(used_quantities)]
    num_quantities = len(used_names)
    n_colors = 9  # 10 colours in default, one reserved for "all data"; then used dashed lines

    percentile = safe_arange(0., 100., 0.5, endpoint=True)

    # define the lines
    x_line = safe_arange(-150., 150., endpoint=True)
    y_line_10 = np.ones(np.shape(x_line)) * 10.
    y_line_20 = np.ones(np.shape(x_line)) * 20.
    l10, l20 = "k--", "k--"

    one_sigma = 68.2
    sigma_c, sigma_ls = "k", "--"

    shift = 3.  # Control ranges of axes (from 0 - shift to 100 + shift)

    xticks = safe_arange(0., 100., 10., endpoint=True)
    # yticks = safe_arange(0., 100., 10., endpoint=True)

    left, right = -shift, 100. + shift
    bottom = -shift

    quantile = my_quantile(percentile=percentile, all_to_one=False)(y_true, y_pred).numpy()

    used_names = np.insert(used_names, 0, "All data")
    quantile_all = my_quantile(percentile=percentile, all_to_one=True)(y_true, y_pred).numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(percentile, quantile_all, linewidth=3, zorder=100)

    if num_quantities > n_colors:
        ax.plot(percentile, quantile[:, :n_colors], linewidth=2)
        ax.plot(percentile, quantile[:, n_colors:], "--", linewidth=2)
        ncol = 2
    else:
        ax.plot(percentile, quantile, linewidth=2)
        ncol = 1

    # constant error lines
    ax.plot(x_line, y_line_10, l10, zorder=101)
    ax.plot(x_line, y_line_20, l20, zorder=102)

    ax.axvline(one_sigma, color=sigma_c, ls=sigma_ls, zorder=103)

    ax.set_xlabel("Percentile")
    ax.set_ylabel("Absolute error")

    ax.set_xticks(xticks)
    # ax.set_yticks(yticks)
    ax.set_ylim(bottom=bottom)
    ax.set_xlim(left=left, right=right)

    ax.legend(used_names, loc="upper left", ncol=ncol)

    plt.draw()
    plt.tight_layout()

    fig_name = f"quantile_error_plot{suf}.{fig_format}"
    fig.savefig(path.join(outdir_HMI_to_SOT, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_error_density_plots(y_true: np.ndarray, y_pred: np.ndarray,
                             used_quantities: np.ndarray | None = None,
                             offset: float = 0., suf: str = "", quiet: bool = False) -> None:
    # TOHLE ZATIM MOC NEFUNGUJE

    if not quiet:
        print("Print density plots")
    change_params(offset)

    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    used_names = quantity_names[used_indices(used_quantities)]
    num_quantities = len(used_names)

    nbins = 20
    cmap = "viridis_r"  # cmap of points
    fs = SMALL_SIZE + 6

    # define the line styles
    ls_hor, ls_ver = "r--", "k--"

    fig, ax = plt.subplots(ncols=num_quantities, nrows=1, figsize=(4.5 * num_quantities, 6))
    ax = np.reshape(ax, (-1,))  # to force iterable for the for cycle

    # 2D density plots first
    for i in range(num_quantities):
        y_true_part, y_pred_part = np.reshape(y_true[..., i], (-1,)),  np.reshape(y_pred[..., i], (-1,))

        xi, yi, zi = kernel_density_estimation_2d(y_true_part, y_pred_part, nbins=nbins)
        ax[i].pcolormesh(xi, yi, zi, shading="gouraud", cmap=cmap)
        ax[i].contour(xi, yi, zi)

        ax[i].axhline(y=0, linestyle=ls_hor[1:], color=ls_hor[0])

        ax[i].set_xlabel(used_names[i], fontsize=fs + 4)
        ax[i].tick_params(axis="both", labelsize=fs)

    ax[0].set_ylabel("Error", fontsize=fs + 4)

    plt.draw()
    plt.tight_layout()

    fig_name = f"density_error_plot{suf}.{big_fig_format}"
    fig.savefig(path.join(outdir_HMI_to_SOT, fig_name), format=big_fig_format, **savefig_kwargs, **pil_kwargs)
    plt.close(fig)

    change_params(offset, reset=True)


def plot_model_history(model: Model, dt_string: str | None = None, offset: float = 0., quiet: bool = False) -> None:
    if not quiet:
        print("Model history")
    change_params(offset)

    if dt_string is None:
        fig_name = f"model_history.{fig_format}"
    else:
        fig_name = f"model_history_{dt_string}.{fig_format}"

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

    fig.savefig(path.join(outdir, fig_name), format=fig_format, **savefig_kwargs, **pil_kwargs)
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
        labels = np.char.replace(labels, "_", r"\_")
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


def result_plots(x_true: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                 used_quantities: np.ndarray | None = None,
                 density_plot: bool = False,
                 suf: str = "", quiet: bool = True) -> None:
    if used_quantities is None: used_quantities = conf_output_setup["used_quantities"]

    if "val" in suf or "train" in suf:
        merge_rows, merge_cols = 2, 3
        num_plots = 1
    else:
        merge_rows, merge_cols = 1, 1
        num_plots = 5

    plot_quantity_maps(y_pred, y_true, x_true, merge_rows=merge_rows, merge_cols=merge_cols, num_plots=num_plots,
                       used_quantities=used_quantities, suf=suf, quiet=quiet)
    plot_scatter_plots(y_true, y_pred, used_quantities=used_quantities, suf=suf, quiet=quiet)
    plot_error_evaluation(y_true, y_pred, used_quantities=used_quantities, suf=suf, quiet=quiet)
    if density_plot:
        plot_error_density_plots(y_true, y_pred, used_quantities=used_quantities, suf=suf, quiet=quiet)


def plot_acc_test_results(filename: str, subfolder: str = "") -> None:
    full_path = path.join(_path_accuracy_tests, subfolder, filename)
    data = load_npz(full_path)

    x_true, y_true, y_pred = data[_observations_name], data[_label_true_name], data[_label_pred_name]

    config = data[_config_name][()]
    used_quantities = config["output_setup"]["used_quantities"]

    suf = "_accuracy_test"

    result_plots(x_true, y_true, y_pred, used_quantities=used_quantities, density_plot=True, suf=suf)
