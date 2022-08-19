import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd
import re
from typing import List

from modules.BAR_BC_method import calc_BAR_BC, calc_composition, filter_data_mask
from modules.collect_data import resave_Tomas_OL_OPX_mixtures

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.models import Functional
from mpl_toolkits.axes_grid1 import make_axes_locatable

from modules.NN_data import *
from modules.NN_config import *
from modules.NN_losses_metrics_activations import *
from modules.CD_parameters import path_relab, path_taxonomy
from modules.NN_config_taxonomy import classes, classes2
from modules.utilities_spectra import combine_same_range_models
from modules.utilities import check_dir, get_weights_from_model, best_blk
from modules.utilities import kernel_density_estimation_1d, kernel_density_estimation_2d

mpl.use('Agg')

TEXT_SIZE = 12
SMALL_SIZE = 16
MEDIUM_SIZE = 20
BIGGER_SIZE = 24

LW = 1

plt.rc('font', size=TEXT_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)

plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
# plt.rc('font', **{'family': 'serif', 'serif': ['Palatino']})

outdir = "".join((project_dir, '/figures/'))
check_dir(outdir)

fig_format = 'pdf'


def plot_scatter_plots(y_true: np.ndarray, y_pred: np.ndarray, suf: str = '', quiet: bool = False) -> None:
    if not quiet:
        print('Scatter plots')

    y_true = y_true[:] * 100
    y_pred = y_pred[:] * 100

    # limit = 0.25
    shift = 3  # Control ranges of axes (from 0 - shift to 100 + shift)
    s = 30  # scaling parameter (marker size)
    vmin = 0  # minimum for colormap in scatter of chemical
    cmap = 'viridis_r'  # cmap of points

    titles_all = [['Fa', 'Fo'], ['Fs (OPX)', 'En (OPX)', 'Wo (OPX)'],
                  ['Fs (CPX)', 'En (CPX)', 'Wo (CPX)'], ['An', 'Ab', 'Or']]
    if not chem_without_modal:
        titles_all = [titles_all[k] for k in range(len(use_minerals)) if use_minerals_all[k]]

    titles_all = [[titles_all[k][j] for j in range(len(subtypes_all_used[k])) if subtypes_all_used[k][j]]
                  for k in range(len(subtypes))]

    # define the lines
    x_line = np.arange(-150, 151)
    y_line = x_line
    y1p_line, y1m_line = y_line + 10, y_line - 10
    y2p_line, y2m_line = y_line + 20, y_line - 20
    l10, l20 = 'm-', 'c-'

    lab_line0, lab_line10, lab_line20 = '0 pp error', '10 pp error', '20 pp error'

    RMSE = my_rmse(num_minerals)(y_true, y_pred).numpy() / 100  # is multiplied with 100 in the code
    R2 = my_r2(num_minerals)(y_true, y_pred).numpy()
    SAM = my_sam(num_minerals)(y_true, y_pred).numpy() * 180 / np.pi

    actual_errorbar = 3  # pp

    # modal first
    start, stop = 0, num_minerals

    if num_minerals > 1:
        x_tmp, y_tmp = y_true[:, start:stop], y_pred[:, start:stop]
        predicted_errorbar = RMSE[start:stop]

        lower_error = (x_tmp + actual_errorbar - np.abs(x_tmp - actual_errorbar)) / 2
        upper_error = (100 + actual_errorbar - x_tmp - np.abs(x_tmp - 100 + actual_errorbar)) / 2
        actual_errorbar_reduced = np.array(list(zip(lower_error, upper_error))).T

        lower_error = (y_tmp + predicted_errorbar - np.abs(y_tmp - predicted_errorbar)) / 2
        upper_error = (100 + predicted_errorbar - y_tmp - np.abs(y_tmp - 100 + predicted_errorbar)) / 2
        predicted_errorbar_reduced = np.array(list(zip(lower_error, upper_error))).T

        fig, ax = plt.subplots(1, num_minerals, figsize=(4.5 * num_minerals, 6))
        for i in range(num_minerals):
            # lines
            lns1 = ax[i].plot(x_line, y_line, 'k', label=lab_line0, linewidth=LW)
            lns2 = ax[i].plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW)
            ax[i].plot(x_line, y1m_line, l10, linewidth=LW)
            lns3 = ax[i].plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW)
            ax[i].plot(x_line, y2m_line, l20, linewidth=LW)
            # data
            ax[i].scatter(x_tmp[:, i], y_tmp[:, i], c='black', s=s, zorder=100)
            ax[i].errorbar(x_tmp[:, i], y_tmp[:, i], yerr=predicted_errorbar_reduced[i],
                           xerr=actual_errorbar_reduced[i], fmt='r', ls='',zorder=99)

            ax[i].set_xlabel('Actual [vol\%]')
            ax[i].tick_params(axis='both')
            ax[i].axis('square')
            ax[i].set_ylim(bottom=-shift, top=100 + shift)
            ax[i].set_xlim(left=-shift, right=100 + shift)
            ax[i].set_title(minerals[i])
            ax[i].set_xticks(np.arange(0, 100.1, 25))
            ax[i].set_yticks(np.arange(0, 100.1, 25))

            ax[i].text(0.8, 0.15,
                       r'\['  # every line is a separate raw string...
                       r'\begin{split}'  # ...but they are all concatenated by the interpreter :-)
                       r'\mathsf{RMSE} &= ' + '{:4.1f}'.format(RMSE[i]) + r'\text{ pp}' + r'\\'
                                                                          r'\mathsf{R}^2 &= ' + '{:4.2f}'.format(
                           R2[i]) + r'\\'
                                    r'\mathsf{SAM} &= ' + '{:4.1f}'.format(SAM[i]) + r'\text{ deg}'
                                                                                     r'\end{split}'
                                                                                     r'\]',
                       horizontalalignment='center', verticalalignment='center', transform=ax[i].transAxes)

            if i > 0:
                ax[i].set_yticklabels([])

            lns = lns1 + lns2 + lns3
            labs = [l.get_label() for l in lns]
            ax[i].legend(lns, labs, loc='upper left', frameon=False)

        ax[0].set_ylabel('Predicted [vol\%]')

        plt.draw()
        plt.tight_layout()

        outdir_chemical = outdir + '/chemical/'
        check_dir(outdir_chemical)

        fig_name = "".join(('scatter_plot_modal', suf, '.', fig_format))
        fig.savefig("".join((outdir_chemical, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

    # each mineral on separate plot
    for i in range(len(subtypes)):
        start, stop = stop, stop + subtypes[i]

        titles = titles_all[i]
        predicted_errorbar = RMSE[start:stop]

        # fig size tuned such that the plots are approximately of the same size
        fig, ax = plt.subplots(1, subtypes[i], figsize=(4.4 * subtypes[i] + 1.5, 6))

        if num_minerals > 1:
            # non-zero modal
            mask = y_true[:, i] > 0
            x_tmp, y_tmp = y_true[mask, start:stop], y_pred[mask, start:stop]
            c = y_true[mask, i]
        else:
            x_tmp, y_tmp = y_true[:, start:stop], y_pred[:, start:stop]
            c = 'black'

        lower_error = (x_tmp + actual_errorbar - np.abs(x_tmp - actual_errorbar)) / 2
        upper_error = (100 + actual_errorbar - x_tmp - np.abs(x_tmp - 100 + actual_errorbar)) / 2
        actual_errorbar_reduced = np.array(list(zip(lower_error, upper_error))).T

        lower_error = (y_tmp + predicted_errorbar - np.abs(y_tmp - predicted_errorbar)) / 2
        upper_error = (100 + predicted_errorbar - y_tmp - np.abs(y_tmp - 100 + predicted_errorbar)) / 2
        predicted_errorbar_reduced = np.array(list(zip(lower_error, upper_error))).T

        for j in range(subtypes[i]):
            # lines
            lns1 = ax[j].plot(x_line, y_line, 'k', label=lab_line0, linewidth=LW)
            lns2 = ax[j].plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW)
            ax[j].plot(x_line, y1m_line, l10, linewidth=LW)
            lns3 = ax[j].plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW)
            ax[j].plot(x_line, y2m_line, l20, linewidth=LW)
            # data
            sc = ax[j].scatter(x_tmp[:, j], y_tmp[:, j], c=c, cmap=cmap, vmin=vmin, vmax=100, s=s, zorder=100)
            ax[j].errorbar(x_tmp[:, j], y_tmp[:, j], yerr=predicted_errorbar_reduced[j],
                           xerr=actual_errorbar_reduced[j], fmt='r', ls='',zorder=99)

            # if only 1 mineral or if only pure samples, colorbar is not needed
            if not (num_minerals <= 1 or (use_pure_only and not use_mix_of_the_pure_ones)):
                divider = make_axes_locatable(ax[j])
                cax = divider.append_axes("right", size="5%", pad=0.1)
                cbar = plt.colorbar(sc, ax=ax[j], cax=cax)
                if j == subtypes[i] - 1:
                    cbar.ax.set_ylabel('Modal abundance [vol\%]')
                else:
                    cbar.remove()

            ax[j].set_xlabel('Actual')
            ax[j].tick_params(axis='both')
            ax[j].axis('square')
            ax[j].set_ylim(bottom=-shift, top=100 + shift)
            ax[j].set_xlim(left=-shift, right=100 + shift)
            ax[j].set_title(titles[j])
            ax[j].set_xticks(np.arange(0, 100.1, 25))
            ax[j].set_yticks(np.arange(0, 100.1, 25))

            ax[j].text(0.8, 0.15,
                       r'\['  # every line is a separate raw string...
                       r'\begin{split}'  # ...but they are all concatenated by the interpreter :-)
                       r'\mathsf{RMSE} &= ' + '{:4.1f}'.format(RMSE[start + j]) + r'\text{ pp}' + r'\\'
                                                                                  r'\mathsf{R}^2 &= ' + '{:4.2f}'.format(
                           R2[start + j]) + r'\\'
                                            r'\mathsf{SAM} &= ' + '{:4.1f}'.format(SAM[start + j]) + r'\text{ deg}'
                                                                                                     r'\end{split}'
                                                                                                     r'\]',
                       horizontalalignment='center', verticalalignment='center', transform=ax[j].transAxes)
            if j > 0:
                ax[j].set_yticklabels([])

            lns = lns1 + lns2 + lns3
            labs = [l.get_label() for l in lns]
            ax[j].legend(lns, labs, loc='upper left', frameon=False)

        ax[0].set_ylabel('Predicted')

        plt.draw()
        plt.tight_layout()

        outdir_chemical = outdir + '/chemical/'
        check_dir(outdir_chemical)

        fig_name = "".join(('scatter_plot_', minerals_used[i], suf, '.', fig_format))
        fig.savefig("".join((outdir_chemical, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)


def plot_error_evaluation(y_true: np.ndarray, y_pred: np.ndarray, quiet: bool = False) -> None:
    if not quiet:
        print('Print quantiles')

    # define the lines
    x_line = np.arange(-150, 151)
    y_line_10 = np.ones(np.shape(x_line)) * 10
    y_line_20 = np.ones(np.shape(x_line)) * 20
    l10, l20 = 'k--', 'k--'

    percentile = np.arange(0, 101, 5)
    quantile = my_quantile(num_minerals, percentile)(y_true, y_pred).numpy()

    titles_all = [['OL', 'OPX', 'CPX', 'PLG'], ['Fa', 'Fo'], ['Fs (OPX)', 'En (OPX)', 'Wo (OPX)'],
                  ['Fs (CPX)', 'En (CPX)', 'Wo (CPX)'], ['An', 'Ab', 'Or']]

    titles_all = np.array(flatten_list(titles_all))[used_indices]
    keep_inds = unique_indices()

    titles_all = titles_all[keep_inds]
    quantile = quantile[:, keep_inds]

    shift = 3  # Control ranges of axes (from 0 - shift to 100 + shift)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if num_minerals > 1:
        n_max = 9
        titles_all = np.insert(titles_all, 0, 'all data')
        quantile_all = my_quantile(num_minerals, percentile, True)(y_true, y_pred).numpy()

        ax.plot(percentile, quantile_all, linewidth=3, zorder=100)
    else:
        n_max = 10

    if np.shape(quantile)[1] > n_max:
        ax.plot(percentile, quantile[:, :n_max], linewidth=2)
        ax.plot(percentile, quantile[:, n_max:], '--', linewidth=2)
        ncol = 2
    else:
        ax.plot(percentile, quantile, linewidth=2)
        ncol = 1
    # constant error lines
    ax.plot(x_line, y_line_10, l10)
    ax.plot(x_line, y_line_20, l20)

    ax.set_xlabel('Percentile')
    ax.set_ylabel('Absolute error [pp]')

    ax.set_ylim(bottom=-shift, top=100 + shift)
    ax.set_xlim(left=-shift, right=100 + shift)
    ax.set_xticks(np.arange(0, 100.1, 10))
    ax.set_yticks(np.arange(0, 100.1, 10))

    ax.legend(titles_all, loc='upper left', ncol=ncol)

    plt.draw()
    plt.tight_layout()

    outdir_chemical = outdir + '/chemical/'
    check_dir(outdir_chemical)

    fig_name = "".join(('quantile_error_plot', '.', fig_format))
    fig.savefig("".join((outdir_chemical, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_error_density_plots(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    nbins = 200
    cmap = 'viridis_r'  # cmap of points
    fs = SMALL_SIZE + 2
    error_lim = 40

    # define the line styles
    ls_hor = 'r--'
    ls_ver = 'k--'

    titles_all = [['OL [vol\%]', 'OPX [vol\%]', 'CPX [vol\%]', 'PLG [vol\%]'],
                  ['Fa', 'Fo'], ['Fs (OPX)', 'En (OPX)', 'Wo (OPX)'],
                  ['Fs (CPX)', 'En (CPX)', 'Wo (CPX)'], ['An', 'Ab', 'Or']]

    titles_all = np.array(flatten_list(titles_all))[used_indices]
    keep_inds = unique_indices()

    titles_all = titles_all[keep_inds]

    fig, ax = plt.subplots(ncols=4, nrows=3, figsize=(22, 12))  # this was set ad hoc; 28, 16

    # 2D density plots first
    for i in range(len(keep_inds)):
        ii, jj = np.unravel_index(i, (3, 4))

        # ad hoc logic
        if i < 3:  # modal abundances
            y_true_part, y_pred_part = y_true[:, keep_inds[i]], y_pred[:, keep_inds[i]]
        # else filter out samples with w_true = 0
        elif i == 3:  # Fa
            mask = y_true[:, 0] > 0
            y_true_part, y_pred_part = y_true[mask, keep_inds[i]], y_pred[mask, keep_inds[i]]
        elif i == 4:  # Fs (OPX)
            mask = y_true[:, 1] > 0
            y_true_part, y_pred_part = y_true[mask, keep_inds[i]], y_pred[mask, keep_inds[i]]
        elif i > 4:  # CPX chemical
            mask = y_true[:, 2] > 0
            y_true_part, y_pred_part = y_true[mask, keep_inds[i]], y_pred[mask, keep_inds[i]]

        xi, yi, zi = kernel_density_estimation_2d(y_true_part, y_pred_part, nbins=nbins)
        ax[ii, jj].pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=cmap)
        ax[ii, jj].contour(xi, yi, zi.reshape(xi.shape))

        ax[ii, jj].axhline(y=0, linestyle=ls_hor[1:], color=ls_hor[0])

        if jj == 0:
            ax[ii, jj].set_ylabel('Error [pp]', fontsize=fs + 4)

        ax[ii, jj].set_xlabel(titles_all[i], fontsize=fs + 4)

        ax[ii, jj].set_xticks(np.arange(0, 100.1, 25))
        ax[ii, jj].set_xlim(left=0, right=100)

        ax[ii, jj].set_yticks(np.arange(-100, 100.1, 10))
        ax[ii, jj].set_ylim(bottom=-error_lim, top=error_lim)

        ax[ii, jj].tick_params(axis='both', labelsize=fs)

    # 1D density plots (stack modal, OL, OPX, CPX)
    ax[2, 0].set_ylabel('Density', fontsize=fs + 4)

    titles_all[:3] = np.array(['OL', 'OPX', 'CPX'])

    # modal
    ii, jj = 2, 0
    for i in range(3):
        y_true_part, y_pred_part = y_true[:, keep_inds[i]], y_pred[:, keep_inds[i]]
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

        ax[ii, jj].set_xticks(np.arange(-100, 100, 10))
        ax[ii, jj].set_xlim(left=-error_lim, right=error_lim)
        ax[ii, jj].tick_params(axis='both', labelsize=fs)
        ax[ii, jj].set_xlabel('Error [pp]', fontsize=fs + 4)
        ax[ii, jj].legend(loc='upper right', fontsize=SMALL_SIZE - 4)

        if jj in [0, 2]:
            ticksy = np.arange(0, 0.1005, 0.02)
        elif jj == 1:
            ticksy = np.arange(0, 0.0805, 0.02)
        else:
            ticksy = np.arange(0, 0.0605, 0.02)
        ax[ii, jj].set_yticks(ticksy)

    plt.draw()
    plt.tight_layout()

    outdir_chemical = outdir + '/chemical/'
    check_dir(outdir_chemical)

    fig_name = "".join(('density_error_plot', '.', fig_format))
    fig.savefig("".join((outdir_chemical, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_error_range_plots(file_names: List[str], error_type: str = 'RMSE_all',
                           applied_function=np.mean, line_plot_or_matrix: str = 'both') -> None:

    from modules.NN_config_range_test import constant_range_or_spacing

    if constant_range_or_spacing == "spacing":
        from modules.NN_config_range_test import start, stop, step
    elif constant_range_or_spacing == 'window':
        from modules.NN_config_range_test import start, stop, _spacing, window_size, window_spacing

    # load data
    def load_data_for_plot(file_names, error_type):
        for i, file_name in enumerate(file_names):
            filename = project_dir + '/range_test_data/' + file_name
            data = pd.read_csv(filename, sep='\t')  # to read the file

            # select wanted RMSE
            indices = data['index'].to_numpy()
            ranges_all = data['range'].to_numpy()
            spacing_all = data['spacing'].to_numpy()
            rmse_all = data[error_type].to_numpy()

            # remove NaNs
            inds_not_nan = np.argwhere(np.isfinite(rmse_all))
            indices = indices[inds_not_nan]
            ranges_all = ranges_all[inds_not_nan]
            spacing_all = spacing_all[inds_not_nan]
            rmse_all = rmse_all[inds_not_nan]

            # constant range => slightly different spacing for each dataset
            if constant_range_or_spacing == 'range':
                spacing, rmse = combine_same_range_models(indices, spacing_all, rmse_all, applied_function)

                spacing = np.reshape(spacing, (1, len(spacing)))
                rmse = np.reshape(rmse, (1, len(rmse)))

                if i == 0:
                    x_axis = spacing
                    rmse_all_data = rmse
                else:
                    x_axis = np.concatenate((x_axis, spacing), axis=0)
                    rmse_all_data = np.concatenate((rmse_all_data, rmse), axis=0)

            else:  # spacing or window do the same
                x_axis, rmse = combine_same_range_models(indices, ranges_all, rmse_all, applied_function)
                rmse = np.reshape(rmse, (1, len(rmse)))

                if i == 0:
                    rmse_all_data = rmse
                else:
                    rmse_all_data = np.concatenate((rmse_all_data, rmse), axis=0)

        if constant_range_or_spacing == 'spacing':
            rmse_all_data = np.mean(rmse_all_data, axis=0)

        return x_axis, rmse_all_data

    def prepare_data(ranges_or_spacings, error):
        if constant_range_or_spacing == 'spacing':
            wvl = np.arange(start, stop + step / 2, step)

            # line plot
            x = [[int(ranges_or_spacings[i].split()[-1][:-1]) for i in range(len(ranges_or_spacings)) if
                  int(ranges_or_spacings[i].split()[0][1:-1]) == wvl[j]] for j in range(len(wvl) - 1)]
            y = [[error[i] for i in range(len(error)) if int(ranges_or_spacings[i].split()[0][1:-1]) == wvl[j]]
                 for j in range(len(wvl) - 1)]

            # mat plot
            res_mat = np.empty((len(wvl), len(wvl)))
            res_mat.fill(np.nan)

            for j in range(len(wvl) - 1):
                tmp = np.array([error[i] for i in range(len(error))
                                if int(ranges_or_spacings[i].split()[0][1:-1]) == wvl[j]])
                res_mat[1 + j:1 + j + np.size(y), j] = tmp

            return x, y, res_mat
        elif constant_range_or_spacing == 'range':
            return ranges_or_spacings, error, 0
        else:
            x = [str(window_start) + '\u2013' + str(window_start + window_size)
                 for window_start in np.arange(start, stop, _spacing) if window_start + window_size <= stop]
            return x, error, 0

    def line_plot(ax, x, y):
        cm = plt.get_cmap('gist_rainbow')
        if constant_range_or_spacing == 'range':
            num_colors = int(np.ceil(len(y) / 2))  # SWIR full lines; no SWIR dashed lines
        else:
            num_colors = len(y)
        ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])

        if constant_range_or_spacing == "spacing":
            wvl = np.arange(start, stop + step / 2, step).astype(np.int)

            for j in range(num_colors):
                ax.plot(x[j], y[j], '--o', label='From ' + str(wvl[j]) + ' nm')
            ax.set_xticks(wvl[1:])
            ax.set_xlabel('To wavelength [nm]')

            ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0,
                      ncol=best_blk(len(y))[1])
        elif constant_range_or_spacing == 'range':
            # extract ranges from file_names
            pos_under = [[m.start() for m in re.finditer('_', file)] for file in file_names]
            labels = [file_names[i][pos_under[i][1] + 1:pos_under[i][2]] + '\u2013' +
                      file_names[i][pos_under[i][2] + 1:-4]
                      for i in range(len(file_names))]
            styles = ['-or', '--or', '-ob', '--ob']
            for j in range(len(y)):
                ax.plot(x[j], y[j], styles[j], label=labels[j] + ' nm')

            ax.set_xlabel('Spacing [nm]')
            ax.legend(ncol=2)
        else:
            ax.plot(y.ravel(), '--o', label='Window spacing ' + str(int(window_spacing)))

            ax.set_xticks(np.arange(0, len(x)))
            ax.set_xticklabels(x, ha='center', fontsize=10)
            ax.set_xlabel('Window range [nm]')
            ax.legend()

        ax.set_ylabel(error_label)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=False, left=True, right=False)

    def matrix_plot(ax_matrix, res_mat):
        if 'RMSE' in error_type:
            vmin = np.nanmin(res_mat)
            vmax = np.nanmin((np.nanmax(res_mat), np.nanmin(res_mat) + 10))
            cmap = 'jet_r'
        else:
            vmin = np.nanmax((np.nanmin(res_mat), np.nanmax(res_mat) - 20))
            vmax = np.nanmax(res_mat)
            cmap = 'jet'

        wvl = np.arange(start, stop + step / 2, step).astype(np.int)
        im = ax_matrix.imshow(res_mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        ax_matrix.set_xticks(np.arange(0, len(wvl)))
        ax_matrix.set_yticks(np.arange(0, len(wvl)))

        ax_matrix.set_xticklabels(wvl, rotation=90, ha='center')
        ax_matrix.set_yticklabels(wvl)

        ax_matrix.set_xlabel('From wavelength [nm]')
        ax_matrix.set_ylabel('To wavelength [nm]')
        ax_matrix.xaxis.set_label_position('bottom')
        ax_matrix.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                              bottom=True, top=False, left=True, right=False)

        divider = make_axes_locatable(ax_matrix)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(error_label)

    x, error = load_data_for_plot(file_names, error_type)
    x, y, mat = prepare_data(x, error)

    error_type = error_type.replace('_nooutliers', '')
    tmp = error_type.split('_')

    if 'RMSE' in error_type:
        error_label = tmp[0] + ' (pp; ' + " ".join(tmp[1:]) + ')'
    else:
        error_label = tmp[0] + ' (' + tmp[1] + ' ' + tmp[2] + '; ' + " ".join(tmp[3:]) + ')'

    # no matrix plot for constant range
    if constant_range_or_spacing != "spacing":
        line_plot_or_matrix = 'line'

    if constant_range_or_spacing == 'window':
        fig_name_suffix = "".join((constant_range_or_spacing, '_', str(int(window_size)), '_',
                                   error_type, '.', fig_format))
    else:
        fig_name_suffix = "".join((constant_range_or_spacing, '_', error_type, '.', fig_format))

    if 'line' in line_plot_or_matrix:
        if constant_range_or_spacing == 'window':  # needed bigger window
            fig, ax = plt.subplots(1, 1, figsize=(18, 6))
        else:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        line_plot(ax, x, y)
        fig_name = "".join(('range_line_', fig_name_suffix))

    elif 'matrix' in line_plot_or_matrix:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        matrix_plot(ax, mat)
        fig_name = "".join(('range_matrix_', fig_name_suffix))

    elif 'both' in line_plot_or_matrix:
        fig, ax = plt.subplots(1, 2, figsize=(25, 10), gridspec_kw={'width_ratios': [2.3, 1]})
        line_plot(ax[0], x, y)
        matrix_plot(ax[1], mat)
        fig_name = "".join(('range_plot_', fig_name_suffix))
    else:
        raise ValueError('There is no mineral in use_minerals.')

    plt.draw()
    plt.tight_layout()

    outdir_range_tests = outdir + '/range_test/'
    check_dir(outdir_range_tests)

    fig.savefig("".join((outdir_range_tests, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    # plt.close(fig)


def plot_error_spacing(file_name: str, error_type: str = 'RMSE_all', applied_function=np.mean,
                       line_plot_or_matrix: str = 'both') -> None:

    from modules.NN_config_range_test import start, stop, step

    # load data
    def load_data_for_plot(file_name, error_type):
        filename = project_dir + '/range_test_data/' + file_name
        data = pd.read_csv(filename, sep='\t')  # to read the file

        # select wanted metric
        indices = data['index'].to_numpy()
        ranges_all = data['range'].to_numpy()
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

    def prepare_data(x_axis, error):
        wvl = np.arange(start, stop + step / 2, step)

        # line plot
        x = [[int(x_axis[i].split()[-1][:-1]) for i in range(len(x_axis)) if int(x_axis[i].split()[0][1:-1]) == wvl[j]]
             for j in range(len(wvl) - 1)]
        y = [[error[i] for i in range(len(error)) if int(x_axis[i].split()[0][1:-1]) == wvl[j]]
             for j in range(len(wvl) - 1)]

        # mat plot
        res_mat = np.empty((len(wvl), len(wvl)))
        res_mat.fill(np.nan)

        for j in range(len(wvl) - 1):
            tmp = np.array([error[i] for i in range(len(error)) if int(x_axis[i].split()[0][1:-1]) == wvl[j]])
            res_mat[1 + j:1 + j + np.size(y), j] = tmp

        return x, y, res_mat

    def line_plot(ax, x, y):
        cm = plt.get_cmap('gist_rainbow')
        num_colors = len(y)
        ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])

        wvl = np.arange(start, stop + step / 2, step).astype(np.int)

        for j in range(num_colors):
            ax.plot(x[j], y[j], '--o', label='From ' + str(wvl[j]) + ' nm')
        ax.set_xticks(wvl[1:])
        ax.set_xlabel('To wavelength [nm]')

        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0,
                  ncol=best_blk(num_colors)[1])

        ax.set_ylabel(error_label)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=False, left=True, right=False)

    def matrix_plot(ax_matrix, res_mat):
        if 'RMSE' in error_type:
            vmin = np.nanmin(res_mat)
            vmax = np.nanmin((np.nanmax(res_mat), np.nanmin(res_mat) + 10))
            cmap = 'jet_r'
        else:
            vmin = np.nanmax((np.nanmin(res_mat), np.nanmax(res_mat) - 20))
            vmax = np.nanmax(res_mat)
            cmap = 'jet'

        wvl = np.arange(start, stop + step / 2, step).astype(np.int)
        im = ax_matrix.imshow(res_mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

        ax_matrix.set_xticks(np.arange(0, len(wvl)))
        ax_matrix.set_yticks(np.arange(0, len(wvl)))

        ax_matrix.set_xticklabels(wvl, rotation=90, ha='center')
        ax_matrix.set_yticklabels(wvl)

        ax_matrix.set_xlabel('From wavelength [nm]')
        ax_matrix.set_ylabel('To wavelength [nm]')
        ax_matrix.xaxis.set_label_position('bottom')
        ax_matrix.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                              bottom=True, top=False, left=True, right=False)

        divider = make_axes_locatable(ax_matrix)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.set_ylabel(error_label)

    x, error = load_data_for_plot(file_name, error_type)
    x, y, mat = prepare_data(x, error)

    error_type = error_type.replace('_nooutliers', '')
    tmp = error_type.split('_')

    if 'RMSE' in error_type:
        error_label = tmp[0] + ' [pp; ' + " ".join(tmp[1:]) + ']'
    else:
        error_label = tmp[0] + ' [' + tmp[1] + ' ' + tmp[2] + '; ' + " ".join(tmp[3:]) + ']'

    fig_name_suffix = "".join(('spacing', '_', error_type, '.', fig_format))

    if 'line' in line_plot_or_matrix:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        line_plot(ax, x, y)
        fig_name = "".join(('range_line_', fig_name_suffix))

    elif 'matrix' in line_plot_or_matrix:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        matrix_plot(ax, mat)
        fig_name = "".join(('range_matrix_', fig_name_suffix))

    elif 'both' in line_plot_or_matrix:
        fig, ax = plt.subplots(1, 2, figsize=(25, 10), gridspec_kw={'width_ratios': [2.3, 1]})
        line_plot(ax[0], x, y)
        matrix_plot(ax[1], mat)
        fig_name = "".join(('range_plot_', fig_name_suffix))
    else:
        raise ValueError('"line_plot_or_matrix" must be one of "both", "line", "matrix".')

    plt.draw()
    plt.tight_layout()

    outdir_range_tests = outdir + '/range_test/'
    check_dir(outdir_range_tests)

    fig.savefig("".join((outdir_range_tests, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    # plt.close(fig)


def plot_error_range(file_names: List[str], error_type: str = 'RMSE_all', applied_function=np.mean) -> None:

    # load data
    def load_data_for_plot(file_names, error_type):
        for i, file_name in enumerate(file_names):
            filename = project_dir + '/range_test_data/' + file_name
            data = pd.read_csv(filename, sep='\t')  # to read the file

            # select wanted metrics
            indices = data['index'].to_numpy()
            spacing_all = data['spacing'].to_numpy()
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

            spacing = np.reshape(spacing, (1, len(spacing)))
            error = np.reshape(error, (1, len(error)))

            if i == 0:
                x_axis = spacing
                error_all_data = error
            else:
                x_axis = np.concatenate((x_axis, spacing), axis=0)
                error_all_data = np.concatenate((error_all_data, error), axis=0)

        return x_axis, error_all_data

    def line_plot(ax, x, y):
        cm = plt.get_cmap('gist_rainbow')
        num_colors = int(np.ceil(len(y) / 2))  # SWIR full lines; no SWIR dashed lines
        ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])

        # extract ranges from file_names
        pos_under = [[m.start() for m in re.finditer('_', file)] for file in file_names]
        labels = [file_names[i][pos_under[i][1] + 1:pos_under[i][2]] + '\u2013' +
                  file_names[i][pos_under[i][2] + 1:-4]
                  for i in range(len(file_names))]
        styles = ['-or', '--or', '-ob', '--ob']
        for j in range(len(y)):
            ax.plot(x[j], y[j], styles[j], label=labels[j] + ' nm')

        ax.set_xlabel('Spacing [nm]')
        ax.legend(ncol=2)

        ax.set_ylabel(error_label)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=False, left=True, right=False)

    x, y = load_data_for_plot(file_names, error_type)

    error_type = error_type.replace('_nooutliers', '')
    tmp = error_type.split('_')

    if 'RMSE' in error_type:
        error_label = tmp[0] + ' [pp; ' + " ".join(tmp[1:]) + ']'
    else:
        error_label = tmp[0] + ' [' + tmp[1] + ' ' + tmp[2] + '; ' + " ".join(tmp[3:]) + ']'

    fig_name_suffix = "".join(('range', '_', error_type, '.', fig_format))

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    line_plot(ax, x, y)
    fig_name = "".join(('range_line_', fig_name_suffix))

    plt.draw()
    plt.tight_layout()

    outdir_range_tests = outdir + '/range_test/'
    check_dir(outdir_range_tests)

    fig.savefig("".join((outdir_range_tests, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    # plt.close(fig)


def plot_error_window(file_name: str, error_type: str = 'RMSE', applied_function=np.mean) -> None:

    from modules.NN_config_range_test import start, stop, _spacing, window_size

    # load data
    def load_data_for_plot(file_name):
        filename = project_dir + '/range_test_data/' + file_name
        data = pd.read_csv(filename, sep='\t')  # to read the file

        # select metrices
        indices = data['index'].to_numpy()
        ranges_all = data['range'].to_numpy()

        if 'RMSE' in error_type:
            inds = np.arange(3, 12)
        elif 'within_5' in error_type:
            inds = np.arange(21, 30)
        elif 'within_10' in error_type:
            inds = np.arange(30, 39)
        elif 'within_15' in error_type:
            inds = np.arange(39, 48)
        elif 'within_20' in error_type:
            inds = np.arange(48, 57)
        else:
            raise ValueError('Invalid "error_type.')

        error_all = data.to_numpy()[:, inds].astype(np.float)
        labels = np.array(data.keys()[inds])

        x_axis = [str(window_start) + '\u2013' + str(window_start + window_size)
                  for window_start in np.arange(start, stop, _spacing) if window_start + window_size <= stop]

        # this function must be applied column by column
        for i in range(np.shape(error_all)[1]):
            _, error = combine_same_range_models(indices, ranges_all, error_all[:, i], applied_function)
            error = np.reshape(error, (len(error), 1))

            if i == 0:
                error_all_data = error
            else:
                error_all_data = np.concatenate((error_all_data, error), axis=1)

        return x_axis, error_all_data, labels

    def line_plot(ax, x, y):
        cm = plt.get_cmap('gist_rainbow')
        num_colors = np.shape(y)[1]
        ax.set_prop_cycle(color=[cm(1. * i / num_colors) for i in range(num_colors)])

        ax.plot(y, '--o')

        ax.set_xticks(np.arange(0, len(x)))
        ax.set_xticklabels(x, ha='center', fontsize=10)
        ax.set_xlabel('Window range [nm]')

        labels = np.array(['ALL', 'OL', 'OPX', 'CPX', 'Fa', 'Fs OPX', 'Fs CPX', 'En CPX', 'Wo CPX'])

        ax.legend(labels, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0,
                  ncol=best_blk(num_colors)[1])

        ax.set_ylabel(error_label)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False,
                       bottom=True, top=False, left=True, right=False)

    x, y, labels = load_data_for_plot(file_name)

    error_type = error_type.replace('_nooutliers', '')
    tmp = error_type.split('_')

    if 'RMSE' in error_type:
        error_label = tmp[0] + ' [pp]'
    else:
        error_label = tmp[0] + ' [' + tmp[1] + ' ' + tmp[2] + ']'

    fig_name_suffix = "".join(('window', '_', str(int(window_size)), '_', error_type, '_all.', fig_format))

    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    line_plot(ax, x, y)
    fig_name = "".join(('range_line_', fig_name_suffix))

    plt.draw()
    plt.tight_layout()

    outdir_range_tests = outdir + '/range_test/'
    check_dir(outdir_range_tests)

    fig.savefig("".join((outdir_range_tests, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    # plt.close(fig)


def plot_ast_PC1_PC2(y_pred: np.ndarray) -> None:
    print('Plot PC1 vs PC2 for asteroids')

    filename = project_dir + '/Datasets/taxonomy/asteroid_spectra-denoised-norm-meta.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = data[:, 1].astype(np.str)

    inds_S = np.array(['S' in ast_type for ast_type in types])
    inds_Q = np.array(['Q' in ast_type for ast_type in types])
    inds_A = np.array(['A' in ast_type for ast_type in types])
    inds_V = np.array(['V' in ast_type for ast_type in types])

    inds = inds_S + inds_Q + inds_V + inds_A

    # ignore Sq: since its spectrum is very weird
    inds[types == 'Sq:'] = False

    unique, counts = np.unique(types[inds], return_counts=True)

    PCA = data[inds, 3:5].astype(np.float64)
    predictions = y_pred[inds] * 100

    cmap = 'viridis_r'  # cmap of points
    s = 30  # scaling parameter (marker size)
    vmin = 0  # minimum for colormap in scatter of chemical

    labels = np.core.defchararray.add('$', types[inds])
    labels = np.core.defchararray.add(labels, '$')

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Why I cannot set marker=labels and do it all as for vectors??
    for i in range(len(PCA)):
        if len(labels[i]) == 3:
            fact = 1
        if len(labels[i]) == 4:
            fact = 2.5
        if len(labels[i]) == 5:
            fact = 5.5

        c = predictions[i, 0]
        sp = ax.scatter(PCA[i, 1], PCA[i, 0], c=c, cmap=cmap, vmin=vmin, vmax=100, s=s * fact, marker=labels[i])

    ax.set_xlabel("PC2'")
    ax.set_ylabel("PC1'")

    x0, x1 = np.array([0.5, -0.5]), np.array(([0.0, 0.8]))

    plt.annotate("", xy=x1, xytext=x0, zorder=-1,
                 arrowprops=dict(arrowstyle="->", color='r', facecolor='black', lw=3))

    plt.text(x=0.1, y=-0.2, s="Space weathering", rotation=-51, c='r', fontsize=SMALL_SIZE)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(sp, ax=ax, cax=cax)
    cbar.ax.set_ylabel('Olivine abundance [vol\%]')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.draw()

    outdir_chemical = outdir + '/chemical/'
    check_dir(outdir_chemical)

    fig_name = "".join(('PCA_plot.', fig_format))
    fig.savefig("".join((outdir_chemical, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_conv_kernels(model: Functional = None) -> None:
    if model is None:
        from keras.models import load_model

        custom_objects = {loss_name: loss, output_activation_name: output_activation,
                          mse_name: mse, rmse_name: rmse, quantile_name: quantile, mae_name: mae, Lp_norm_name: Lp_norm,
                          r2_name: r2, sam_name: sam}
        model_name = '20220330113805_CNN.h5'
        subfolder_model = 'chemical'

        model = load_model("".join((project_dir, '/Models/', subfolder_model, '/', model_name)),
                           custom_objects=custom_objects)

    weights = np.squeeze(get_weights_from_model(model)['Conv_0'])
    x = np.arange(0, 5 * len(weights), 5)  # kernel width

    # sort weights from higher amplitude to lowest
    sorted_inds = np.argsort(np.max(np.abs(weights), 0))[::-1]

    blk = best_blk(np.shape(weights)[1])

    fig, ax = plt.subplots(blk[0], blk[1], figsize=(4 * blk[0], 4 * blk[1]))

    c = 0
    for row in range(blk[0]):
        for column in range(blk[1]):
            if c < np.shape(weights)[1]:
                ax[row, column].plot(x, weights[:, sorted_inds[c]])
                c += 1


def plot_Fa_vs_Fs_ast_only() -> None:
    print('Plot Fa vs Fs')

    limx1, limx2 = 15, 35
    limy1, limy2 = 10, 30

    shift = 1  # Control ranges of axes
    s = 50  # scaling parameter
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # definition of boxes from (for some reasons should be used just once)
    # https://www.researchgate.net/figure/Mineral-compositional-data-for-the-Morokweng-meteoriteDiagram-shows-ferrosilite-Fs_fig3_7093505
    H_rect = patches.Rectangle((16.2, 14.5), 3.8, 3.5, linewidth=1.5, edgecolor='r', facecolor='none')
    L_rect = patches.Rectangle((22.0, 19.0), 4.0, 3.0, linewidth=1.5, edgecolor='g', facecolor='none')
    LL_rect = patches.Rectangle((26.0, 22.0), 6.0, 4.2, linewidth=1.5, edgecolor='b', facecolor='none')

    ax.set_xlabel('Fa')
    ax.set_ylabel('Fs (OPX)')
    ax.tick_params(axis='both')
    ax.axis('square')
    ax.set_xticks(np.arange(limx1, limx2 + 0.1, 5))
    ax.set_yticks(np.arange(limy1, limy2 + 0.1, 5))

    ax.set_ylim(bottom=limy1 - shift, top=limy2 + shift)
    ax.set_xlim(left=limx1 - shift, right=limx2 + shift)

    # add the patches
    ax.add_patch(H_rect)
    ax.add_patch(L_rect)
    ax.add_patch(LL_rect)

    # artificial data to get labels for legend
    ax.plot(120, 120, 'rs', label='H', markerfacecolor='none')
    ax.plot(120, 120, 'gs', label='L', markerfacecolor='none')
    ax.plot(120, 120, 'bs', label='LL', markerfacecolor='none')

    ax.legend(loc='upper left')

    Fa_S, Fs_S, sigma_fas, sigma_fss = [20.7], [18.3], 4.5, 4.9
    Fa_Sq, Fs_Sq, sigma_fasq, sigma_fssq = [23.2], [21.0], 4.6, 4.1
    Fa_Sr, Fs_Sr, sigma_fasr, sigma_fssr, = [18.3], [19.2], 5.7, 5.4
    Fa_Sw, Fs_Sw, sigma_fasw, sigma_fssw = [21.3], [14.7], 4.1, 4.2

    Fa_Q, Fs_Q, sigma_faq, sigma_fsq = [26.2], [23.8], 5.4, 5.1

    ax.scatter(Fa_S, Fs_S, marker='$S$', c='k', s=s * 2.5, zorder=100)
    ax.errorbar(Fa_S, Fs_S, xerr=sigma_fas, yerr=sigma_fss, c='c', ls='None')
    ax.scatter(Fa_Sq, Fs_Sq, marker='$Sq$', c='k', s=s * 5, zorder=100)
    ax.errorbar(Fa_Sq, Fs_Sq, xerr=sigma_fasq, yerr=sigma_fssq, c='c', ls='None')
    ax.scatter(Fa_Sr, Fs_Sr, marker='$Sr$', c='k', s=s * 5, zorder=100)
    ax.errorbar(Fa_Sr, Fs_Sr, xerr=sigma_fasr, yerr=sigma_fssr, c='c', ls='None')
    ax.scatter(Fa_Sw, Fs_Sw, marker='$Sw$', c='k', s=s * 6, zorder=100)
    ax.errorbar(Fa_Sw, Fs_Sw, xerr=sigma_fasw, yerr=sigma_fssw, c='c', ls='None')

    ax.scatter(Fa_Q, Fs_Q, marker='$Q$', c='k', s=s * 2.5, zorder=100)
    ax.errorbar(Fa_Q, Fs_Q, xerr=sigma_faq, yerr=sigma_fsq, c='c', ls='None')

    plt.draw()
    plt.tight_layout()

    outdir_chemical = outdir + '/chemical/'
    check_dir(outdir_chemical)

    fig_name = "".join(('Fa_vs_Fs.', fig_format))
    fig.savefig("".join((outdir_chemical, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_scatter_NN_BC():
    from modules.NN_evaluate import evaluate_test_data
    wvl = np.arange(450, 2451, 5)

    filename_train_data = 'combined-denoised-norm.dat'
    x_train, y_train = load_data(filename_train_data, clean_dataset=True)
    x_train, y_train, x_val, y_val, x_test, y_true = split_data_proportional(x_train, y_train)
    model_names = ['20220330113805_CNN.h5']  # [14.4, 5.7, 5.7, 10.7]
    y_pred, accuracy = evaluate_test_data(model_names, x_test, y_true)

    # to get the test types
    filename = project_dir + '/Datasets/combined-denoised-norm-clean-meta.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = data[:, 7]
    types = np.reshape(types, (len(types), 1))
    filename_train_data = 'combined-denoised-norm-clean.dat'
    x_tmp, y_tmp = load_data(filename_train_data)
    _, _, _, _, meta, _ = split_data_proportional(
        np.concatenate((x_tmp, types), axis=1), y_tmp)

    types = meta[:, -1].astype(np.str)

    # jenom ty, co obrasuhuji OL a OPX (jen mixtures 12 a 14)
    binary = (y_true[:, :num_minerals] > 0).astype(int)
    base = np.array([8, 4, 2, 1])[use_minerals]
    mixtures = np.sum(binary * base, axis=1)

    mask1 = np.logical_or(mixtures == 12, mixtures == 14)
    # only those with CPX < 10
    mask2 = y_true[:, 2] < 0.1

    mask = np.logical_and(mask1, mask2)

    x_true = x_test[mask]
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    types = types[mask]

    inds_HED = np.array(['HED' in type for type in types])
    types[inds_HED] = 'V'
    types[~inds_HED] = 'S'

    BAR, BIC, BIIC = calc_BAR_BC(wvl, x_true)

    # remove nans
    mask = np.logical_and(np.logical_and(BAR > 0, BIC > 0), BIIC > 0)
    BAR, BIC, BIIC = BAR[mask], BIC[mask], BIIC[mask]
    OL_true = y_true[mask, 0] * 100
    OL_pred = y_pred[mask, 0] * 100
    Fs_true = y_true[mask, 5] * 100
    Fs_pred = y_pred[mask, 5] * 100
    types = types[mask]

    OL_fraction, Fs, Wo = calc_composition(BAR, BIC, BIIC, types, method=0)
    _, Fs_v2, _ = calc_composition(BAR, BIC, BIIC, types, method=1)

    # filter the data
    mask = filter_data_mask(OL_fraction, Fs, Wo)
    OL_fraction = OL_fraction[mask]
    Fs, Wo = Fs[mask], Wo[mask]
    OL_true = OL_true[mask]
    OL_pred = OL_pred[mask]
    Fs_true = Fs_true[mask]
    Fs_pred = Fs_pred[mask]
    Fs_v2 = Fs_v2[mask]

    LW = 1
    s = 30  # scaling parameter (marker size)
    x_line = np.arange(-150, 151)
    y_line = x_line
    y1p_line, y1m_line = y_line + 10, y_line - 10
    y2p_line, y2m_line = y_line + 20, y_line - 20
    l10, l20 = 'm-', 'c-'
    lab_line0, lab_line10, lab_line20 = '0 pp error', '10 pp error', '20 pp error'

    shift = 3  # Control ranges of axes (from 0 - shift to 100 + shift)
    outdir = "".join((project_dir, '/figures/'))
    check_dir(outdir)

    fig, ax = plt.subplots(1, 2, figsize=(4.7 * 2 + 1.5, 6))
    # line first in legend
    for i in range(2):
        ax[i].plot(x_line, y_line, 'k', label=lab_line0, linewidth=LW)
        ax[i].plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW)
        ax[i].plot(x_line, y1m_line, l10, linewidth=LW)
        ax[i].plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW)
        ax[i].plot(x_line, y2m_line, l20, linewidth=LW)

    ax[0].scatter(OL_true, OL_pred, s=s, label='NN pred.', zorder=100)
    ax[0].scatter(OL_true, OL_fraction, s=s, label='BAR', zorder=99)

    ax[1].scatter(Fs_true, Fs_pred, s=s, label='NN pred.', zorder=100)
    ax[1].scatter(Fs_true, Fs_v2, s=s, label='BIC', zorder=99)
    ax[1].scatter(Fs_true, Fs, s=s, label='BIIC', zorder=98)

    ax[0].set_title('olivine')
    ax[1].set_title('Fs (OPX)')

    ax[0].set_xlabel('Actual [vol\%]')
    ax[0].set_ylabel('Modelled [vol\%]')

    ax[1].set_xlabel('Actual')
    ax[1].set_ylabel('Modelled')

    for i in range(2):
        ax[i].tick_params(axis='both')
        ax[i].axis('square')
        ax[i].set_ylim(bottom=-shift, top=100 + shift)
        ax[i].set_xlim(left=-shift, right=100 + shift)

        ax[i].set_xticks(np.arange(0, 100.1, 25))
        ax[i].set_yticks(np.arange(0, 100.1, 25))

        """
        if i > 0:
            ax[i].set_yticklabels([])
        """

        ax[i].legend(loc='upper left', frameon=False)

    plt.draw()
    plt.tight_layout()

    outdir_chemical = outdir + '/chemical/'
    check_dir(outdir_chemical)

    fig_name = "".join(('scatter_plot_NN_BAR_BC_met_mix.', fig_format))
    fig.savefig("".join((outdir_chemical, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_PCA_BAR():
    from modules.NN_evaluate import evaluate

    model_names = ['20220330113805_CNN.h5']  # [14.4, 5.7, 5.7, 10.7]
    wvl = np.arange(450, 2451, 5)

    filename_data = 'asteroid_spectra-denoised-norm-nolabel.dat'
    data_file = "".join((project_dir, '/Datasets/', filename_data))
    data = np.loadtxt(data_file, delimiter='\t')

    y__pred = evaluate(model_names, filename_data)

    filename = project_dir + '/Datasets/taxonomy/asteroid_spectra-denoised-norm-meta.dat'
    meta = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = meta[:, 1].astype(np.str)

    inds_S = np.array(['S' in ast_type for ast_type in types])
    inds_Q = np.array(['Q' in ast_type for ast_type in types])
    inds_A = np.array(['A' in ast_type for ast_type in types])
    inds_V = np.array(['V' in ast_type for ast_type in types])

    inds = inds_S + inds_Q + inds_V + inds_A

    # ignore Sq: since its spectrum is very weird
    inds[types == 'Sq:'] = False

    types = types[inds]

    x_data = data[inds]
    y_pred = y__pred[inds]
    PCA = meta[inds, 3:5].astype(np.float64)

    labels = np.core.defchararray.add('$', types)
    labels = np.core.defchararray.add(labels, '$')

    BAR, BIC, BIIC = calc_BAR_BC(wvl, x_data)

    # remove nans
    mask = np.logical_and(np.logical_and(BAR > 0, BIC > 0), BIIC > 0)
    BAR, BIC, BIIC = BAR[mask], BIC[mask], BIIC[mask]
    OL_pred = y_pred[mask, 0] * 100
    Fs_pred = y_pred[mask, 5] * 100
    labels = labels[mask]
    types = types[mask]
    PCA = PCA[mask]

    OL_fraction, Fs, Wo = calc_composition(BAR, BIC, BIIC, types, method=0)
    _, Fs_v2, _ = calc_composition(BAR, BIC, BIIC, types, method=1)

    # filter the data
    mask = filter_data_mask(OL_fraction, Fs, Wo)
    OL_fraction = OL_fraction[mask]
    Fs, Wo = Fs[mask], Wo[mask]
    OL_pred = OL_pred[mask]
    Fs_pred = Fs_pred[mask]
    labels = labels[mask]
    types = types[mask]
    PCA = PCA[mask]

    unique, counts = np.unique(types, return_counts=True)

    cmap = 'viridis_r'  # cmap of points
    s = 30  # scaling parameter (marker size)
    vmin = 0  # minimum for colormap in scatter of chemical

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Why I cannot set marker=labels and do it all as for vectors??
    for i in range(len(PCA)):
        if len(labels[i]) == 3:
            fact = 1
        if len(labels[i]) == 4:
            fact = 2.5
        if len(labels[i]) == 5:
            fact = 5.5

        c = OL_fraction[i]
        sp = ax.scatter(PCA[i, 1], PCA[i, 0], c=c, cmap=cmap, vmin=vmin, vmax=100, s=s * fact, marker=labels[i])

    ax.set_xlabel("PC2'")
    ax.set_ylabel("PC1'")

    x0, x1 = np.array([0.5, -0.5]), np.array(([0.0, 0.8]))

    plt.annotate("", xy=x1, xytext=x0, zorder=-1,
                 arrowprops=dict(arrowstyle="->", color='r', facecolor='black', lw=3))

    plt.text(x=0.16, y=-0.2, s="Space weathering", rotation=-59.5, c='r', fontsize=16)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(sp, ax=ax, cax=cax)
    cbar.ax.set_ylabel('Olivine abundance [vol\%]')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.draw()

    outdir_chemical = outdir + '/chemical/'
    check_dir(outdir_chemical)

    fig_name = "".join(('PCA_plot_BAR_BC.', fig_format))
    fig.savefig("".join((outdir_chemical, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_spectra(x_data, y_data) -> None:
    x = np.arange(450, 2451, 5) / 1000  # um
    titles = ['olivine', 'orthopyroxene', 'clinopyroxene', 'laboratory mixtures', 'meteorites', 'asteroids']

    m, n = best_blk(len(titles))

    # asteroid data
    filename = project_dir + '/Datasets/taxonomy/asteroid_spectra-denoised-norm-meta.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = data[:, 1].astype(np.str)

    inds_S = np.array(['S' in ast_type for ast_type in types])
    inds_Q = np.array(['Q' in ast_type for ast_type in types])
    inds_A = np.array(['A' in ast_type for ast_type in types])
    inds_V = np.array(['V' in ast_type for ast_type in types])

    inds = inds_S + inds_Q + inds_V + inds_A

    # ignore Sq: since its spectrum is very weird
    inds[types == 'Sq:'] = False

    filename_data = 'asteroid_spectra-denoised-norm-nolabel.dat'
    data_file = "".join((project_dir, '/Datasets/', filename_data))
    ast_data = np.loadtxt(data_file, delimiter='\t')[inds]

    fig, ax = plt.subplots(m, n, figsize=(4.7 * n, 4.7 * m))

    ax = np.reshape(ax, (m, n))

    inds = np.where(np.sum(y_data[:, :3] > 0, axis=1) > 1)[0]
    # urceno z konkretnich dat (90, 63)
    inds_met, inds_mix = inds[:63], inds[63:]

    for j in range(len(titles)):
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

        ax[i, k].tick_params(axis='both')
        ax[i, k].set_ylim(bottom=0, top=5)
        ax[i, k].set_xlim(left=0.4, right=2.500)
        ax[i, k].set_title(titles[j])
        ax[i, k].set_xticks(np.arange(0.5, 2.501, 0.5))

        if k > 0:
            ax[i, k].set_yticklabels([])
        else:
            ax[i, k].set_ylabel('Reflectance [normalised]')

        if i == 0:
            ax[i, k].set_xticklabels([])
        else:
            ax[i, k].set_xlabel('Wavelength [$\mu$m]')

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(('spectra_all.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_mineralogy_histogram(y_data: np.ndarray) -> None:
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    i = 0
    mask_ol = y_data[:, i] > 0
    ol = y_data[mask_ol, i] * 100
    i = 1
    mask_opx = y_data[:, i] > 0
    opx = y_data[mask_opx, i] * 100
    i = 2
    mask_cpx = y_data[:, i] > 0
    cpx = y_data[mask_cpx, i] * 100

    i = 3
    fa = y_data[mask_ol, i] * 100

    i = 5
    fs_opx = y_data[mask_opx, i] * 100

    i = 7
    fs_cpx = y_data[mask_cpx, i] * 100
    i = 8
    en_cpx = y_data[mask_cpx, i] * 100
    i = 9
    wo_cpx = y_data[mask_cpx, i] * 100

    fig, ax = plt.subplots(1, 4, figsize=(5 * 4, 5))
    # modal
    ax[0].hist(ol, bins=bins, linewidth=2, edgecolor='c', fill=False, label='olivine')
    ax[0].hist(opx, bins=bins, linewidth=2, edgecolor='m', fill=False, label='orthopyroxene')
    ax[0].hist(cpx, bins=bins, linewidth=2, edgecolor='y', fill=False, label='clinopyroxene')
    ax[0].set_xlabel('Modal abundance [vol\%]')
    ax[0].set_ylabel('Counts')

    # olivine
    ax[1].hist(fa, bins=bins, linewidth=2, edgecolor='c', fill=False, label='fayalite')
    ax[1].set_xlabel('Olivine')

    # orthopyroxene
    ax[2].hist(fs_opx, bins=bins, linewidth=2, edgecolor='c', fill=False, label='ferrosilite')
    ax[2].set_xlabel('Orthopyroxene')

    # clinopyroxene
    ax[3].hist(fs_cpx, bins=bins, linewidth=2, edgecolor='c', fill=False, label='ferrosilite')
    ax[3].hist(en_cpx, bins=bins, linewidth=2, edgecolor='m', fill=False, label='enstatite')
    ax[3].hist(wo_cpx, bins=bins, linewidth=2, edgecolor='y', fill=False, label='wollastonite')
    ax[3].set_xlabel('Clinopyroxene')

    for i in range(4):
        ax[i].set_xlim(left=0, right=100)
        ax[i].tick_params(axis='both')
        ax[i].legend(loc='best')

    plt.draw()
    plt.tight_layout()

    outdir_chemical = outdir + '/chemical/'
    check_dir(outdir_chemical)

    fig_name = "".join(('spectra_min_hist.', fig_format))
    fig.savefig("".join((outdir_chemical, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_model_history(model: Functional, quiet: bool = False) -> None:
    if not quiet:
        print('Model history')

    history = model.history.history

    kernel_width = 5  # Width of the convolution kernel
    conv_kernel = np.ones((kernel_width,))
    conv_kernel /= np.sum(conv_kernel)

    fig = plt.figure("Loss and accuracy", figsize=(8, 6))
    ax1 = fig.add_subplot(111)

    color1, color2 = 'tab:red', 'tab:blue'

    # Normalisation of the edges
    norm = np.convolve(np.ones((len(history['loss']),)), conv_kernel, 'same')

    plot1 = np.convolve(history['loss'], conv_kernel, 'same') / norm

    if model.metrics_names[1] == 'mse':  # MSE to RMSE
        plot3 = np.convolve(np.sqrt(history[model.metrics_names[1]]), conv_kernel, 'same') / norm
        labely = 'rmse'
    else:
        plot3 = np.convolve(history[model.metrics_names[1]], conv_kernel, 'same') / norm
        labely = model.metrics_names[1]

    labely = str(np.char.replace(labely, '_', ' '))

    lns1 = ax1.plot(plot1, color=color1, linestyle='-', label='loss - training')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    lns3 = ax2.plot(plot3, color=color2, linestyle='-', label=labely + ' - training')

    if val_portion > 0:
        plot2 = np.convolve(history['val_loss'], conv_kernel, 'same') / norm
        if model.metrics_names[1] == 'mse':  # MSE to RMSE
            plot4 = np.convolve(np.sqrt(history['val_' + model.metrics_names[1]]), conv_kernel, 'same') / norm
        else:
            plot4 = np.convolve(history['val_' + model.metrics_names[1]], conv_kernel, 'same') / norm
        lns2 = ax1.plot(plot2, color=color1, linestyle=':', label='loss - validation')
        lns4 = ax2.plot(plot4, color=color2, linestyle=':', label=labely + ' - validation')

        lns = lns1 + lns2 + lns3 + lns4
    else:
        lns = lns1 + lns3

    ax1.set_xlabel('epoch')
    ax1.tick_params(axis='x')
    ax1.set_ylabel('loss', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0, right=model.history.params['epochs'])
    ax1.grid(False)

    ax2.set_ylabel(labely, color=color2)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(bottom=0)
    ax2.grid(False)

    labs = [l.get_label() for l in lns]

    if plot3[0] > plot3[-1]:  # the metric decreases if quality increases
        loc = 'upper right'
    else:
        loc = 'center right'
    ax1.legend(lns, labs, loc=loc)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title('Model history')

    plt.draw()
    '''
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    fig_name = "".join((dt_string, '_', 'model_history.', fig_format)) 
    '''
    fig_name = "".join(('model_history.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_Fa_vs_Fs(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print('Plot Fa vs Fs')

    # Indices in which Fa and Fs are
    ind_Fa, ind_Fs = num_minerals, num_minerals + subtypes[0]

    filename = project_dir + '/Datasets/combined-denoised-norm-clean-meta.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = data[:, 7]
    types = np.reshape(types, (len(types), 1))

    # to get the test types
    filename_train_data = 'combined-denoised-norm-clean.dat'
    x_train, y_train = load_data(filename_train_data)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data_proportional(
        np.concatenate((x_train, types), axis=1), y_train)

    types = x_test[:, -1].astype(np.str)

    Fa_true, Fs_true = y_true[:, ind_Fa] * 100, y_true[:, ind_Fs] * 100
    Fa_pred, Fs_pred = y_pred[:, ind_Fa] * 100, y_pred[:, ind_Fs] * 100

    inds_H = np.array([('H' in OC_type) and ('HH' not in OC_type) if len(OC_type) == 2 else False for OC_type in types])
    inds_L = np.array([('L' in OC_type) and ('LL' not in OC_type) if len(OC_type) == 2 else False for OC_type in types])
    inds_LL = np.array(['LL' in OC_type if len(OC_type) == 3 else False for OC_type in types])

    limx1, limx2 = 15, 35
    limy1, limy2 = 10, 30

    shift = 3  # Control ranges of axes
    s = 40  # scaling parameter
    fig, ax = plt.subplots(1, 2, figsize=(4.5 * 2, 6))

    error_Fa, error_Fs = 5.7, 5.7

    for i in range(2):
        # definition of boxes from (for some reasons should be used just once)
        # https://www.researchgate.net/figure/Mineral-compositional-data-for-the-Morokweng-meteoriteDiagram-shows-ferrosilite-Fs_fig3_7093505
        H_rect = patches.Rectangle((16.2, 14.5), 3.8, 3.5, linewidth=1.5, edgecolor='r', facecolor='none')
        L_rect = patches.Rectangle((22.0, 19.0), 4.0, 3.0, linewidth=1.5, edgecolor='g', facecolor='none')
        LL_rect = patches.Rectangle((26.0, 22.0), 6.0, 4.2, linewidth=1.5, edgecolor='b', facecolor='none')

        if i == 0:
            ax[i].scatter(Fa_true[inds_H], Fs_true[inds_H], c='r', s=s, label='H')
            ax[i].scatter(Fa_true[inds_L], Fs_true[inds_L], c='g', s=s, label='L')
            ax[i].scatter(Fa_true[inds_LL], Fs_true[inds_LL], c='b', s=s, label='LL')
        else:
            ax[i].scatter(Fa_pred[inds_H], Fs_pred[inds_H], c='r', s=s, label='H')
            ax[i].scatter(Fa_pred[inds_L], Fs_pred[inds_L], c='g', s=s, label='L')
            ax[i].scatter(Fa_pred[inds_LL], Fs_pred[inds_LL], c='b', s=s, label='LL')

            ax[i].errorbar(Fa_pred[inds_H], Fs_pred[inds_H], xerr=error_Fa, yerr=error_Fs, c='r', fmt='o')
            ax[i].errorbar(Fa_pred[inds_L], Fs_pred[inds_L], xerr=error_Fa, yerr=error_Fs, c='g', fmt='o')
            ax[i].errorbar(Fa_pred[inds_LL], Fs_pred[inds_LL], xerr=error_Fa, yerr=error_Fs, c='b', fmt='o')

        ax[i].set_xlabel('Fa')
        if i == 0:
            ax[i].set_ylabel('Fs')
            ax[i].set_title('ordinary chondrites')
        else:
            ax[i].set_title('predictions')
            ax[i].set_yticklabels([])
        ax[i].tick_params(axis='both')
        ax[i].axis('square')
        ax[i].set_xticks(np.arange(limx1, limx2 + 0.1, 5))
        ax[i].set_yticks(np.arange(limy1, limy2 + 0.1, 5))

        ax[i].set_ylim(bottom=limy1 - shift, top=limy2 + shift)
        ax[i].set_xlim(left=limx1 - shift, right=limx2 + shift)

        # add the patches
        ax[i].add_patch(H_rect)
        ax[i].add_patch(L_rect)
        ax[i].add_patch(LL_rect)

        ax[i].legend()

    Fa_S, Fs_S = [20.7], [18.3]
    Fa_Sq, Fs_Sq = [23.2], [21.0]
    Fa_Sr, Fs_Sr = [18.3], [19.2]
    Fa_Sw, Fs_Sw = [21.3], [14.7]

    Fa_Q, Fs_Q = [26.2], [23.8]

    ax[1].scatter(Fa_S, Fs_S, marker='$S$', c='k', s=s * 2.5)
    ax[1].scatter(Fa_Sq, Fs_Sq, marker='$Sq$', c='k', s=s * 5)
    ax[1].scatter(Fa_Sr, Fs_Sr, marker='$Sr$', c='k', s=s * 5)
    ax[1].scatter(Fa_Sw, Fs_Sw, marker='$Sw$', c='k', s=s * 6)

    ax[1].scatter(Fa_Q, Fs_Q, marker='$Q$', c='k', s=s * 2.5)

    plt.draw()
    plt.tight_layout()

    outdir_chemical = outdir + '/chemical/'
    check_dir(outdir_chemical)

    fig_name = "".join(('Fa_vs_Fs.', fig_format))
    fig.savefig("".join((outdir_chemical, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_surface_spectra(y_pred: np.ndarray, filename: str, what_type: str) -> None:
    # Set is already processed at ray_casting_mean

    def get_most_probable_classes(limit: float = 2) -> Tuple[int, np.ndarray]:
        n_probable_classes = np.sum(sum_of_predictions >= limit)
        most_probable_classes = np.argsort(sum_of_predictions)[-n_probable_classes:][::-1]

        print('\nMost probable classes:')
        for cls in most_probable_classes:
            print('{:4s} {:5.2f}%'.format(classes2[cls], round(sum_of_predictions[cls], 2)))

        return n_probable_classes, most_probable_classes

    def get_most_winning_classes(limit: float = 0) -> Tuple[int, np.ndarray]:
        unique, counts = np.unique(y_pred.argmax(axis=1), return_counts=True)
        unique, counts = unique[counts >= limit], counts[counts >= limit]

        n_probable_classes = np.size(unique)
        most_probable_classes = unique[np.argsort(counts)][::-1]

        norm_counts = np.sort(counts)[::-1] / np.sum(counts) * 100
        print('\nMost winning classes:')
        for icls, cls in enumerate(most_probable_classes):
            print('{:4s} {:5.2f}%'.format(classes2[cls], round(norm_counts[icls], 2)))

        return n_probable_classes, most_probable_classes

    if 'Itokawa' in filename:
        indices_file = "".join((project_dir, '/Datasets/taxonomy/polysum.h5'))
        background_image = 'new_itokawa_mosaic.jpg'
        name = 'Itokawa'
    elif 'Eros' in filename:
        indices_file = "".join((project_dir, '/Datasets/taxonomy/polysumeros1000.h5'))
        background_image = 'eros_cyl_near.jpg'
        name = 'Eros'
    else:
        raise ValueError('"filename" must contain either "Itokawa" or "Eros"')

    with h5py.File(indices_file, 'r') as f:
        # Choosing only the spectra, first two indexes contain coordinates
        # Keeping only the coordinates
        indices = np.array(f['d'][:, :2])

    sum_of_predictions = np.mean(y_pred, axis=0) * 100

    if what_type == 'taxonomy':
        _, most_probable_classes_1 = get_most_probable_classes()
        _, most_probable_classes_2 = get_most_winning_classes()
        most_probable_classes = np.concatenate((most_probable_classes_1,
                                                np.setdiff1d(most_probable_classes_2, most_probable_classes_1)))
        n_probable_classes = len(most_probable_classes)

        titles = ["".join((name, ' spectral ', classes2[most_probable_classes[i]],
                           '-class predictions with confidence')) for i in range(n_probable_classes)]

        labels = [classes2[most_probable_classes[i]] for i in range(n_probable_classes)]

        outdir_surface = outdir + '/classification/'
    elif what_type == 'mineralogy':
        # set titles (this should work well)
        titles_all = [['olivine', 'orthopyroxene', 'clinopyroxene', 'plagioclase'], ['Fa', 'Fo'],
                      ['Fs (OPX)', 'En (OPX)', 'Wo (OPX)'], ['Fs (CPX)', 'En (CPX)', 'Wo (CPX)'], ['An', 'Ab', 'Or']]
        titles_all = np.array(flatten_list(titles_all))[used_indices]
        keep_inds = unique_indices()

        most_probable_classes = keep_inds
        labels = titles_all[keep_inds]

        n_probable_classes = len(most_probable_classes)

        print('\nSelected mineralogy:')
        for i, cls in enumerate(most_probable_classes):
            print('{:14s} {:5.2f}%'.format(labels[i], round(sum_of_predictions[cls], 2)))

        titles = ["".join((name, ' ', labels[i], ' predictions with confidence')) for i in range(n_probable_classes)]

        outdir_surface = outdir + '/chemical/'
    else:
        raise ValueError('"what_type" must be either "taxonomy" or "mineralogy"')

    # Color code dominant classes / labels
    probability_values = np.transpose(np.array([y_pred[:, most_probable_classes[i]]
                                                for i in range(n_probable_classes)]))

    if 'Itokawa' in filename:
        # list of craters from "A survey of possible impact structures on 25143 Itokawa"
        # id, lon, lat, confidence
        craters = np.zeros((38, 3))
        craters[0], craters[1], craters[2], craters[3], craters[4] = [348, 25, 4], [175, -10, 2], [275, -2, 4], [128, 0, 3], [112, 40, 2]
        craters[5], craters[6], craters[7], craters[8], craters[9] = [8, 8, 2], [17, -8, 4], [172, 15, 4], [134, 20, 4], [244, -40, 3]

        craters[10], craters[11], craters[12], craters[13], craters[14] = [151, -6, 3], [215, 17, 3], [269, 34, 1], [145, 4, 3], [102, -10, 1]
        craters[15], craters[16], craters[17], craters[18], craters[19] = [205, -18, 2], [216, -26, 4], [221, -36, 4], [212, -33, 3], [254, -15, 4]

        craters[20], craters[21], craters[22], craters[23], craters[24] = [7, -18, 4], [162, 1, 2], [14, -17, 2], [52, 12, 3], [183, 17, 4]
        craters[25], craters[26], craters[27], craters[28], craters[29] = [169, 24, 4], [345, -17, 3], [277, -13, 4], [45, 19, 3], [117, -39, 3]

        craters[30], craters[31], craters[32], craters[33], craters[34] = [202, 28, 4], [207, 33, 4], [232, -40, 4], [45, -28, 1], [244, 6, 2]
        craters[35], craters[36], craters[37] = [111, -33, 3], [319, -28, 1], [205, -76, 1]

        craters = np.concatenate((np.reshape(np.arange(1, 39), (len(craters), 1)), craters), axis=1)  # ID

    for i in range(n_probable_classes):
        # Plot the coverage map using latitude and longitude from HB
        img = plt.imread("".join((project_dir, '/Asteroid_images/', background_image)))  # Background image
        fig, ax = plt.subplots(figsize=(30, 25))
        ax.imshow(img, cmap="gray", extent=[0, 360, -90, 90], alpha=1)

        # Draw the predictions map
        im = ax.scatter(indices[:, 0], indices[:, 1], s=10, c=probability_values[:, i] * 100,
                        marker=',', cmap="viridis_r", vmin=0, vmax=100, alpha=0.4)

        if 'Itokawa' in filename:
            # draw craters
            cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["pink", "pink", "red", "red"])

            for j in range(len(craters)):
                if craters[j, 0] < 10:
                    fact = 1
                else:
                    fact = 2

                ax.scatter(craters[j, 1], craters[j, 2], s=180 * fact, c=craters[j, 3], cmap=cmap, vmin=1, vmax=4,
                           marker='${:.0f}$'.format(craters[j, 0]), zorder=100)

        plt.xlim([0, 360])
        plt.ylim([-90, 90])
        ax.set_xticks(np.arange(0, 361, 10))
        plt.xticks(rotation=90)
        ax.set_yticks(np.arange(-90, 91, 10))
        plt.grid()
        plt.xlabel('longitude [deg]')  # \N{DEGREE SIGN}
        plt.ylabel('latitude [deg]')
        plt.title(titles[i])

        divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1)
        # cbar = plt.colorbar(im, cax=cax)
        cax = divider.append_axes("bottom", size="10%", pad=1)
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_ticks(np.arange(0, 101, 10))
        cbar.set_ticklabels(np.arange(0, 101, 10))

        plt.draw()
        plt.tight_layout()

        check_dir(outdir_surface)

        fig_name = "".join((name, '_', labels[i].replace(' ', '_'), '.', fig_format))
        fig.savefig("".join((outdir_surface, '/', fig_name)), format=fig_format, bbox_inches='tight',
                    pad_inches=0.05)
    plt.close('all')


def plot_average_surface_spectra(y_pred: np.ndarray, asteroid_name: str) -> None:
    filename_data = asteroid_name + '-nolabel.dat'
    data_file = "".join((project_dir, '/Datasets/', filename_data))
    spectra = np.loadtxt(data_file, delimiter='\t')

    wvl_new_tmp = np.concatenate((np.arange(450, 550, 10), np.arange(560, 2451, 10)))
    if asteroid_name == 'Itokawa':  # (HB:770-2200 vs BDM:450-2450); rozmezi nesedi
        wvl_new = wvl_new_tmp[39:164]
    elif asteroid_name == 'Eros':  # (Eros:810-2450 vs BDM:450-2450); rozmezi drobne nesedi
        wvl_new = wvl_new_tmp[[37, 39, 41, 43, 45, 47, 49, 52, 54, 56, 58, 60, 62, 65, 67, 69, 71, 73, 75, 77, 80, 82,
                               109, 113, 117, 121, 126, 130, 134, 139, 143, 147, 156, 160, 164, 169, 173, 177, 182, 186,
                               191, 199]]
    else:
        raise ValueError('"asteroid_name" must be either "Itokawa" or "Eros"')

    most_probable_classes = np.unique(np.argmax(y_pred, axis=1))

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in most_probable_classes:
        ax.plot(wvl_new, np.mean(spectra[y_pred.argmax(axis=1) == cls], axis=0), label=classes2[cls])

    ax.set_ylabel('Reflectance')
    ax.set_xlabel('Wavelength [nm]')
    plt.legend()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, suf: str = '', quiet: bool = False) -> None:
    if not quiet:
        print('Confusion matrix')

    array = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    sum_cols = np.sum(array, axis=0)  # true positive + false positive
    sum_rows = np.sum(array, axis=1)  # true positive + false negative

    precision = np.round(np.diag(array) / sum_cols * 100)  # in %
    recall = np.round(np.diag(array) / sum_rows * 100)  # in %

    precision_str = np.array(list(map('{:d}'.format, precision.astype(np.int))))
    recall_str = np.array(list(map('{:d}'.format, recall.astype(np.int))))

    # remove NaNs
    precision_str[np.where(~(precision >= 0))[0]] = 'NaN'
    recall_str[np.where(~(recall >= 0))[0]] = 'NaN'

    dim = np.unique(y_true.argmax(axis=1))  # true values in the confusion matrix  # list(range(y_true.shape[1]))
    # normalise colours to sum of rows
    df_cm = pd.DataFrame(normalise_in_rows(array, sum_rows), dim, dim)

    labels = np.array(list(classes.keys()))[dim]

    fig = plt.figure("Confusion Matrix", figsize=(18, 15))
    ax1 = sns.heatmap(df_cm, annot=array, fmt="d", annot_kws={"size": 14}, cmap="Blues", cbar=False)

    # Plot diagonal line
    ax1.plot([0, np.max(dim) + 1], [0, np.max(dim) + 1], 'k--')

    ax2 = ax1.twinx()
    ax3 = ax1.twiny()

    ax1.grid(False)
    ax2.grid(False)
    ax3.grid(False)
    ax1.tick_params(length=0)
    ax2.tick_params(length=0)
    ax3.tick_params(length=0)

    ax1.xaxis.tick_top()  # x axis on top
    ax1.xaxis.set_label_position('top')

    ax3.xaxis.tick_bottom()  # x axis on bottom
    ax3.xaxis.set_label_position('bottom')

    ax1.set_xlabel("Predicted taxonomy")
    ax1.set_ylabel("Actual taxonomy")
    ax3.set_xlabel("Precision [\%]")
    ax2.set_ylabel("Recall [\%]", rotation=270)

    ax1.set_xticklabels(labels)
    ax1.set_yticklabels(labels)

    ax2.set_ylim([0, ax1.get_ylim()[0]])
    ax2.set_yticks(ax1.get_yticks())
    ax2.set_yticklabels(recall_str[::-1], ha="right")
    yax = ax2.get_yaxis()
    yax.set_tick_params(pad=27)
    ax2.yaxis.set_label_coords(1.048, 0.5)

    ax3.set_xlim([0, ax1.get_xlim()[1]])
    ax3.set_xticks(ax1.get_xticks())
    ax3.set_xticklabels(precision_str)

    ax1.axhline(y=0, color='k', linewidth=10)
    ax1.axhline(y=df_cm.shape[1], color='k', linewidth=10)
    ax1.axvline(x=0, color='k', linewidth=10)
    ax1.axvline(x=df_cm.shape[0], color='k', linewidth=10)

    """
    nx, ny = np.shape(array)
    for ix in range(nx):
        for iy in range(ny):
            ax.text(iy + 0.5, ix + 0.5, int(array[ix, iy]), ha="center", va="center", color="r")
    """

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(('confusion_matrix', suf, '.', fig_format))

    outdir_classification = outdir + '/classification/'
    check_dir(outdir_classification)

    fig.savefig("".join((outdir_classification, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_corr_matrix(labels: np.ndarray, hp_values: np.ndarray, suf: str) -> None:
    # pandas corr is NaN tolerant whereas numpy corr is not
    hp_values = pd.DataFrame(np.transpose(hp_values))
    corr_matrix = hp_values.corr().to_numpy()

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.matshow(corr_matrix, vmin=-1, vmax=1)

    plt.xticks(np.arange(0, len(labels), 1.0), rotation=90)
    plt.yticks(np.arange(0, len(labels), 1.0))

    if plt.rcParams['text.usetex']:
        labels = np.char.replace(labels, '_', '\_')
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)

    for ix in range(len(labels)):
        for iy in range(len(labels)):
            ax.text(iy, ix, "".join('{:.2f}'.format(corr_matrix[ix, iy])), ha="center", va="center", color="r")

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(('correlation_matrix', suf, '.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_range_histogram(start: np.ndarray, stop: np.ndarray, step: np.ndarray) -> None:
    print('Range histograms')
    x_lambda = np.arange(0, 3001, 1)
    y_lambda = np.zeros(len(x_lambda))

    for i in range(len(start)):
        tmp = np.zeros(y_lambda.shape)
        tmp[np.where((x_lambda >= start[i]) & (x_lambda <= stop[i]))] = 1
        y_lambda += tmp

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.plot(x_lambda, y_lambda)
    ax2.hist(step, bins='auto')

    ax1.set_xlabel('Wavelength [nm]')
    ax1.set_ylabel('Counts')
    ax1.tick_params(axis='both')
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=np.min(x_lambda), right=np.max(x_lambda))
    ax1.set_title('Histogram of ranges')

    ax2.set_xlabel('Wavelength [nm]')
    ax2.set_xlabel('Wavelength [nm]')
    ax2.set_ylabel('Counts')
    ax2.tick_params(axis='both')
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=0, right=20)
    ax2.set_title('Histogram of resolution')

    plt.draw()

    fig_name = 'hist_range.png'
    fig.savefig("".join((outdir, '/', fig_name)), format='png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_numbers_histogram(fa: np.ndarray, fs: np.ndarray, wo: np.ndarray) -> None:
    print('Numbers histograms')
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.hist(fa, bins=bins)
    ax2.hist(fs, bins=bins)
    ax3.hist(wo, bins=bins)

    ax1.set_xlabel('Fa')
    ax2.set_xlabel('Fs')
    ax3.set_xlabel('Wo')

    ax1.set_ylabel('Counts')

    ax1.tick_params(axis='both')
    ax2.tick_params(axis='both')
    ax3.tick_params(axis='both')

    ax1.set_xlim(left=0, right=1)
    ax2.set_xlim(left=0, right=1)
    ax3.set_xlim(left=0, right=1)

    ax1.set_title('Histogram of Fa')
    ax2.set_title('Histogram of Fs')
    ax3.set_title('Histogram of Wo')

    plt.draw()

    fig_name = 'hist_numbers.png'
    fig.savefig("".join((outdir, '/', fig_name)), format='png', bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_ast_type_hist(y_pred: np.ndarray) -> None:
    print('Plot histograms of asteroids compositions')

    fig_format = 'png'

    nbins = 10
    shift = 3  # Control ylim

    predictions = y_pred * 100

    filename = project_dir + '/Datasets/asteroid_spectra-denoised-norm.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = data[:, 0]

    inds_S = np.array(['S' in ast_type for ast_type in types])
    inds_Q = np.array(['Q' in ast_type for ast_type in types])
    inds_A = np.array(['A' in ast_type for ast_type in types])
    inds_V = np.array(['V' in ast_type for ast_type in types])
    inds = np.array([inds_S, inds_Q, inds_A, inds_V])

    PCA = pd.read_csv(path_taxonomy + 'asteroid_pca-spectra-combined.dat', sep='\t', header=None).to_numpy()

    color = ['blue', 'magenta', 'brown', 'black']
    labels = ['S type', 'Q type', 'A type', 'V type']

    # modal first (expect all four minerals are in the model)
    n_min = 3
    fig, ax = plt.subplots(1, n_min, figsize=(6 * (n_min), 5))  # !!!!!

    labelx = ['Olivine fraction [vol\%]', 'Orthopyroxene fraction [vol\%]',
              'Clinopyroxene fraction [vol\%]', 'Plagioclase fraction [vol\%]']

    limy = 0

    for j in range(n_min):
        # if j > 1:  # !!!!!
        #    continue
        for i in range(len(inds)):
            # for i in [0, 1, 3]:  # !!!!! for i in range(len(inds)):
            #    if j == 0 and i == 3:  # !!!!!
            #        continue
            #    if j == 1 and i == 1:  # !!!!!
            #        continue
            hist, bins = np.histogram(predictions[inds[i, :], j], bins=nbins, range=(0, 100))
            ax[j].bar(bins[:-1] + (bins[1] - bins[0]) / 2, hist.astype(np.float32) / hist.sum() * 100,
                      width=(bins[1] - bins[0]), fill=False, edgecolor=color[i], label=labels[i], linewidth=2)

            if np.max(hist.astype(np.float32) / hist.sum() * 100) > limy:
                limy = np.max(hist.astype(np.float32) / hist.sum() * 100)

        if j > 0:
            ax[j].set_yticklabels([])
        else:
            ax[j].set_ylabel('Normalised counts [\%]')

        ax[j].set_xlabel(labelx[j])
        ax[j].legend(loc='upper left')

    # must be done after the first loop to get limy
    limy = np.min((100, np.round(limy))) + shift
    for j in range(n_min):
        # if j > 1:  # !!!!!
        #    continue
        ax[j].set_ylim(bottom=0, top=limy)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.draw()

    fig_name = "".join(('ast_type_hist_modal.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    # OL
    n_ol = 2
    fig, ax = plt.subplots(1, n_ol, figsize=(6 * n_ol, 5))

    labelx = ['Fa', 'Fo']

    limy = 0

    for j in range(n_ol):
        for i in range(len(inds)):
            hist, bins = np.histogram(predictions[inds[i, :], n_min + j], bins=nbins, range=(0, 100))
            ax[j].bar(bins[:-1] + (bins[1] - bins[0]) / 2, hist.astype(np.float32) / hist.sum() * 100,
                      width=(bins[1] - bins[0]), fill=False, edgecolor=color[i], label=labels[i], linewidth=2)

            if np.max(hist.astype(np.float32) / hist.sum() * 100) > limy:
                limy = np.max(hist.astype(np.float32) / hist.sum() * 100)

        if j > 0:
            ax[j].set_yticklabels([])
        else:
            ax[j].set_ylabel('Normalised counts [\%]')

        ax[j].set_xlabel(labelx[j])
        ax[j].legend(loc='upper left')

    # must be done after the first loop to get limy
    limy = np.min((100, np.round(limy))) + shift
    for j in range(n_ol):
        ax[j].set_ylim(bottom=0, top=limy)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.draw()

    fig_name = "".join(('ast_type_hist_ol.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    # OPX
    n_opx = 2
    fig, ax = plt.subplots(1, n_opx, figsize=(6 * n_opx, 5))

    labelx = ['Fs', 'En']

    limy = 0

    for j in range(n_opx):
        for i in range(len(inds)):
            hist, bins = np.histogram(predictions[inds[i, :], n_min + n_ol + j], bins=nbins, range=(0, 100))
            ax[j].bar(bins[:-1] + (bins[1] - bins[0]) / 2, hist.astype(np.float32) / hist.sum() * 100,
                      width=(bins[1] - bins[0]), fill=False, edgecolor=color[i], label=labels[i], linewidth=2)

            if np.max(hist.astype(np.float32) / hist.sum() * 100) > limy:
                limy = np.max(hist.astype(np.float32) / hist.sum() * 100)

        if j > 0:
            ax[j].set_yticklabels([])
        else:
            ax[j].set_ylabel('Normalised counts [\%]')

        ax[j].set_xlabel(labelx[j])
        ax[j].legend(loc='upper left')

    # must be done after the first loop to get limy
    limy = np.min((100, np.round(limy))) + shift
    for j in range(n_opx):
        ax[j].set_ylim(bottom=0, top=limy)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.draw()

    fig_name = "".join(('ast_type_hist_opx.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)

    # CPX
    n_cpx = 3
    fig, ax = plt.subplots(1, n_cpx, figsize=(6 * n_cpx, 5))

    labelx = ['Fs', 'En', 'Wo']

    limy = 0

    for j in range(n_cpx):
        for i in range(len(inds)):
            hist, bins = np.histogram(predictions[inds[i, :], n_min + n_ol + n_opx + j], bins=nbins, range=(0, 100))
            ax[j].bar(bins[:-1] + (bins[1] - bins[0]) / 2, hist.astype(np.float32) / hist.sum() * 100,
                      width=(bins[1] - bins[0]), fill=False, edgecolor=color[i], label=labels[i], linewidth=2)

            if np.max(hist.astype(np.float32) / hist.sum() * 100) > limy:
                limy = np.max(hist.astype(np.float32) / hist.sum() * 100)

        if j > 0:
            ax[j].set_yticklabels([])
        else:
            ax[j].set_ylabel('Normalised counts [\%]')

        ax[j].set_xlabel(labelx[j])
        ax[j].legend(loc='upper left')

    # must be done after the first loop to get limy
    limy = np.min((100, np.round(limy))) + shift
    for j in range(n_cpx):
        ax[j].set_ylim(bottom=0, top=limy)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    plt.draw()

    fig_name = "".join(('ast_type_hist_cpx.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_Tomas_mixtures() -> None:
    x, spectra_msm, spectra_syn, C = resave_Tomas_OL_OPX_mixtures()

    colours = ['k', 'r', 'b', 'g', 'c', 'm', 'y', 'k', 'r', 'b', 'g', 'c', 'm', 'y']
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    for i in range(len(C)):
        ax.plot(x, spectra_msm[:, i], colours[i], label=C[i])
        ax.plot(x, spectra_syn[:, i], colours[i] + '--')
    ax.set_ylabel('Reflectance')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylim(bottom=0.2, top=0.8)
    ax.tick_params(axis='both')
    ax.set_xlim(left=400, right=2500)

    # ax.legend(loc='upper right')

    plt.draw()
    plt.tight_layout()

    fig_name = 'spectra_mixtures_Tomas.png'
    fig.savefig("".join((outdir, '/', fig_name)), format='png', bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_OC_distance(y_pred: np.ndarray):
    print('plot distance of OC predictions from the corresponding box')

    fig_format = 'png'

    # Indices in which Fa and Fs are
    ind_Fa, ind_Fs = num_minerals, num_minerals + subtypes[0]

    filename = project_dir + '/Datasets/combined-denoised-norm-clean-meta.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = data[:, 7]
    types = np.reshape(types, (len(types), 1))

    # to get the test types
    filename_train_data = 'combined-denoised-norm-clean.dat'
    x_train, y_train = load_data(filename_train_data)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data_proportional(
        np.concatenate((x_train, types), axis=1), y_train)

    types = x_test[:, -1].astype(np.str)

    Fa_pred, Fs_pred = y_pred[:, ind_Fa] * 100, y_pred[:, ind_Fs] * 100

    inds_H = np.array([('H' in OC_type) and ('HH' not in OC_type) if len(OC_type) == 2 else False for OC_type in types])
    inds_L = np.array([('L' in OC_type) and ('LL' not in OC_type) if len(OC_type) == 2 else False for OC_type in types])
    inds_LL = np.array(['LL' in OC_type if len(OC_type) == 3 else False for OC_type in types])

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

    s = 30  # scaling parameter (marker size)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    ax.scatter(distance_H, np.tile(np.array([[0, 1, 2]]).transpose(), (1, np.shape(distance_H)[1])), color='r',
               label='H', s=s)
    ax.scatter(distance_L, np.tile(np.array([[0, 1, 2]]).transpose(), (1, np.shape(distance_L)[1])), color='g',
               label='L', s=s)
    ax.scatter(distance_LL, np.tile(np.array([[0, 1, 2]]).transpose(), (1, np.shape(distance_LL)[1])), color='b',
               label='LL', s=s)

    ax.set_xlabel('Distance')
    ax.set_ylabel('Type')

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['H', 'L', 'LL'])

    ax.legend(loc='center right')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.draw()

    fig_name = "".join(('OC_box_dist.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
