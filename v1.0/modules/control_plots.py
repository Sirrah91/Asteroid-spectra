from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import matplotlib as mpl
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from keras.models import Functional

from modules.NN_config import project_dir, num_minerals, subtypes, subtypes_all_used, use_minerals, chem_without_modal
from modules.NN_config import minerals, use_minerals_all, use_pure_only, use_mix_of_the_pure_ones, minerals_used
from modules.NN_config import used_indices, val_portion

from modules.NN_config_taxonomy import classes

from modules.NN_losses_metrics_activations import my_rmse, my_sam, my_r2, my_quantile, loss_name, loss
from modules.NN_losses_metrics_activations import output_activation_name, output_activation, mse_name, mse
from modules.NN_losses_metrics_activations import rmse_name, rmse, quantile_name, quantile, mae_name, mae
from modules.NN_losses_metrics_activations import Lp_norm_name, Lp_norm, r2_name, r2, sam_name, sam

from modules.utilities_spectra import error_estimation_bin_like

from modules.utilities import check_dir, get_weights_from_model, best_blk, flatten_list
from modules.utilities import unique_indices, normalise_in_rows

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
plt.rc('legend', fontsize=SMALL_SIZE)  # fontsize of legend
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# plt.rcParams['text.usetex'] = True
plt.rc('text', usetex=True)

plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
# plt.rc('font', **{'family': 'serif', 'serif': ['Palatino']})

outdir = "".join((project_dir, '/figures/'))
check_dir(outdir)

outdir_composition = outdir + '/compositional/'
check_dir(outdir_composition)

outdir_taxonomy = outdir + '/taxonomical/'
check_dir(outdir_taxonomy)

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

    # predicted_errorbar_reduced, actual_errorbar_reduced = error_estimation_overall(y_true, y_pred, actual_errorbar)
    predicted_errorbar_reduced, actual_errorbar_reduced = error_estimation_bin_like(y_true, y_pred, actual_errorbar)

    # modal first
    start, stop = 0, num_minerals

    if num_minerals > 1:
        x_tmp, y_tmp = y_true[:, start:stop], y_pred[:, start:stop]
        error_pred, error_true = predicted_errorbar_reduced[start:stop], actual_errorbar_reduced[start:stop]

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
            ax[i].errorbar(x_tmp[:, i], y_tmp[:, i], yerr=error_pred[i],
                           xerr=error_true[i], fmt='r', ls='', zorder=99)

            ax[i].set_xlabel('Actual [vol\%]')
            ax[i].tick_params(axis='both')
            ax[i].axis('square')
            ax[i].set_title(minerals[i])
            ax[i].set_xticks(np.arange(0, 100.1, 25))
            ax[i].set_yticks(np.arange(0, 100.1, 25))
            ax[i].set_ylim(bottom=-shift, top=100 + shift)
            ax[i].set_xlim(left=-shift, right=100 + shift)

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

        fig_name = "".join(('scatter_plot_modal', suf, '.', fig_format))
        fig.savefig("".join((outdir_composition, '/', fig_name)), format=fig_format,
                    bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

    # each mineral on separate plot
    for i in range(len(subtypes)):
        start, stop = stop, stop + subtypes[i]

        titles = titles_all[i]
        error_pred, error_true = predicted_errorbar_reduced[start:stop], actual_errorbar_reduced[start:stop]

        # fig size tuned such that the plots are approximately of the same size
        fig, ax = plt.subplots(1, subtypes[i], figsize=(4.4 * subtypes[i] + 1.5, 6))

        if num_minerals > 1:
            # non-zero modal
            mask = y_true[:, i] > 0
            x_tmp, y_tmp = y_true[mask, start:stop], y_pred[mask, start:stop]
            c = y_true[mask, i]
            error_pred, error_true = error_pred[:, :, mask], error_true[:, :, mask]
        else:
            x_tmp, y_tmp = y_true[:, start:stop], y_pred[:, start:stop]
            c = 'black'

        for j in range(subtypes[i]):
            # lines
            lns1 = ax[j].plot(x_line, y_line, 'k', label=lab_line0, linewidth=LW)
            lns2 = ax[j].plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW)
            ax[j].plot(x_line, y1m_line, l10, linewidth=LW)
            lns3 = ax[j].plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW)
            ax[j].plot(x_line, y2m_line, l20, linewidth=LW)
            # data
            sc = ax[j].scatter(x_tmp[:, j], y_tmp[:, j], c=c, cmap=cmap, vmin=vmin, vmax=100, s=s, zorder=100)
            ax[j].errorbar(x_tmp[:, j], y_tmp[:, j], yerr=error_pred[j],
                           xerr=error_true[j], fmt='r', ls='', zorder=99)

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
            ax[j].set_title(titles[j])
            ax[j].set_xticks(np.arange(0, 100.1, 25))
            ax[j].set_yticks(np.arange(0, 100.1, 25))
            ax[j].set_ylim(bottom=-shift, top=100 + shift)
            ax[j].set_xlim(left=-shift, right=100 + shift)

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

        fig_name = "".join(('scatter_plot_', minerals_used[i], suf, '.', fig_format))
        fig.savefig("".join((outdir_composition, '/', fig_name)), format=fig_format,
                    bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)


def plot_error_evaluation(y_true: np.ndarray, y_pred: np.ndarray, suf: str = '', quiet: bool = False) -> None:
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

    ax.set_xticks(np.arange(0, 100.1, 10))
    ax.set_yticks(np.arange(0, 100.1, 10))
    ax.set_ylim(bottom=-shift, top=100 + shift)
    ax.set_xlim(left=-shift, right=100 + shift)

    ax.legend(titles_all, loc='upper left', ncol=ncol)

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(('quantile_error_plot', suf, '.', fig_format))
    fig.savefig("".join((outdir_composition, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_conv_kernels(model_name: str, subfolder_model: str) -> None:
    from keras.models import load_model

    custom_objects = {loss_name: loss, output_activation_name: output_activation,
                      mse_name: mse, rmse_name: rmse, quantile_name: quantile, mae_name: mae, Lp_norm_name: Lp_norm,
                      r2_name: r2, sam_name: sam}

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


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, suf: str = '', quiet: bool = False) -> None:
    if not quiet:
        print('Confusion matrix')

    array = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    sum_cols = np.sum(array, axis=0)  # true positive + false positive
    sum_rows = np.sum(array, axis=1)  # true positive + false negative

    precision = np.round(np.diag(array) / sum_cols * 100)  # in %; here might be zero division
    recall = np.round(np.diag(array) / sum_rows * 100)  # in %

    precision_str = precision.astype(np.int).astype(np.str)
    recall_str = recall.astype(np.int).astype(np.str)

    # remove NaNs
    precision_str[np.where(sum_cols == 0)[0]] = '0'
    recall_str[np.where(sum_rows == 0)[0]] = '0'  # this should never happen

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

    ax2.set_yticks(ax1.get_yticks())
    ax2.set_yticklabels(recall_str[::-1], ha="right")
    ax2.set_ylim([0, ax1.get_ylim()[0]])
    yax = ax2.get_yaxis()
    yax.set_tick_params(pad=27)
    ax2.yaxis.set_label_coords(1.048, 0.5)

    ax3.set_xticks(ax1.get_xticks())
    ax3.set_xticklabels(precision_str)
    ax3.set_xlim([0, ax1.get_xlim()[1]])

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

    fig.savefig("".join((outdir_taxonomy, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
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
