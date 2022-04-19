# These are control plot used in the NN pipeline
import h5py
import matplotlib as mpl
import numpy as np
import pandas as pd

from modules.BAR_BC_method import calc_BAR_BC, calc_composition, filter_data_mask
from modules.collect_data import resave_Tomas_OL_OPX_mixtures

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.models import Functional
from mpl_toolkits.axes_grid1 import make_axes_locatable

from modules.NN_losses_metrics_activations import my_rmse, my_r2, my_sam, my_quantile
from modules.NN_data import *
from modules.NN_config import *
from modules.CD_parameters import path_relab, path_tuomas
from modules.NN_config_taxonomy import classes, classes2

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
plt.rc('text', usetex=True)  # this cases issues when saving to eps.. need to add some nicer fonts

mpl.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
]

plt.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
## for Palatino and other serif fonts use:
# plt.rc('font', **{'family': 'serif', 'serif': ['Palatino']})

outdir = "".join((project_dir, '/figures/'))
check_dir(outdir)


def plot_scatter_plots(y_true: np.ndarray, y_pred: np.ndarray, type: str = '') -> None:
    print('Scatter plots')

    fig_format = 'pdf'

    y_true = y_true[:] * 100
    y_pred = y_pred[:] * 100

    # limit = 0.25
    shift = 3  # Control ranges of axes (from 0 - shift to 100 + shift)
    s = 30  # scaling parameter (marker size)
    vmin = 0  # minimum for colormap in scatter of chemical
    cmap = 'viridis_r'  # cmap of points

    titles_all = [['Fa', 'Fo'], ['Fs', 'En', 'Wo'], ['Fs', 'En', 'Wo'], ['An', 'Ab', 'Or']]
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

    lab_line0, lab_line10, lab_line20 = '\phantom{1}0\% error', '10\% error', '20\% error'

    RMSE = my_rmse(num_minerals)(y_true, y_pred).numpy() / 100  # is multiplied with 100 in the code
    R2 = my_r2(num_minerals)(y_true, y_pred).numpy()
    SAM = my_sam(num_minerals)(y_true, y_pred).numpy() * 180 / np.pi

    # modal first
    start, stop = 0, num_minerals

    if num_minerals > 1:
        x_tmp, y_tmp = y_true[:, start:stop], y_pred[:, start:stop]
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

            ax[i].set_xlabel('Actual')
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
                       r'\mathsf{RMSE} &= ' + '{:4.1f}'.format(RMSE[i]) + r'\\'
                                                                          r'\mathsf{R}^2 &= ' + '{:4.2f}'.format(
                           R2[i]) + r'\\'
                                    r'\mathsf{SAM} &= ' + '{:4.1f}'.format(SAM[i]) + r'\text{ [deg]}'
                                                                                     r'\end{split}'
                                                                                     r'\]',
                       horizontalalignment='center', verticalalignment='center', transform=ax[i].transAxes)

            if i > 0:
                ax[i].set_yticklabels([])

            lns = lns1 + lns2 + lns3
            labs = [l.get_label() for l in lns]
            ax[i].legend(lns, labs, loc='upper left', frameon=False)

        ax[0].set_ylabel('Predicted')

        plt.draw()
        plt.tight_layout()

        fig_name = "".join(('scatter_plot_modal', type, '.', fig_format))
        fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

    # each mineral on separate plot
    for i in range(len(subtypes)):
        start, stop = stop, stop + subtypes[i]

        titles = titles_all[i]
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

        for j in range(subtypes[i]):
            # lines
            lns1 = ax[j].plot(x_line, y_line, 'k', label=lab_line0, linewidth=LW)
            lns2 = ax[j].plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW)
            ax[j].plot(x_line, y1m_line, l10, linewidth=LW)
            lns3 = ax[j].plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW)
            ax[j].plot(x_line, y2m_line, l20, linewidth=LW)
            # data
            sc = ax[j].scatter(x_tmp[:, j], y_tmp[:, j], c=c, cmap=cmap, vmin=vmin, vmax=100, s=s, zorder=100)

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
                       r'\mathsf{RMSE} &= ' + '{:4.1f}'.format(RMSE[start + j]) + r'\\'
                                                                                  r'\mathsf{R}^2 &= ' + '{:4.2f}'.format(
                           R2[start + j]) + r'\\'
                                            r'\mathsf{SAM} &= ' + '{:4.1f}'.format(SAM[start + j]) + r'\text{ [deg]}'
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

        fig_name = "".join(('scatter_plot_', minerals_used[i], type, '.', fig_format))
        fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)


def plot_error_evaluation(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print('Print quantiles')

    fig_format = 'pdf'

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

    # modification of used indices (if there are two labels, their absolute errors are the same; one is enough)
    use_inds = deepcopy([list(use_minerals_all)] + [*subtypes_all])
    for i in range(len(use_inds)):
        if np.sum(use_inds[i]) == 2:
            use_inds[i][np.where(use_inds[i])[0][-1]] = False
    use_inds = flatten_list(use_inds) * used_indices

    # from quantile delete indices, where used_indices are not use_indices
    keep_inds = np.array([i for i in range(len(np.where(used_indices)[0]))
                          if np.where(used_indices)[0][i] in np.where(use_inds)[0]])

    titles_all = titles_all[keep_inds]
    quantile = quantile[:, keep_inds]

    shift = 3  # Control ranges of axes (from 0 - shift to 100 + shift)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if np.shape(quantile)[1] > 10:
        ax.plot(percentile, quantile[:, :10], linewidth=2)
        ax.plot(percentile, quantile[:, 10:], '--', linewidth=2)
        ncol = 2
    else:
        ax.plot(percentile, quantile, linewidth=2)
        ncol = 1
    # constant error lines
    ax.plot(x_line, y_line_10, l10)
    ax.plot(x_line, y_line_20, l20)

    ax.set_xlabel('Percentile')
    ax.set_ylabel('Absolute error')

    ax.set_ylim(bottom=-shift, top=100 + shift)
    ax.set_xlim(left=-shift, right=100 + shift)
    ax.set_xticks(np.arange(0, 100.1, 10))
    ax.set_yticks(np.arange(0, 100.1, 10))

    ax.legend(titles_all, loc='upper left', ncol=ncol)

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(('quantile_error_plot', '.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_ast_PC1_PC2(y_pred: np.ndarray) -> None:
    print('Plot PC1 vs PC2 for asteroids')

    fig_format = 'pdf'

    filename = project_dir + '/Datasets/AP_spectra-denoised-norm-meta.dat'
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

    cbar = plt.colorbar(sp)
    cbar.ax.set_ylabel('Olivine abundance [vol\%]')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.draw()

    fig_name = "".join(('PCA_plot.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_Fa_vs_Fs_ast_only() -> None:
    print('Plot Fa vs Fs')

    fig_format = 'pdf'

    limx1, limx2 = 15, 35
    limy1, limy2 = 10, 30

    shift = 1  # Control ranges of axes
    s = 50  # scaling parameter
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    # definition of boxes from (for some reasons should be used just once)
    # https://www.researchgate.net/figure/Mineral-compositional-data-for-the-Morokweng-meteoriteDiagram-shows-ferrosilite-Fs_fig3_7093505
    H_rect = patches.Rectangle((16.2, 14.5), 3.8, 3.5, linewidth=1.5, edgecolor='r', facecolor='none')
    L_rect = patches.Rectangle((22.0, 19.0), 4.0, 3.0, linewidth=1.5, edgecolor='g', facecolor='none')
    LL_rect = patches.Rectangle((26.0, 22.0), 6.0, 4.2, linewidth=1.5, edgecolor='b', facecolor='none')

    ax.set_xlabel('Fa')
    ax.set_ylabel('Fs')
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

    Fa_S, Fs_S, sigma_fas, sigma_fss = [20.4], [18.4], 4.4, 4.9
    Fa_Sq, Fs_Sq, sigma_fasq, sigma_fssq = [22.9], [21.0], 4.5, 4.1
    Fa_Sr, Fs_Sr, sigma_fasr, sigma_fssr, = [17.9], [19.3], 5.5, 5.3
    Fa_Sw, Fs_Sw, sigma_fasw, sigma_fssw = [20.9], [14.7], 4.0, 4.1

    Fa_Q, Fs_Q, sigma_faq, sigma_fsq = [25.9], [23.9], 5.2, 5.1

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

    fig_name = "".join(('Fa_vs_Fs.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
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
    lab_line0, lab_line10, lab_line20 = '\phantom{1}0\% error', '10\% error', '20\% error'

    shift = 3  # Control ranges of axes (from 0 - shift to 100 + shift)
    fig_format = 'pdf'
    outdir = "".join((project_dir, '/figures/'))
    check_dir(outdir)

    fig, ax = plt.subplots(1, 2, figsize=(4.4 * 2 + 1.5, 6))
    ax[0].scatter(OL_true, OL_pred, s=s, label='NN pred.', zorder=100)
    ax[0].scatter(OL_true, OL_fraction, s=s, label='BAR', zorder=99)

    ax[1].scatter(Fs_true, Fs_pred, s=s, label='NN pred.', zorder=100)
    ax[1].scatter(Fs_true, Fs_v2, s=s, label='BIC', zorder=99)
    ax[1].scatter(Fs_true, Fs, s=s, label='BIIC', zorder=98)

    ax[0].set_title('olivine')
    ax[1].set_title('Fs')

    ax[0].set_ylabel('Modelled')

    for i in range(2):
        ax[i].plot(x_line, y_line, 'k', label=lab_line0, linewidth=LW)
        ax[i].plot(x_line, y1p_line, l10, label=lab_line10, linewidth=LW)
        ax[i].plot(x_line, y1m_line, l10, linewidth=LW)
        ax[i].plot(x_line, y2p_line, l20, label=lab_line20, linewidth=LW)
        ax[i].plot(x_line, y2m_line, l20, linewidth=LW)

        ax[i].set_xlabel('Actual')
        ax[i].tick_params(axis='both')
        ax[i].axis('square')
        ax[i].set_ylim(bottom=-shift, top=100 + shift)
        ax[i].set_xlim(left=-shift, right=100 + shift)

        ax[i].set_xticks(np.arange(0, 100.1, 25))
        ax[i].set_yticks(np.arange(0, 100.1, 25))

        if i > 0:
            ax[i].set_yticklabels([])

        ax[i].legend(loc='upper left', frameon=False)

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(('scatter_plot_NN_BAR_BC_met_mix.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_PCA_BAR():
    from modules.NN_evaluate import evaluate
    fig_format = 'pdf'
    model_names = ['20220330113805_CNN.h5']  # [14.4, 5.7, 5.7, 10.7]
    wvl = np.arange(450, 2451, 5)

    filename_data = 'AP_spectra-denoised-norm-nolabel.dat'
    data_file = "".join((project_dir, '/Datasets/', filename_data))
    data = np.loadtxt(data_file, delimiter='\t')

    y__pred = evaluate(model_names, filename_data)

    filename = project_dir + '/Datasets/AP_spectra-denoised-norm-meta.dat'
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

    cbar = plt.colorbar(sp)
    cbar.ax.set_ylabel('Olivine abundance [vol\%]')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.draw()

    fig_name = "".join(('PCA_plot_BAR_BC.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_spectra(x_data, y_data) -> None:
    fig_format = 'pdf'

    x = np.arange(450, 2451, 5) / 1000  # um
    titles = ['olivine', 'orthopyroxene', 'clinopyroxene', 'laboratory mixtures', 'meteorites', 'asteroids']

    m, n = best_blk(len(titles))

    # asteroid data
    filename = project_dir + '/Datasets/AP_spectra-denoised-norm-meta.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = data[:, 1].astype(np.str)

    inds_S = np.array(['S' in ast_type for ast_type in types])
    inds_Q = np.array(['Q' in ast_type for ast_type in types])
    inds_A = np.array(['A' in ast_type for ast_type in types])
    inds_V = np.array(['V' in ast_type for ast_type in types])

    inds = inds_S + inds_Q + inds_V + inds_A

    # ignore Sq: since its spectrum is very weird
    inds[types == 'Sq:'] = False

    filename_data = 'AP_spectra-denoised-norm-nolabel.dat'
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
            ax[i, k].set_xlabel('$\lambda$ [$\mu$m]')

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(('spectra_all.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close()


def plot_model_history(model: Functional) -> None:
    print('Model history')

    fig_format = 'png'

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
    ax1.legend(lns, labs, loc='upper right')

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

    Fa_S, Fs_S = [20.4], [18.4]
    Fa_Sq, Fs_Sq = [22.9], [21.0]
    Fa_Sr, Fs_Sr = [17.9], [19.3]
    Fa_Sw, Fs_Sw = [20.9], [14.7]

    Fa_Q, Fs_Q = [25.9], [23.9]

    ax[1].scatter(Fa_S, Fs_S, marker='$S$', c='k', s=s * 2.5)
    ax[1].scatter(Fa_Sq, Fs_Sq, marker='$Sq$', c='k', s=s * 5)
    ax[1].scatter(Fa_Sr, Fs_Sr, marker='$Sr$', c='k', s=s * 5)
    ax[1].scatter(Fa_Sw, Fs_Sw, marker='$Sw$', c='k', s=s * 6)

    ax[1].scatter(Fa_Q, Fs_Q, marker='$Q$', c='k', s=s * 2.5)

    plt.draw()
    plt.tight_layout()

    fig_name = "".join(('Fa_vs_Fs.', fig_format))
    fig.savefig("".join((outdir, '/', fig_name)), format=fig_format, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def plot_surface_spectra(y_pred: np.ndarray, filename: str) -> None:
    from modules.NN_config_taxonomy import model_name_suffix
    # Set is already processed at ray_casting_mean

    with h5py.File("".join((project_dir, 'Datasets/Tuomas/', filename)), 'r') as f:
        # Choosing only the spectra, first two indexes contain coordinates
        # Keeping only the coordinates
        indices = np.array(f['d'][:, :2])

    n, _ = y_pred.shape
    n_probable_classes = 3
    most_probable_classes = [0] * n_probable_classes

    # Print percentage of classes against the whole data
    unique, counts = np.unique(y_pred.argmax(axis=1), return_counts=True)  # Dominant classes
    counts = (counts / n) * 100
    print('\nMost probable classes:')
    most_probable_classes[0] = unique[np.argmax(counts)]
    for i in range(len(unique)):
        print('{:4s} {:4.2f}%'.format(classes2[unique[i]], round(counts[i], 2)))

    another_predictions = [0] * n
    for i in range(n):
        another_predictions[i] = np.argsort(y_pred[i])[-2]
    unique, counts = np.unique(another_predictions, return_counts=True)  # Second dominant classes
    counts = (counts / n) * 100
    print('\nSecond most probable classes:')
    most_probable_classes[1] = unique[np.argmax(counts)]
    for i in range(len(unique)):
        print('{:4s} {:4.2f}%'.format(classes2[unique[i]], round(counts[i], 2)))

    for i in range(n):
        another_predictions[i] = np.argsort(y_pred[i])[-3]
    unique, counts = np.unique(another_predictions, return_counts=True)  # Third dominant classes
    counts = (counts / n) * 100
    print('\nThird most probable classes:')
    most_probable_classes[2] = unique[np.argmax(counts)]
    for i in range(len(unique)):
        print('{:4s} {:4.2f}%'.format(classes2[unique[i]], round(counts[i], 2)))

    # Color code dominant classes
    probability_values = np.transpose(np.array([y_pred[:, most_probable_classes[i]]
                                                for i in range(n_probable_classes)]))

    if model_name_suffix == 'Itokawa':
        background_image = 'new_itokawa_mosaic.jpg'
    elif model_name_suffix == 'Eros':
        background_image = 'eros_cyl_near.jpg'
    else:
        raise ValueError('"model_name_suffix" in the config file must be either "Itokawa" or "Eros"')

    for i in range(n_probable_classes):
        # Plot the coverage map using latitude and longitude from HB
        img = plt.imread("".join((project_dir, '/Asteroid_images/', background_image)))  # Background image
        fig, ax = plt.subplots(figsize=(30, 25))
        ax.imshow(img, cmap="gray", extent=[0, 360, -90, 90], alpha=1)

        # Draw the predictions map
        plt.scatter(indices[:, 0], indices[:, 1], s=2, c=probability_values[:, i],
                    marker=',', cmap="viridis_r", alpha=0.4)
        plt.colorbar(orientation="horizontal")
        plt.xlim([0, 360])
        plt.ylim([-90, 90])
        ax.set_xticks(np.arange(0, 361, 10))
        plt.xticks(rotation=90)
        ax.set_yticks(np.arange(-90, 91, 10))
        plt.grid()
        plt.xlabel('longitude [\N{DEGREE SIGN}]')
        plt.ylabel('latitude [\N{DEGREE SIGN}]')
        plt.title("".join((model_name_suffix, ' spectral ', classes2[most_probable_classes[i]],
                           '-class predictions with confidence')))
        plt.draw()

        fig_name = "".join((model_name_suffix, '_', str(i), '.png'))
        fig.savefig("".join((outdir, '/', fig_name)), format='png', bbox_inches='tight', pad_inches=0.05)
    plt.close('all')


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print('Confusion matrix')

    array = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    dim = list(range(y_true.shape[1]))
    df_cm = pd.DataFrame(array, dim, dim)

    labels = classes.keys()

    fig = plt.figure("Confusion Matrix", figsize=(18, 15))
    sns.set(font_scale=1.4)  # label size
    ax = sns.heatmap(df_cm, annot=False, annot_kws={"size": 8}, cmap="Blues", cbar=False)

    ax.xaxis.tick_top()  # x axis on top
    ax.xaxis.set_label_position('top')
    ax.tick_params(length=0)
    ax.tick_params(axis='both')

    # Plot diagonal line
    ax.plot([0, np.max(dim) + 1], [0, np.max(dim) + 1], 'k--')

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.xlabel("Predicted classes")
    plt.ylabel("Actual classes")

    nx, ny = np.shape(array)

    for ix in range(nx):
        for iy in range(ny):
            ax.text(iy + 0.5, ix + 0.5, int(array[ix, iy]), ha="center", va="center", color="r")

    plt.draw()

    fig_name = 'confusion_matrix.eps'
    fig.savefig("".join((outdir, '/', fig_name)), format='eps', bbox_inches='tight', pad_inches=0.05)
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

    filename = path_tuomas + 'AP_spectra-denoised-norm.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = data[:, 0]

    inds_S = np.array(['S' in ast_type for ast_type in types])
    inds_Q = np.array(['Q' in ast_type for ast_type in types])
    inds_A = np.array(['A' in ast_type for ast_type in types])
    inds_V = np.array(['V' in ast_type for ast_type in types])
    inds = np.array([inds_S, inds_Q, inds_A, inds_V])

    PCA = pd.read_csv(project_dir + '/Datasets/' + 'AP_pca-data.dat', sep='\t', header=None).to_numpy()

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
    ax.set_xlabel('$\lambda$ [nm]')
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
