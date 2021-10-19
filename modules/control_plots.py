# These are control plot usage in the NN pipeline
import h5py
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.models import Sequential

from modules.NN_config import *
from modules.CD_parameters_RELAB import path_relab
from modules.NN_config_Tuomas import classes, classes2

font_size = 20


def plot_scatter_plots(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print('Scatter plots')

    y_true = y_true[:] * 100
    y_pred = y_pred[:] * 100

    # limit = 0.25
    shift = 3  # Control ranges of axes (from 0 - shift to 100 + shift)
    s = 30  # scaling parameter
    titles_all = [['Fa', 'Fo'], ['Fs', 'En', 'Wo'], ['Fs', 'En', 'Wo'], ['An', 'Ab', 'Or']]
    titles_all = [titles_all[k] for k in range(len(use_minerals)) if use_minerals_all[k]]
    titles_all = [[titles_all[k][j] for j in range(len(subtypes_all_used[k])) if subtypes_all_used[k][j]]
                  for k in range(len(subtypes))]

    # modal first
    start, stop = 0, num_minerals

    if num_minerals > 1:
        y_tmp, x_tmp = y_true[:, start:stop], y_pred[:, start:stop]
        fig, ax = plt.subplots(1, num_minerals, figsize=(4.5 * num_minerals, 6))
        for i in range(num_minerals):
            ax[i].scatter(x_tmp[:, i], y_tmp[:, i], c='black', s=s)
            ax[i].plot([-50, 150], [-50, 150], 'k')

            ax[i].set_xlabel('Predicted', fontsize=font_size)
            ax[i].tick_params(axis='both', labelsize=font_size)
            ax[i].axis('square')
            ax[i].set_ylim(bottom=-shift, top=100 + shift)
            ax[i].set_xlim(left=-shift, right=100 + shift)
            ax[i].set_title(minerals[i], fontsize=font_size)
            ax[i].set_xticks(np.arange(0, 100.1, 25))
            ax[i].set_yticks(np.arange(0, 100.1, 25))
            if i > 0:
                ax[i].set_yticklabels([])

        ax[0].set_ylabel('Actual', fontsize=font_size)

        plt.draw()
        plt.tight_layout()
        fig.savefig("".join((project_dir, '/figures/scatter_plot_modal.eps')), format='eps', bbox_inches='tight',
                    pad_inches=0)
        plt.close(fig)

    # each mineral on separate plot
    for i in range(len(subtypes)):
        start, stop = stop, stop + subtypes[i]

        y_tmp, x_tmp = y_true[:, start:stop], y_pred[:, start:stop]

        titles = titles_all[i]

        fig, ax = plt.subplots(1, subtypes[i], figsize=(4.5 * subtypes[i], 6))

        if num_minerals > 1:
            c = y_true[:, i]
        else:
            c = 'black'

        for j in range(subtypes[i]):
            ax[j].scatter(x_tmp[:, j], y_tmp[:, j], c=c, cmap='gray_r', vmin=0, vmax=100, s=s)
            ax[j].plot([-50, 150], [-50, 150], 'k')

            ax[j].set_xlabel('Predicted', fontsize=font_size)
            ax[j].tick_params(axis='both', labelsize=font_size)
            ax[j].axis('square')
            ax[j].set_ylim(bottom=-shift, top=100 + shift)
            ax[j].set_xlim(left=-shift, right=100 + shift)
            ax[j].set_title(titles[j], fontsize=font_size)
            ax[j].set_xticks(np.arange(0, 100.1, 25))
            ax[j].set_yticks(np.arange(0, 100.1, 25))
            if j > 0:
                ax[j].set_yticklabels([])

        ax[0].set_ylabel('Actual', fontsize=font_size)

        plt.draw()
        plt.tight_layout()
        fig.savefig("".join((project_dir, '/figures/scatter_plot_', minerals_used[i], '.eps')), format='eps',
                    bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def plot_Fa_vs_Fs(y_pred_or_y_true: np.ndarray) -> None:
    print('Plot Fa vs Fs')

    # Indices in which Fa and Fs are
    ind_Fa, ind_Fs = num_minerals, num_minerals + subtypes[0]

    filename = path_relab + 'OC-norm-denoised_meta.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = data[:, 7]

    # definition of boxes from
    # https://www.researchgate.net/figure/Mineral-compositional-data-for-the-Morokweng-meteoriteDiagram-shows-ferrosilite-Fs_fig3_7093505
    H_rect = patches.Rectangle((16.2, 14.5), 3.8, 3.5, linewidth=1, edgecolor='r', facecolor='none')
    L_rect = patches.Rectangle((22.0, 19.0), 4.0, 3.0, linewidth=1, edgecolor='g', facecolor='none')
    LL_rect = patches.Rectangle((26.0, 22.0), 6.0, 4.2, linewidth=1, edgecolor='b', facecolor='none')

    Fa, Fs = y_pred_or_y_true[:, ind_Fa] * 100, y_pred_or_y_true[:, ind_Fs] * 100

    inds_H = np.array(['H' in type for type in types])
    inds_L = np.array([('L' in type) and ('LL' not in type) for type in types])
    inds_LL = np.array(['LL' in type for type in types])

    """
    limx1, limx2 = (np.floor(np.min(Fa) / 5) * 5, np.ceil(np.max(Fa) / 5) * 5)
    limy1, limy2 = (np.floor(np.min(Fs) / 5) * 5, np.ceil(np.max(Fs) / 5) * 5)
    limx1, limx2 = np.min((limx1, 15)), np.max((limx1, 35))
    limy1, limy2 = np.min((limy1, 10)), np.max((limx1, 30))
    """

    limx1, limx2 = 15, 35
    limy1, limy2 = 10, 30

    shift = 3  # Control ranges of axes
    s = 30  # scaling parameter
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))

    ax.scatter(Fa[inds_H], Fs[inds_H], c='r', s=s)
    ax.scatter(Fa[inds_L], Fs[inds_L], c='g', s=s)
    ax.scatter(Fa[inds_LL], Fs[inds_LL], c='b', s=s)

    ax.set_xlabel('Mole% fayalite', fontsize=font_size)
    ax.set_ylabel('Mole% ferrosilite', fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.axis('square')
    ax.set_xticks(np.arange(limx1, limx2 + 0.1, 5))
    ax.set_yticks(np.arange(limy1, limy2 + 0.1, 5))
    ax.set_ylim(bottom=limy1 - shift, top=limy2 + shift)
    ax.set_xlim(left=limx1 - shift, right=limx2 + shift)

    # add the patches
    ax.add_patch(H_rect)
    ax.add_patch(L_rect)
    ax.add_patch(LL_rect)

    plt.draw()
    plt.tight_layout()
    fig.savefig("".join((project_dir, '/figures/Fa_vs_Fs.eps')), format='eps',
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_Fa_vs_Fs_v2(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    print('Plot Fa vs Fs')

    # Indices in which Fa and Fs are
    ind_Fa, ind_Fs = num_minerals, num_minerals + subtypes[0]

    filename = path_relab + 'OC-norm-denoised_meta.dat'
    data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    types = data[:, 7]

    Fa_true, Fs_true = y_true[:, ind_Fa] * 100, y_true[:, ind_Fs] * 100
    Fa_pred, Fs_pred = y_pred[:, ind_Fa] * 100, y_pred[:, ind_Fs] * 100

    inds_H = np.array(['H' in type for type in types])
    inds_L = np.array([('L' in type) and ('LL' not in type) for type in types])
    inds_LL = np.array(['LL' in type for type in types])

    limx1, limx2 = 15, 35
    limy1, limy2 = 10, 30

    shift = 3  # Control ranges of axes
    s = 30  # scaling parameter
    fig, ax = plt.subplots(1, 2, figsize=(4.5 * 2, 6))

    for i in range(2):

        # definition of boxes from (for some reasons should be used just once)
        # https://www.researchgate.net/figure/Mineral-compositional-data-for-the-Morokweng-meteoriteDiagram-shows-ferrosilite-Fs_fig3_7093505
        H_rect = patches.Rectangle((16.2, 14.5), 3.8, 3.5, linewidth=1, edgecolor='r', facecolor='none')
        L_rect = patches.Rectangle((22.0, 19.0), 4.0, 3.0, linewidth=1, edgecolor='g', facecolor='none')
        LL_rect = patches.Rectangle((26.0, 22.0), 6.0, 4.2, linewidth=1, edgecolor='b', facecolor='none')

        if i == 0:
            ax[i].scatter(Fa_true[inds_H], Fs_true[inds_H], c='r', s=s)
            ax[i].scatter(Fa_true[inds_L], Fs_true[inds_L], c='g', s=s)
            ax[i].scatter(Fa_true[inds_LL], Fs_true[inds_LL], c='b', s=s)
        else:
            ax[i].scatter(Fa_pred[inds_H], Fs_pred[inds_H], c='r', s=s)
            ax[i].scatter(Fa_pred[inds_L], Fs_pred[inds_L], c='g', s=s)
            ax[i].scatter(Fa_pred[inds_LL], Fs_pred[inds_LL], c='b', s=s)

        ax[i].set_xlabel('Mole% fayalite', fontsize=font_size)
        if i == 0:
            ax[i].set_ylabel('Mole% ferrosilite', fontsize=font_size)
            ax[i].set_title('Ordinary chondrites', fontsize=font_size)
        else:
            ax[i].set_title('Predictions', fontsize=font_size)
            ax[i].set_yticklabels([])
        ax[i].tick_params(axis='both', labelsize=font_size)
        ax[i].axis('square')
        ax[i].set_xticks(np.arange(limx1, limx2 + 0.1, 5))
        ax[i].set_yticks(np.arange(limy1, limy2 + 0.1, 5))
        ax[i].set_ylim(bottom=limy1 - shift, top=limy2 + shift)
        ax[i].set_xlim(left=limx1 - shift, right=limx2 + shift)

        # add the patches
        ax[i].add_patch(H_rect)
        ax[i].add_patch(L_rect)
        ax[i].add_patch(LL_rect)

    plt.draw()
    plt.tight_layout()
    fig.savefig("".join((project_dir, '/figures/Fa_vs_Fs_v2.eps')), format='eps',
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def plot_spectra() -> None:
    x = np.arange(350, 2551, 5) / 1000  # um

    fig, ax = plt.subplots(2, 3, figsize=(4.5 * 3, 4.7 * 2))
    titles = ['OL', 'OPX', 'CPX', 'PLG', 'OL-PX mixtures', 'OC']

    for j in range(6):
        if j == 0: filename = 'OL-norm-denoised.dat'
        if j == 1: filename = 'OPX-norm-denoised.dat'
        if j == 2: filename = 'CPX-norm-denoised.dat'
        if j == 3: filename = 'PLG-norm-denoised.dat'
        if j == 4: filename = 'mixtures-norm-denoised.dat'
        if j == 5: filename = 'OC-norm-denoised.dat'

        data_file = "".join((path_relab, filename))
        data = pd.read_csv(data_file, sep='\t', header=None).to_numpy()
        data = np.transpose(data[:, :-num_labels_all])

        i, k = np.unravel_index(j, (2, 3))

        ax[i, k].plot(x, data)

        ax[i, k].tick_params(axis='both', labelsize=font_size)
        ax[i, k].set_ylim(bottom=0, top=5)
        ax[i, k].set_xlim(left=0.3, right=2.600)
        ax[i, k].set_title(titles[j], fontsize=font_size)
        ax[i, k].set_xticks(np.arange(0.5, 2.501, 0.5))

        if k > 0:
            ax[i, k].set_yticklabels([])
        else:
            ax[i, k].set_ylabel('Reflectance [normalised]', fontsize=font_size)

        if i == 0:
            ax[i, k].set_xticklabels([])
        else:
            ax[i, k].set_xlabel('$\lambda$ [$\mu$m]', fontsize=font_size)

    plt.draw()
    plt.tight_layout()
    fig.savefig("".join((project_dir, '/figures/spectra_all.eps')), format='eps',
                bbox_inches='tight', pad_inches=0)
    plt.close()


def plot_model_history(model: Sequential) -> None:
    print('Model history')

    history = model.history.history

    kernel_width = 5  # Width of the convolution kernel

    fig = plt.figure("Loss and accuracy", figsize=(8, 6))
    ax1 = fig.add_subplot(111)

    color1, color2 = 'tab:red', 'tab:blue'

    # Normalisation of the edges
    norm = np.convolve(np.ones((len(history['loss']),)), np.ones((kernel_width,)) / kernel_width, 'same')

    plot1 = np.convolve(history['loss'], np.ones((kernel_width,)) / kernel_width, 'same') / norm
    plot3 = np.convolve(history[model.metrics_names[1]], np.ones((kernel_width,)) / kernel_width, 'same') / norm

    lns1 = ax1.plot(plot1, color=color1, linestyle='-', label='Loss')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    lns3 = ax2.plot(plot3, color=color2, linestyle='-', label='RMSE')

    if val_portion > 0:
        plot2 = np.convolve(history['val_loss'], np.ones((kernel_width,)) / kernel_width, 'same') / norm
        plot4 = np.convolve(history['val_' + model.metrics_names[1]], np.ones((kernel_width,))
                            / kernel_width, 'same') / norm
        lns2 = ax1.plot(plot2, color=color1, linestyle=':', label='Val loss')
        lns4 = ax2.plot(plot4, color=color2, linestyle=':', label='Val RMSE')

        lns = lns1 + lns2 + lns3 + lns4
    else:
        lns = lns1 + lns3

    ax1.set_xlabel('Epoch', fontsize=font_size)
    ax1.tick_params(axis='x', labelsize=font_size)
    ax1.set_ylabel('Loss', color=color1, fontsize=font_size)
    ax1.tick_params(axis='y', labelcolor=color1, labelsize=font_size)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0, right=model.history.params['epochs'])
    ax1.grid(False)

    ax2.set_ylabel('RMSE', color=color2, fontsize=font_size)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=font_size)
    ax2.set_ylim(bottom=0)
    ax2.grid(False)

    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper right', fontsize=font_size)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.title('Model history', fontsize=font_size * 1.3)

    plt.draw()
    fig.savefig("".join((project_dir, '/figures/model_history.png')))
    plt.close(fig)


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
    ax.tick_params(axis='both', labelsize=font_size)

    # Plot diagonal line
    ax.plot([0, np.max(dim) + 1], [0, np.max(dim) + 1], 'k--')

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.xlabel("Predicted classes", fontsize=font_size)
    plt.ylabel("Actual classes", fontsize=font_size)

    nx, ny = np.shape(array)

    for ix in range(nx):
        for iy in range(ny):
            ax.text(iy + 0.5, ix + 0.5, int(array[ix, iy]), ha="center", va="center", color="r", fontsize=font_size)

    plt.draw()
    fig.savefig("".join((project_dir, '/figures/confusion_matrix.eps')), format='eps')
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

    ax1.set_xlabel('Wavelength [nm]', fontsize=font_size)
    ax1.set_ylabel('Counts', fontsize=font_size)
    ax1.tick_params(axis='both', labelsize=font_size)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=np.min(x_lambda), right=np.max(x_lambda))
    ax1.set_title('Histogram of ranges', fontsize=font_size * 1.3)

    ax2.set_xlabel('Wavelength [nm]', fontsize=font_size)
    ax2.set_ylabel('Counts', fontsize=font_size)
    ax2.tick_params(axis='both', labelsize=font_size)
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=0, right=20)
    ax2.set_title('Histogram of resolution', fontsize=font_size * 1.3)

    plt.draw()
    fig.savefig("".join((project_dir, '/figures/hist_range.png')))
    plt.close(fig)


def plot_numbers_histogram(fa: np.ndarray, fs: np.ndarray, wo: np.ndarray) -> None:
    print('Numbers histograms')
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.hist(fa, bins=bins)
    ax2.hist(fs, bins=bins)
    ax3.hist(wo, bins=bins)

    ax1.set_xlabel('Fa', fontsize=font_size)
    ax2.set_xlabel('Fs', fontsize=font_size)
    ax3.set_xlabel('Wo', fontsize=font_size)

    ax1.set_ylabel('Counts', fontsize=font_size)

    ax1.tick_params(axis='both', labelsize=font_size)
    ax2.tick_params(axis='both', labelsize=font_size)
    ax3.tick_params(axis='both', labelsize=font_size)

    ax1.set_xlim(left=0, right=1)
    ax2.set_xlim(left=0, right=1)
    ax3.set_xlim(left=0, right=1)

    ax1.set_title('Histogram of Fa', fontsize=font_size * 1.3)
    ax2.set_title('Histogram of Fs', fontsize=font_size * 1.3)
    ax3.set_title('Histogram of Wo', fontsize=font_size * 1.3)

    plt.draw()
    fig.savefig("".join((project_dir, '/figures/hist_numbers.png')))
    plt.close(fig)


def plot_surface_spectra(y_pred: np.ndarray, filename: str) -> None:
    from modules.NN_config_Tuomas import model_name_suffix
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
        plt.xlabel('longitude [\N{DEGREE SIGN}]', fontsize=font_size)
        plt.ylabel('latitude [\N{DEGREE SIGN}]', fontsize=font_size)
        plt.title("".join((model_name_suffix, ' spectral ', classes2[most_probable_classes[i]],
                           '-class predictions with confidence')), fontsize=font_size)
        plt.draw()
        fig.savefig("".join((project_dir, '/figures/', model_name_suffix, '_', str(i), '.png')), format='png')
    plt.close('all')
