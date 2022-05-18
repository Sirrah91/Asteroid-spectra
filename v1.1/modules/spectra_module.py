from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

mpl.use('TkAgg')

from modules.NN_data import load_data, clean_data
from modules.utilities import *
from modules.CD_parameters import *
from modules.NN_config import num_labels_all


def combine_files(filenames: Tuple[str, ...], final_name: str) -> str:
    outfile_name = "".join((project_dir, '/Datasets/', final_name))
    tmp = Path(outfile_name)
    if tmp.suffix == '':
        outfile_name += '.dat'
    with open(outfile_name, 'w') as outfile:
        for fname in filenames:
            tmp = Path(fname)
            if tmp.suffix == '':
                fname += '.dat'
            if Path(fname).parent.__str__() == '.':  # no directory; relab is default
                with open("".join((project_dir, '/Datasets/RELAB/', fname)), 'r') as infile:
                    for line in infile:
                        outfile.write(line)

            else:
                with open(fname, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
    return outfile_name


def clean_and_resave(filename: str, meta: bool = False, reinterpolate:bool = False) -> None:
    # meta = True for resaving metadata
    tmp = Path(filename)
    if not meta:
        final_name = tmp.stem + '-clean'  # '.dat' is added in save_data
    else:
        final_name = tmp.stem + '-clean-meta'

    if not meta:
        # open data file and clean it
        spectra, labels = load_data(filename, clean_dataset=True, keep_all_labels=True, reinterpolation=reinterpolate)
        # resave it (it is saved to relab folder...)
        file = save_data(final_name, spectra=spectra, labels=labels)
    else:
        file = project_dir + '/Datasets/' + final_name + '.dat'
        metadata = pd.read_csv(project_dir + '/Datasets/' + tmp.stem + '-meta.dat', sep='\t', header=None).to_numpy()

        data_file = "".join((project_dir, '/Datasets/', filename))
        data = np.loadtxt(data_file, delimiter='\t')

        # Select training data
        x_train, y_train = deepcopy(data[:, :-num_labels_all].astype(np.float32)), deepcopy(
            data[:, -num_labels_all:].astype(np.float32))

        *_, inds = clean_data(x_train, y_train, return_indices=True)

        # filter metadata
        metadata = metadata[inds]

        with open(file, 'w') as f:
            for element in metadata:
                final_string = '\t'.join([str(e) for e in element]) + '\n'

                for e in final_string:
                    f.write(e)


def denoise_and_norm(data: np.ndarray, denoising: bool, normalising: bool) -> np.ndarray:
    xq = np.arange(lambda_min, lambda_max + resolution_final / 2, resolution_final)
    if denoising:
        width = 9
        cent = int(np.round(width / 2))
        kernel = np.zeros(width)

        for ii in range(int(np.floor(width / 2))):
            kernel[ii] = 1 / (np.abs(ii - cent) + 1)
            kernel[-ii - 1] = 1 / (np.abs(ii - cent) + 1)
        kernel[cent] = 1

        kernel = kernel / np.sum(kernel)
        correction = np.convolve(np.ones(len(xq)), kernel, 'same')

        if data.ndim == 1:
            data = np.reshape(data, (1, len(data)))
        data_denoised = np.apply_along_axis(lambda m: np.convolve(m, kernel, 'same'), axis=1, arr=data) / correction
    else:
        data_denoised = data

    # Normalised reflectance
    if normalising:
        fun = interp1d(xq, data_denoised, kind='cubic')  # v_final differs from v
        v_norm = np.reshape(fun(normalised_at), (len(data_denoised), 1))
    else:
        v_norm = 1

    return data_denoised / v_norm


def save_data(final_name: str, spectra: np.ndarray, labels: np.ndarray = None, meta: np.ndarray = None,
              subfolder: str = '') -> str:
    path_data = project_dir + '/Datasets/' + subfolder + '/'

    check_dir("".join((path_data, final_name, '.dat')))

    if denoise:
        if '-denoised' not in final_name:
            final_name += '-denoised'
    if normalise:
        if '-norm' not in final_name:
            final_name += '-norm'

    if labels is None:  # Save data without labels
        filename_nolabel = "".join((path_data, final_name, '-nolabel.dat'))
        filename = filename_nolabel
        np.savetxt(filename_nolabel, spectra, fmt='%.5f', delimiter='\t')
    else:  # Save data with labels
        filename_label = "".join((path_data, final_name, '.dat'))
        filename = filename_label

        if type(labels.ravel()[0]) == str:  # taxonomy, labels first
            spectra = np.hstack((labels, spectra))
            np.savetxt(filename_label, spectra, fmt='%s', delimiter='\t')
        else:
            spectra = np.hstack((spectra, labels))
            np.savetxt(filename_label, spectra, fmt='%.5f', delimiter='\t')

    # save metadata if these exist
    # data = pd.read_csv(filename, sep='\t', header=None).to_numpy()  # to read the file
    if meta is not None:
        filename_meta = "".join((path_data, final_name, '-meta.dat'))
        with open(filename_meta, 'w') as f:
            for element in meta:
                final_string = '\t'.join([str(e) for e in element]) + '\n'

                for e in final_string:
                    f.write(e)

    return filename


def normalize_spectra(file: str, save_it: bool = False) -> None:
    path_to_data = path_relab

    # load the data
    data_file = "".join((path_to_data, '/', file, '.dat'))

    xq = np.arange(lambda_min, lambda_max + resolution_final / 2, resolution_final)
    data = np.loadtxt(data_file, delimiter='\t')

    spectra, numbers = data[:, :-num_labels_CD], data[:, -num_labels_CD:]

    fun = interp1d(xq, spectra, kind='cubic')
    v_norm = fun(normalised_at)
    spectra_final = np.transpose(np.divide(np.transpose(spectra), v_norm))

    if save_it:
        save_data(file + '-normalised', spectra_final, labels=numbers)


def remove_continuum(modal: str) -> np.ndarray:
    control_plot = False
    n_labels = 15

    input_file = "".join((path_relab, '/', modal, '-denoised.dat'))
    output_file = "".join((path_relab, '/', modal, '-denoised-nocont_CH.dat'))

    data = np.loadtxt(input_file, delimiter='\t')
    spectra, numbers = data[:, :-n_labels], data[:, -n_labels:]

    n_data, len_data = np.shape(spectra)
    x = np.arange(lambda_min, lambda_max + resolution_final / 2, resolution_final)
    rectified_spectra = np.zeros((n_data, len_data))

    # 2D data for convex hull
    ch_data = np.zeros((len_data, 2))
    ch_data[:, 0] = x

    for i in range(n_data):
        spectrum = spectra[i]
        ch_data[:, 1] = spectrum

        hull = ConvexHull(ch_data).vertices

        # remove lower branch from vertices (delete all vertices between 0 and len0data - 1
        hull = np.roll(hull, -np.where(hull == 0)[0][0] - 1)  # move 0 to the end of the list
        hull = np.sort(hull[np.where(hull == len_data - 1)[0][0]:])

        # keep the UV bands
        x0 = my_argmax(x, spectrum, x0=650)
        hull = hull[np.argmin(np.abs(x[hull] - x0)):]
        continuum = np.zeros(np.shape(x))  # necessary since the UVs start at different positions

        # linear fit to the convex hull
        for j in range(len(hull) - 1):
            x_fit, y_fit = x[[hull[j], hull[j + 1]]], spectrum[[hull[j], hull[j + 1]]]
            if j == 0 and hull[j] != 0:
                x_new = x[:hull[j + 1] + 1]
                continuum[:hull[j + 1] + 1] = np.polyval(np.polyfit(x_fit, y_fit, 1), x_new)
            else:
                x_new = x[hull[j]:hull[j + 1] + 1]
                continuum[hull[j]:hull[j + 1] + 1] = np.polyval(np.polyfit(x_fit, y_fit, 1), x_new)

        rectified_spectra[i] = spectrum / continuum
        rectified_spectra = np.round(rectified_spectra, 5)

        if control_plot:
            fig, ax = plt.subplots()
            ax.plot(x, spectrum / continuum)
            ax.plot(x, spectrum)
            ax.plot(x, continuum)

    rectified_spectra = np.hstack((rectified_spectra, numbers))
    np.savetxt(output_file, rectified_spectra, fmt='%.5f', delimiter='\t')

    return rectified_spectra
