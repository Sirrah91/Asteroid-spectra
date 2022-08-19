from copy import deepcopy
from typing import Dict
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from scipy.stats import norm

from modules.utilities import *
from modules.CD_parameters import *

mpl.use('Agg')


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


def clean_and_resave(filename: str, meta: bool = False, reinterpolate: bool = False) -> None:
    from modules.NN_config import num_labels_all
    from modules.NN_data import load_data, clean_data

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


def denoise_and_norm(data: np.ndarray, denoising: bool, normalising: bool,
                     normalised_at_wvl: float = 550) -> np.ndarray:
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
        v_norm = np.reshape(fun(normalised_at_wvl), (len(data_denoised), 1))
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


def apply_aspect_filter(x_data: np.ndarray, keep_swir: bool = True) -> np.ndarray:
    wvl_old = np.arange(450, 2451, 5)  # this must follow resolution of your loaded data

    vis, sigma_vis = np.linspace(500, 900, 14), 20 / 2.355  # FWHM -> sigma
    nir1 = np.linspace(850, 1250, 14)
    nir2 = np.linspace(1200, 1600, 14)
    sigma_nir = np.polyval(np.polyfit((850, 1600), (40, 27), 1), np.concatenate((nir1, nir2))) / 2.355  # FWHM -> sigma

    if keep_swir:
        swir, sigma_swir = np.linspace(1650, 2500, 30), 45 / 2.355  # FWHM -> sigma
        N = len(vis) + len(nir1) + len(nir2) + len(swir)  # no. channels
    else:
        N = len(vis) + len(nir1) + len(nir2)  # no. channels

    filtered_data = np.zeros((len(x_data), N))

    for i, v in enumerate(vis):
        gauss = norm.pdf(wvl_old, v, sigma_vis)
        gauss /= np.sum(gauss)
        filtered_data[:, i] = np.sum(x_data * gauss, axis=1)
    for i, n in enumerate(np.concatenate((nir1, nir2))):
        gauss = norm.pdf(wvl_old, n, sigma_nir[i])
        gauss /= np.sum(gauss)
        filtered_data[:, i + len(vis)] = np.sum(x_data * gauss, axis=1)

    if keep_swir:
        for i, s in enumerate(swir):
            gauss = norm.pdf(wvl_old, s, sigma_swir)
            gauss /= np.sum(gauss)
            filtered_data[:, i + N - len(swir)] = np.sum(x_data * gauss, axis=1)

    return filtered_data


def apply_aspect_like_filter(wavelengths: np.ndarray, reflectances: np.ndarray) -> np.ndarray:
    wvl_old = np.arange(450, 2451, 5)  # this must follow resolution of your loaded data

    vis = np.linspace(500, 850, 13)[:-1]
    sigma_vis = np.polyfit((500, 850), (20, 20), 1)

    nir = np.linspace(850, 1600, 26)[:-1]
    sigma_nir = np.polyfit((850, 1600), (40, 27), 1)

    swir = np.linspace(1600, 2500, 30)
    sigma_swir = np.polyfit((1600, 2500), (45, 45), 1)

    filtered_data = np.zeros((len(reflectances), len(wavelengths)))

    fwhm_to_sigma = np.sqrt(8*np.log(2))

    for i, wvl in enumerate(wavelengths):
        if vis[0] <= wvl < nir[0]:
            sigma = np.polyval(sigma_vis, wvl) / fwhm_to_sigma
        elif nir[0] <= wvl < swir[0]:
            sigma = np.polyval(sigma_nir, wvl) / fwhm_to_sigma
        else:
            sigma = np.polyval(sigma_swir, wvl) / fwhm_to_sigma

        gauss = norm.pdf(wvl_old, wvl, sigma)
        gauss /= np.sum(gauss)
        filtered_data[:, i] = np.sum(reflectances * gauss, axis=1)

    return filtered_data


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


def combine_same_range_models(indices, ranges_all_or_spacing_all, what_rmse_all, applied_function):
    #  combine different models

    ranges = len(np.unique(indices)) * ['str']
    what_rmse = np.zeros(len(np.unique(indices)))

    for ind, unique_index in enumerate(np.unique(indices)):
        ranges[ind] = ranges_all_or_spacing_all[np.where(unique_index == indices)[0]][0]
        what_rmse[ind] = applied_function(what_rmse_all[np.where(unique_index == indices)[0]])

    return np.array(ranges).ravel(), what_rmse
