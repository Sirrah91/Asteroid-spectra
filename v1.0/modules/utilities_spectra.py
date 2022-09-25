from copy import deepcopy
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
import h5py

from modules.utilities import *
from modules.CD_parameters import *


def save_data(final_name: str, spectra: np.ndarray, wavelengths: np.ndarray, metadata: np.ndarray,
              labels: np.ndarray = None, labels_key: np.ndarray = None, metadata_key: np.ndarray = None,
              subfolder: str = '') -> str:

    path_data = project_dir + '/Datasets/' + subfolder + '/'

    check_dir("".join((path_data, final_name, '.npz')))

    if denoise:
        if '-denoised' not in final_name:
            final_name += '-denoised'
    if normalise:
        if '-norm' not in final_name:
            final_name += '-norm'

    filename = "".join((path_data, final_name))

    tmp = Path(filename)
    if tmp.suffix == '':
        filename += '.npz'

    # collect data and metadata
    data_and_metadata = {"spectra": spectra.astype(np.float32),  # save spectra
                         "wavelengths": wavelengths.astype(np.float32),  # save wavelengths
                         "metadata": metadata.astype(np.object)}  # save metadata

    if metadata_key is not None:
        data_and_metadata["metadata key"] = metadata_key.astype(np.object)

    if labels is not None:
        if np.shape(labels)[1] == 1:  # taxonomy class
            data_and_metadata["labels"] = labels.astype(np.str)  # save labels
        else:  # composition
            data_and_metadata["labels"] = labels.astype(np.float32)  # save labels

        if labels_key is not None:
            data_and_metadata["label metadata"] = labels_key.astype(np.str)

    with open(filename, 'wb') as f:
        np.savez(f, **data_and_metadata)

    return filename


def combine_files(filenames: Tuple[str, ...], final_name: str) -> str:
    outfile_name = "".join((project_dir, '/Datasets/', final_name))
    tmp = Path(outfile_name)
    if tmp.suffix == '':
        outfile_name += '.npz'

    combined_file = dict(np.load(filenames[0], allow_pickle=True))
    for filename in filenames[1:]:
        file_to_merge = dict(np.load(filename, allow_pickle=True))

        if np.all(combined_file['wavelengths'] == file_to_merge['wavelengths']):
            combined_file["spectra"] = np.vstack((combined_file["spectra"], file_to_merge["spectra"]))
            combined_file["labels"] = np.vstack((combined_file["labels"], file_to_merge["labels"]))
            combined_file["metadata"] = np.vstack((combined_file["metadata"], file_to_merge["metadata"]))

    with open(outfile_name, 'wb') as f:
        np.savez(f, **combined_file)

    return outfile_name


def clean_and_resave(filename: str, reinterpolate: bool = False) -> None:
    from modules.NN_data import load_data

    tmp = Path(filename)
    final_name = tmp.stem + '-clean'  # '.npz' is added in save_data

    # load data for keys and wavelengths
    data = np.load(project_dir + '/Datasets/' + filename, allow_pickle=True)

    # load cleaned data
    spectra, labels, meta = load_data(filename, return_meta=True, keep_all_labels=True,
                                      clean_dataset=True, reinterpolation=reinterpolate)

    # re-save it
    save_data(final_name, spectra=spectra, wavelengths=data["wavelengths"], labels=labels, metadata=meta,
              labels_key=data["label metadata"], metadata_key=data["metadata key"])


def denoise_and_norm(data: np.ndarray, wavelength: np.ndarray, denoising: bool, normalising: bool, sigma_nm: float = 7,
                     normalised_at_wvl: float = 550) -> np.ndarray:
    if denoising:
        if data.ndim == 1:
            data = np.reshape(data, (1, len(data)))

        nm_to_px = 1 / (wavelength[1] - wavelength[0])
        correction = gaussian_filter1d(np.ones(len(wavelength)), sigma=sigma_nm * nm_to_px, mode='constant')
        data_denoised = gaussian_filter1d(data, sigma=sigma_nm * nm_to_px, mode='constant') / correction
    else:
        data_denoised = data

    # Normalised reflectance
    if normalising:
        fun = interp1d(wavelength, data_denoised, kind='cubic')  # v_final differs from v
        v_norm = np.reshape(fun(normalised_at_wvl), (len(data_denoised), 1))
    else:
        v_norm = 1

    return data_denoised / v_norm


def normalize_spectra(file: str, save_it: bool = False) -> None:
    path_to_data = path_relab

    # load the data
    data_file = "".join((path_to_data, '/', file, '.npz'))
    data = np.load(data_file, allow_pickle=True)

    xq, spectra = data["spectra"], data["wavelengths"]

    fun = interp1d(xq, spectra, kind='cubic')
    v_norm = fun(normalised_at)
    spectra_final = np.transpose(np.divide(np.transpose(spectra), v_norm))

    if save_it:
        save_data(file + '-normalised', spectra=spectra_final, wavelengths=xq, labels=data["labels"],
                  metadata=data["metadata"], labels_key=data["label metadata"], metadata_key=data["metadata key"])


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

    fwhm_to_sigma = np.sqrt(8 * np.log(2))

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
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use('Agg')

    control_plot = False
    n_labels = 15

    input_file = "".join((path_relab, '/', modal, '-denoised.npz'))
    output_file = "".join((path_relab, '/', modal, '-denoised-nocont_CH.npz'))

    data = np.load(input_file, allow_pickle=True)
    xq, spectra = data["spectra"], data["wavelengths"]

    n_data, len_data = np.shape(spectra)
    rectified_spectra = np.zeros((n_data, len_data))

    # 2D data for convex hull
    ch_data = np.zeros((len_data, 2))
    ch_data[:, 0] = xq

    for i in range(n_data):
        spectrum = spectra[i]
        ch_data[:, 1] = spectrum

        hull = ConvexHull(ch_data).vertices

        # remove lower branch from vertices (delete all vertices between 0 and len0data - 1
        hull = np.roll(hull, -np.where(hull == 0)[0][0] - 1)  # move 0 to the end of the list
        hull = np.sort(hull[np.where(hull == len_data - 1)[0][0]:])

        # keep the UV bands
        x0 = my_argmax(xq, spectrum, x0=650)
        hull = hull[np.argmin(np.abs(xq[hull] - x0)):]
        continuum = np.zeros(np.shape(xq))  # necessary since the UVs start at different positions

        # linear fit to the convex hull
        for j in range(len(hull) - 1):
            x_fit, y_fit = xq[[hull[j], hull[j + 1]]], spectrum[[hull[j], hull[j + 1]]]
            if j == 0 and hull[j] != 0:
                x_new = xq[:hull[j + 1] + 1]
                continuum[:hull[j + 1] + 1] = np.polyval(np.polyfit(x_fit, y_fit, 1), x_new)
            else:
                x_new = xq[hull[j]:hull[j + 1] + 1]
                continuum[hull[j]:hull[j + 1] + 1] = np.polyval(np.polyfit(x_fit, y_fit, 1), x_new)

        rectified_spectra[i] = spectrum / continuum
        rectified_spectra = np.round(rectified_spectra, 5)

        if control_plot:
            fig, ax = plt.subplots()
            ax.plot(xq, spectrum / continuum)
            ax.plot(xq, spectrum)
            ax.plot(xq, continuum)

    save_data(output_file, spectra=rectified_spectra, wavelengths=xq, labels=data["labels"],
              metadata=data["metadata"], labels_key=data["label metadata"], metadata_key=data["metadata key"])

    return rectified_spectra


def combine_same_range_models(indices, ranges_all_or_spacing_all, what_rmse_all, applied_function):
    #  combine different models

    ranges = len(np.unique(indices)) * ['str']
    what_rmse = np.zeros(len(np.unique(indices)))

    for ind, unique_index in enumerate(np.unique(indices)):
        ranges[ind] = ranges_all_or_spacing_all[np.where(unique_index == indices)[0]][0]
        what_rmse[ind] = applied_function(what_rmse_all[np.where(unique_index == indices)[0]])

    return np.array(ranges).ravel(), what_rmse


def cut_error_bars(y_true: np.ndarray, y_true_error: np.ndarray, y_pred: np.ndarray, y_pred_error: np.ndarray
                   ) -> Tuple[np.ndarray, np.ndarray]:
    lower_error = (y_true + y_true_error - np.abs(y_true - y_true_error)) / 2
    upper_error = (100 + y_true_error - y_true - np.abs(y_true - 100 + y_true_error)) / 2
    actual_errorbar_reduced = np.array(list(zip(lower_error, upper_error))).T

    lower_error = (y_pred + y_pred_error - np.abs(y_pred - y_pred_error)) / 2
    upper_error = (100 + y_pred_error - y_pred - np.abs(y_pred - 100 + y_pred_error)) / 2
    predicted_errorbar_reduced = np.array(list(zip(lower_error, upper_error))).T

    return predicted_errorbar_reduced, actual_errorbar_reduced


def error_estimation_overall(y_true: np.ndarray, y_pred: np.ndarray, actual_error: np.ndarray = 3
                             ) -> Tuple[np.ndarray, np.ndarray]:
    from modules.NN_losses_metrics_activations import my_rmse
    from modules.NN_config import num_minerals

    if np.all(y_true < 1.1):  # to percents
        y_true = y_true[:] * 100
        y_pred = y_pred[:] * 100

    RMSE = my_rmse(num_minerals)(y_true, y_pred).numpy() / 100  # is multiplied with 100 in the code

    return cut_error_bars(y_true, actual_error, y_pred, RMSE)


def error_estimation_bin_like(y_true: np.ndarray, y_pred: np.ndarray, actual_error: np.ndarray = 3
                              ) -> Tuple[np.ndarray, np.ndarray]:
    from modules.NN_losses_metrics_activations import my_rmse, clean_ytrue_ypred
    from modules.NN_config import num_minerals, num_labels

    y_true_clean, y_pred_clean = clean_ytrue_ypred(y_true, y_pred, num_minerals)
    y_true_clean, y_pred_clean = y_true_clean.numpy(), y_pred_clean.numpy()

    if np.any(y_true_clean > 101):  # to percents
        y_true_clean /= 100
        y_pred_clean /= 100

    # N bins (step 100 / N)
    N = 10

    predicted_error = np.zeros((N, np.shape(y_pred)[1]))  # errors for each bin
    predicted_error_no = np.zeros((N, np.shape(y_pred)[1]))  # number of points for each bin
    errors_mod = np.zeros((len(y_pred), num_minerals))  # final errors for each point
    errors_chem = np.zeros((len(y_pred), num_labels - num_minerals))  # final errors for each point

    for i in range(N):
        mask = np.logical_and(100 / N * i <= y_pred_clean, y_pred_clean <= 100 / N * (i + 1))

        predicted_error_no[i] = np.sum(mask, axis=0)

        # modal and chemical must be done separately
        mask_modal, mask_chemical = mask[:, :num_minerals], mask[:, num_minerals:]

        y_pred_mod, y_pred_chem = y_pred_clean[:, :num_minerals], y_pred_clean[:, num_minerals:]
        y_true_mod, y_true_chem = y_true_clean[:, :num_minerals], y_true_clean[:, num_minerals:]

        # modal first
        y_p = np.where(mask_modal, y_pred_mod, np.nan)
        y_t = np.where(mask_modal, y_true_mod, np.nan)

        y_p = np.concatenate((y_p, y_pred_chem), axis=1)
        y_t = np.concatenate((y_t, y_true_chem), axis=1)

        # must be /100
        tmp_rmse = my_rmse(num_minerals)(y_t, y_p).numpy()[:num_minerals]
        predicted_error[i, :num_minerals] = tmp_rmse

        # easier to replicate it and then copy
        tmp_rmse = np.reshape(tmp_rmse, (1, np.size(tmp_rmse)))
        tmp_rmse = np.repeat(tmp_rmse, repeats=len(y_pred), axis=0)
        errors_mod = np.where(mask_modal, tmp_rmse, errors_mod)

        # chemical second
        y_p = np.where(mask_chemical, y_pred_chem, np.nan)
        y_t = np.where(mask_chemical, y_true_chem, np.nan)

        y_p = np.concatenate((y_pred_mod, y_p), axis=1)
        y_t = np.concatenate((y_true_mod, y_t), axis=1)

        tmp_rmse = my_rmse(num_minerals)(y_t, y_p).numpy()[num_minerals:]
        predicted_error[i, num_minerals:] = tmp_rmse

        # easier to replicate it and then copy
        tmp_rmse = np.reshape(tmp_rmse, (1, np.size(tmp_rmse)))
        tmp_rmse = np.repeat(tmp_rmse, repeats=len(y_pred), axis=0)
        errors_chem = np.where(mask_chemical, tmp_rmse, errors_chem)

    errors = np.concatenate((errors_mod, errors_chem), axis=1)
    errors /= 100

    predicted_error /= 100  # can be printed
    predicted_error = np.round(predicted_error, 1)
    np.dstack((predicted_error, predicted_error_no)).T

    return cut_error_bars(y_true_clean, actual_error, y_pred_clean, errors)
