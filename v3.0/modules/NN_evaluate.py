import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.stats import trim_mean
from tqdm import tqdm

from modules.utilities_spectra import (gimme_indices, load_npz, load_keras_model, load_txt, gimme_keys_from_name,
                                       denoise_and_norm, compute_within, compute_metrics, is_taxonomical,
                                       gimme_model_specification, gimme_bin_code_from_name, print_header, print_info,
                                       gimme_custom_objects, normalise_spectra, gimme_grid_setup_from_name)

from modules.NN_data_grids import normalise_spectrum_at_wvl

from modules.control_plots import result_plots

from modules.utilities import normalise_in_rows, is_empty, return_mean_std, to_list, stack

from modules.NN_data import load_composition_data as load_data
from modules.NN_data import split_composition_data_proportional as split_data_proportional

from modules.NN_config_parse import bin_to_used

from modules._constants import _wp, _spectra_name, _sep_in, _quiet, _show_control_plot, _show_result_plot

# defaults only
from modules.NN_config_composition import comp_model_setup, comp_filtering_setup, comp_data_split_setup
from modules.NN_config_taxonomy import tax_model_setup


def average_and_normalise(predictions: np.ndarray, bin_code: str, proportiontocut: float) -> np.ndarray:
    taxonomical = is_taxonomical(bin_code=bin_code)

    # Trimmed mean
    predictions = trim_mean(predictions, proportiontocut, axis=2)

    # Normalisations to 1
    if taxonomical:
        predictions = normalise_in_rows(predictions)

    else:
        used_minerals, used_endmembers = bin_to_used(bin_code=bin_code)

        for start, stop in gimme_indices(used_minerals=used_minerals, used_endmembers=used_endmembers):
            predictions[:, start:stop] = normalise_in_rows(predictions[:, start:stop])


    return np.array(predictions, dtype=_wp)


def check_models(model_names: list[str]) -> None:
    specification_models = [gimme_model_specification(model_name) for model_name in to_list(model_names)]

    # must all be the same
    if not np.all([x == specification_models[0] for x in specification_models]):
        raise ValueError("Not all models have the same specification (grid and output labels).")


def filename_data_to_data(filename_or_data: str | np.ndarray, transpose: bool = False,
                          sep: str = "\t", quiet: bool = False) -> np.ndarray:
    if isinstance(filename_or_data, str):
        # Import the test dataset
        if not quiet:
            print("Loading dataset")

        if ".npz" in filename_or_data:
            filename_or_data = load_npz(filename_or_data, subfolder="")
            filename_or_data = np.array(filename_or_data[_spectra_name], dtype=_wp)

        else:
            filename_or_data = np.array(load_txt(filename_or_data, subfolder="", sep=sep, header=None), dtype=_wp)

    elif isinstance(filename_or_data, np.lib.npyio.NpzFile):
            filename_or_data = np.array(filename_or_data[_spectra_name], dtype=_wp)

    else:
        filename_or_data = np.array(filename_or_data, dtype=_wp)

    if np.ndim(filename_or_data) == 1:
        filename_or_data = np.reshape(filename_or_data, (1, len(filename_or_data)))

    if transpose:
        filename_or_data = np.transpose(filename_or_data)

    # convert data to working precision
    return np.array(filename_or_data, dtype=_wp)


def evaluate(model_names: list[str], filename_or_data: str | np.ndarray,
             proportiontocut: float | None = None,
             subfolder_model: str = "") -> np.ndarray:
    # This function evaluates the mean model on new a dataset

    if not model_names:
        raise ValueError('"model_names" is empty')

    check_models(model_names=model_names)
    bin_code = gimme_bin_code_from_name(model_name=model_names[0])

    # loading needed values
    if is_taxonomical(bin_code=bin_code):
        if proportiontocut is None: proportiontocut = tax_model_setup["trim_mean_cut"]

    else:
        if proportiontocut is None: proportiontocut = comp_model_setup["trim_mean_cut"]

    custom_objects = gimme_custom_objects(model_name=model_names[0])

    filename_or_data = filename_data_to_data(filename_or_data, quiet=_quiet)
    data = filename_or_data

    if not _quiet:
        print("Evaluating the neural network")

    # Calc average prediction over the models
    for idx, model_name in enumerate(model_names):
        model = load_keras_model(model_name, subfolder=subfolder_model, custom_objects=custom_objects)

        if idx == 0:
            predictions = np.zeros((len(data), model.output_shape[1], len(model_names)), dtype=_wp)

        # Evaluate the model on test data
        predictions[:, :, idx] = model.predict(data, verbose=0)  # model.predict(data, verbose=0)

    # Trimmed means and normalisations to 1
    predictions = average_and_normalise(predictions, bin_code=bin_code, proportiontocut=proportiontocut)
    print("-----------------------------------------------------")

    return predictions


def evaluate_test_data(model_names: list[str], x_test: np.ndarray, y_test: np.ndarray,
                       x_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                       x_train: np.ndarray | None = None, y_train: np.ndarray | None = None,
                       proportiontocut: float | None = None,
                       subfolder_model: str = "") -> tuple[np.ndarray, ...]:
    if not model_names:
        raise ValueError('"model_names" is empty')

    check_models(model_names=model_names)
    bin_code = gimme_bin_code_from_name(model_name=model_names[0])

    if is_taxonomical(bin_code=bin_code):
        if proportiontocut is None: proportiontocut = tax_model_setup["trim_mean_cut"]

    else:
        if proportiontocut is None: proportiontocut = comp_model_setup["trim_mean_cut"]

    custom_objects = gimme_custom_objects(model_name=model_names[0])

    # loading needed values
    if "accuracy_test" in subfolder_model or "range_test" in subfolder_model:
        show_result_plot, show_control_plot = False, False
    else:
        show_result_plot, show_control_plot = _show_result_plot, _show_control_plot

    if not _quiet:
        print("Evaluating the neural network on the test data")

    # convert data to working precision
    x_test, y_test = np.array(x_test, dtype=_wp), np.array(y_test, dtype=_wp)
    predictions = np.zeros((*np.shape(y_test), len(model_names)), dtype=_wp)

    do_train = not is_empty(x_train) and not is_empty(y_train)
    do_val = not is_empty(x_val) and not is_empty(y_val)

    if not do_train: y_train = None
    if not do_val: y_val = None

    if do_train:
        # convert data to working precision
        x_train, y_train = np.array(x_train, dtype=_wp), np.array(y_train, dtype=_wp)
        predictions_train = np.zeros((*np.shape(y_train), len(model_names)), dtype=_wp)

    if do_val:
        # convert data to working precision
        x_val, y_val = np.array(x_val, dtype=_wp), np.array(y_val, dtype=_wp)
        predictions_val = np.zeros((*np.shape(y_val), len(model_names)), dtype=_wp)

    # Calc average prediction over the models
    for idx, model_name in enumerate(model_names):
        model = load_keras_model(model_name, subfolder=subfolder_model, custom_objects=custom_objects)

        # Evaluate the model on test data
        predictions[:, :, idx] = model.predict(x_test, verbose=0)  # model.predict(x_test, verbose=0)
        if do_train:
            predictions_train[:, :, idx] = model.predict(x_train, verbose=0)  # model.predict(x_train, verbose=0)
        if do_val:
            predictions_val[:, :, idx] = model.predict(x_val, verbose=0)  # model.predict(x_val, verbose=0)

    # Trimmed means and normalisations to 1
    predictions = average_and_normalise(predictions, bin_code=bin_code, proportiontocut=proportiontocut)
    if do_train:
        predictions_train = average_and_normalise(predictions_train, bin_code=bin_code, proportiontocut=proportiontocut)
    if do_val:
        predictions_val = average_and_normalise(predictions_val, bin_code=bin_code, proportiontocut=proportiontocut)

    if not _quiet:
        print(model_names)

    # These are result plots
    if show_result_plot:
        print("Result plots")
        result_plots(y_test, predictions, bin_code=bin_code)

    if show_control_plot:
        print("Control plots")
        if do_val:
            result_plots(y_val, predictions_val, bin_code=bin_code, suf="_val")
        if do_train:
            result_plots(y_train, predictions_train, bin_code=bin_code, suf="_train")

    # Evaluate the accuracy (this is always printed)
    print("\n-----------------------------------------------------")
    print_header(bin_code=bin_code)

    if do_train:
        print_info(y_train, predictions_train, bin_code=bin_code, which="train")
    if do_val:
        print_info(y_val, predictions_val, bin_code=bin_code, which="validation")
    acc = print_info(y_test, predictions, bin_code=bin_code, which="test")
    print("-----------------------------------------------------\n")

    return predictions, acc


def spectrum_error_transfer(model_name: str, filename_data: str | None = None,
                            snr: float | str | np.ndarray = 50.,
                            filtering_setup: dict | None = None,
                            data_split_setup: dict | None = None,
                            n_trials: int = 100,
                            rnd_seed: int | None = None,
                            subfolder_model: str = "") -> pd.DataFrame:
    if filtering_setup is None: filtering_setup = comp_filtering_setup
    if data_split_setup is None: data_split_setup = comp_data_split_setup

    if filename_data is None:
        if is_taxonomical(model_name):
            raise ValueError("Only composition models are allowed at this moment.")
        else:
            filename_data = "mineral-spectra_denoised.npz"

    if snr == "ASPECT":
        snr = 50.

    label_keys = gimme_keys_from_name(model_name=model_name)
    custom_objects = gimme_custom_objects(model_name=model_name)
    model = load_keras_model(model_name, subfolder=subfolder_model, custom_objects=custom_objects)

    used_minerals, used_endmembers = bin_to_used(bin_code=gimme_bin_code_from_name(model_name), separator=_sep_in)
    data_grid = gimme_grid_setup_from_name(model_name)
    instrument, wvl_grid, wvl_norm = data_grid["instrument"], data_grid["wvl_grid"], data_grid["wvl_norm"]

    # Non-normalised data should be loaded
    data_grid["wvl_norm"] = None
    x_train, y_train, wavelengths = load_data(filename_data, clean_dataset=True,
                                              used_minerals=used_minerals, used_endmembers=used_endmembers,
                                              grid_setup=data_grid, filtering_setup=filtering_setup,
                                              return_wavelengths=True)

    _, _, _, _, spectra, y_true = split_data_proportional(x_train, y_train,
                                                          val_portion=data_split_setup["val_portion"],
                                                          test_portion=data_split_setup["test_portion"],
                                                          used_minerals=used_minerals)
    if np.any(spectra > 1.):
        warnings.warn("You should add noise to non-normalised data.")

    if wvl_norm == "adaptive":
        wvl_norm = normalise_spectrum_at_wvl(wavelengths)
    if wvl_norm is not None:
        spectra_norm = normalise_spectra(spectra, wavelengths, wvl_norm_nm=wvl_norm)
    else:  # This does not have to be here. If wvl_norm_nm == None, normalisation does not do anything
        spectra_norm = deepcopy(spectra)

    calc_rmse = lambda yt, yp: compute_metrics(yt, yp, used_minerals=used_minerals,
                                               used_endmembers=used_endmembers,)
    calc_within = lambda yt, yp: compute_within(yt, yp, error_limit=(10., 20.), used_minerals=used_minerals,
                                                used_endmembers=used_endmembers)

    y_pred_base = np.array(model.predict(spectra_norm, verbose=0), dtype=_wp)
    RMSE_base, = calc_rmse(y_true, y_pred_base)
    within10_base, within20_base = calc_within(y_true, y_pred_base)

    RMSE_noise = np.zeros((n_trials, len(RMSE_base)), dtype=_wp)
    within10_noise = np.zeros((n_trials, len(within10_base)), dtype=_wp)
    within20_noise = np.zeros((n_trials, len(within20_base)), dtype=_wp)

    rng = np.random.default_rng(seed=rnd_seed)

    sigma = spectra / snr

    for i in tqdm(range(n_trials)):
        # add noise
        spectra_noise = spectra + np.array(rng.normal(loc=0., scale=sigma), dtype=_wp)

        # de-noise and normalise
        spectra_noise = denoise_and_norm(spectra_noise, wavelengths, denoising=True, normalising=True, sigma_nm=20.,
                                         wvl_norm_nm=wvl_norm)

        # evaluate
        y_pred_noise = np.array(model.predict(spectra_noise, verbose=0), dtype=_wp)

        RMSE_noise[i], = calc_rmse(y_true, y_pred_noise)
        within10_noise[i], within20_noise[i] = calc_within(y_true, y_pred_noise)

    RMSE_mean, RMSE_std = return_mean_std(RMSE_noise, axis=0)
    within10_mean, within10_std = return_mean_std(within10_noise, axis=0)
    within20_mean, within20_std = return_mean_std(within20_noise, axis=0)

    return pd.DataFrame(stack((RMSE_base, within10_base, within20_base,
                               RMSE_mean, RMSE_std,
                               within10_mean, within10_std,
                               within20_mean, within20_std), axis=1),
                        index=label_keys,
                        columns=["RMSE mean (pp)", "within 10 pp mean (%)", "within 20 pp mean (%)",
                                 "RMSE mean noisy (pp)", "RMSE std noisy (pp)",
                                 "within 10 pp mean noisy (%)", "within 10 pp std noisy (pp)",
                                 "within 20 pp mean noisy (%)", "within 20 pp std noisy (pp)"])
