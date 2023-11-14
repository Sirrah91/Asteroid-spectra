from os import environ, path
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold, LeaveOneOut
import numpy as np
from copy import deepcopy
from functools import partial
from tqdm import tqdm
from pathlib import Path

from modules.utilities import check_dir, is_constant, stack
from modules.utilities_spectra import print_header, print_info, load_npz

from modules.control_plots import result_plots

from modules.NN_data import labels_to_categories, load_composition_data, load_taxonomy_data
from modules.NN_train import train
from modules.NN_evaluate import evaluate_test_data

from modules._constants import _path_accuracy_tests, _spectra_name, _wavelengths_name, _metadata_name, _sep_out, _sep_in
from modules._constants import _metadata_key_name, _label_key_name, _label_true_name, _label_pred_name, _config_name, _wp

# defaults only
from modules.NN_config_composition import minerals_used, endmembers_used, comp_model_setup, comp_filtering_setup
from modules.NN_config_composition import comp_output_setup, comp_grid
from modules.NN_config_taxonomy import classes, tax_filtering_setup, tax_output_setup, tax_grid, tax_model_setup
from modules._constants import _rnd_seed


def split_data_for_testing(x_data: np.ndarray, y_data: np.ndarray,
                           options: tuple[str, int, int]) -> tuple[np.ndarray, ...]:
    method, index, K = options

    if method == "LOO":  # Leave-one-out
        # This can be written using KFold(n_splits=len(x_train)); that s definition of K for this case
        train_indices, test_indices = list(LeaveOneOut().split(x_data))[index]
    elif method == "K-fold":  # K-fold method
        train_indices, test_indices = list(KFold(n_splits=K).split(x_data))[index]
    else:
        raise ValueError('Method must be one of "LOO" and "K-fold".')

    x_data, x_test = deepcopy(x_data[train_indices]), deepcopy(x_data[test_indices])
    y_data, y_test = deepcopy(y_data[train_indices]), deepcopy(y_data[test_indices])

    return x_data, y_data, x_test, y_test


def gimme_info(taxonomy: bool, model_option: tuple[str, int, int],
               output_setup: dict | None = None, grid_setup:  dict | None = None,
               filtering_setup: dict | None = None, model_setup: dict | None = None) -> dict[str, any]:

    if taxonomy:
        if output_setup is None: output_setup = tax_output_setup
        if grid_setup is None: grid_setup = tax_grid
        if filtering_setup is None: filtering_setup = tax_filtering_setup
        if model_setup is None: model_setup = tax_model_setup

    else:
        if output_setup is None: output_setup = comp_output_setup
        if grid_setup is None: grid_setup = comp_grid
        if filtering_setup is None: filtering_setup = comp_filtering_setup
        if model_setup is None: model_setup = comp_model_setup


    output = {}

    output["output_setup"] = deepcopy(output_setup)

    output["grid_setup"] = deepcopy(grid_setup)

    output["model_setup"] = deepcopy(model_setup)
    output["model_setup"]["num_models"] = model_option[2]

    output["data_split_setup"] = deepcopy({})
    output["data_split_setup"]["method"] = model_option[0]
    output["data_split_setup"]["num_splits"] = model_option[1]

    output["filtering_setup"] = deepcopy(filtering_setup)

    return output


def save_results(final_name: str, spectra: np.ndarray, wavelengths: np.ndarray, y_true: np.ndarray,
                 y_pred: np.ndarray, metadata: np.ndarray, config_setup: dict[str, any],
                 labels_key: np.ndarray | None = None, metadata_key: np.ndarray | None = None,
                 subfolder: str = ""):
    filename = path.join(_path_accuracy_tests, subfolder, final_name)
    check_dir(filename)

    tmp = Path(filename)
    if tmp.suffix == "":
        filename += ".npz"

    # collect data and metadata
    data_and_metadata = {_spectra_name: np.array(spectra, dtype=_wp),  # save spectra
                         _wavelengths_name: np.array(wavelengths, dtype=_wp),  # save wavelengths
                         _label_true_name: np.array(y_true, dtype=_wp),  # save labels
                         _label_pred_name: np.array(y_pred, dtype=_wp),  # save labels
                         _metadata_name: np.array(metadata, dtype=object),  # save metadata
                         _config_name: config_setup}  # save config file

    if metadata_key is not None:
        data_and_metadata[_metadata_key_name] = np.array(metadata_key, dtype=str)

    if labels_key is not None:
        data_and_metadata[_label_key_name] = np.array(labels_key, dtype=str)

    with open(filename, "wb") as f:
        np.savez_compressed(f, **data_and_metadata)


def gimme_method(maximum_splits: int, len_data: int) -> tuple[str, int]:
    if maximum_splits >= len_data:
        # If LOO then maximum training size
        method = "LOO"
        K = len_data
    else:
        # K-fold otherwise
        method = "K-fold"
        K = maximum_splits

    return method, K


def shuffle_data(x_data: np.ndarray, y_data: np.ndarray, metadata: pd.DataFrame,
                 rnd_seed: int | None = _rnd_seed) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed=rnd_seed)  # to always get the same permutation
    idx = rng.permutation(len(x_data))
    x_data = x_data[idx]
    y_data = y_data[idx]
    metadata = metadata.iloc[idx]

    return x_data, y_data, metadata


if __name__ == "__main__":

    taxonomy = False

    max_splits = 100
    num_models = 1

    if taxonomy:
        from modules.NN_config_taxonomy import num_labels_in_file

        load_data = partial(load_taxonomy_data, used_classes=classes)

        output_setup = tax_output_setup
        grid_setup = tax_grid
        filtering_setup = tax_filtering_setup
        model_setup = tax_model_setup

        filename_train_data = f"asteroid{_sep_in}spectra{_sep_out}{num_labels_in_file}{_sep_out}reduced{_sep_out}denoised{_sep_out}norm.npz"
        use_class_weights = model_setup["use_class_weights"]

    else:
        load_data = partial(load_composition_data, used_minerals=minerals_used, used_endmembers=endmembers_used)

        output_setup = comp_output_setup
        grid_setup = comp_grid
        filtering_setup = comp_filtering_setup
        model_setup = comp_model_setup

        filename_train_data = f"mineral{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz"
        use_class_weights = False

    bin_code = output_setup["bin_code"]

    model_grid = grid_setup["model_grid"]

    model_name, model_subdir = model_setup["model_name"], model_setup["model_subdir"]
    proportiontocut, metrics, p = model_setup["trim_mean_cut"], model_setup["metrics"], model_setup["params"]
    model_type = p["model_usage"]

    model_subdir = path.join("accuracy_tests", model_subdir)

    dt_string = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    results_name = f"{model_type}{_sep_out}{model_grid}{_sep_out}{bin_code}{_sep_out}{dt_string}.npz"

    data = load_npz(filename_train_data, list_keys=[_metadata_key_name, _label_key_name])
    metadata_key = data[_metadata_key_name]

    if taxonomy:
        labels_key = np.array(list(classes.keys()))
    else:
        labels_key = data[_label_key_name]

    # Load the data
    x_train, y_train, meta, wavelengths = load_data(filename_train_data, clean_dataset=True,
                                                    return_meta=True, return_wavelengths=True,
                                                    grid_setup=grid_setup, filtering_setup=filtering_setup)

    method, K = gimme_method(maximum_splits=max_splits, len_data=len(x_train))
    if method == "K-fold":  # Shuffle the data (it is better to do it for K-fold)
        x_train, y_train, meta = shuffle_data(x_train, y_train, meta)

    normalised_at = wavelengths[is_constant(x_train, constant=1.0, axis=0)]
    if np.size(normalised_at) == 0: normalised_at = None

    grid_setup["wvl_grid"], grid_setup["wvl_norm"] = np.array(wavelengths, dtype=_wp), normalised_at
    model_setup["model_subdir"] = model_subdir

    info = gimme_info(taxonomy=taxonomy, model_option=(method, K, num_models), output_setup=output_setup,
                      grid_setup=grid_setup, filtering_setup=filtering_setup, model_setup=model_setup)

    if taxonomy:
        # labels to categories
        y_train = labels_to_categories(y_train, used_classes=classes)

    y_pred = np.zeros(np.shape(y_train))

    # Splitting test indices
    _, _, indices, _ = zip(*[split_data_for_testing(np.arange(len(x_train)), y_train, (method, i, K)) for i in range(K)])
    indices = stack(indices)

    start, stop = 0, 0
    for i in tqdm(range(K)):
        # Split them to train and test parts
        x_train_part, y_train_part, x_test_part, y_test_part = split_data_for_testing(x_train, y_train, (method, i, K))

        # Create and train the neural network and save the model
        model_names = [train(x_train, y_train, np.array([]), np.array([]), params=p,
                             monitoring=comp_model_setup["monitoring"],
                             model_subdir=model_subdir, model_name=model_name,
                             metrics=comp_model_setup["metrics"]) for _ in range(num_models)]

        y_pred_part, accuracy_part = evaluate_test_data(model_names, x_test_part, y_test_part,
                                                        proportiontocut=proportiontocut,
                                                        subfolder_model=model_subdir)
        start, stop = stop, stop + len(y_test_part)
        y_pred[start:stop] = y_pred_part

    save_results(results_name, spectra=x_train[indices], wavelengths=wavelengths,
                 y_true=y_train[indices], y_pred=y_pred, labels_key=labels_key,
                 metadata=meta.iloc[indices], metadata_key=metadata_key,
                 config_setup=info)

    # One can get config with
    # data = load_npz(path.join(_path_accuracy_tests, results_name))
    # config = data[_config_name][()]

    suf = f"_{model_grid}_accuracy_test"

    result_plots(y_train[indices], y_pred, bin_code=bin_code, density_plot=True, suf=suf, quiet=False)

    print("\n-----------------------------------------------------")
    print_header(bin_code=bin_code)
    print_info(y_train[indices], y_pred, bin_code=bin_code, which=method)
    print("-----------------------------------------------------\n")
