from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from datetime import datetime
import numpy as np
from tqdm import tqdm

from modules.utilities import is_constant, stack
from modules.utilities_spectra import print_header, print_info, load_npz

from modules.NN_data import load_composition_data
from modules.NN_train import train
from modules.NN_evaluate import evaluate_test_data

from modules.NN_accuracy_test import split_data_for_testing, gimme_info, save_results, gimme_method, shuffle_data

from modules._constants import _metadata_key_name, _label_key_name, _sep_out, _sep_in

from modules.NN_config_range_test import model_subdirs, model_names, range_grids

from modules.NN_config_composition import minerals_used, endmembers_used, comp_model_setup, comp_filtering_setup
from modules.NN_config_composition import comp_output_setup

if __name__ == "__main__":

    max_splits = 2
    num_models = 1

    load_data = load_composition_data

    output_setup = comp_output_setup
    grid_setup_list = range_grids
    filtering_setup = comp_filtering_setup
    model_setup = comp_model_setup

    filename_train_data = f"mineral{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz"

    bin_code = output_setup["bin_code"]

    proportiontocut, metrics, p = model_setup["trim_mean_cut"], model_setup["metrics"], model_setup["params"]
    model_type = p["model_usage"]

    data = load_npz(filename_train_data, list_keys=[_metadata_key_name, _label_key_name])
    labels_key, metadata_key = data[_label_key_name], data[_metadata_key_name]

    for index_of_range in tqdm(range(len(model_names))):
        grid_setup = grid_setup_list[index_of_range]
        model_grid = grid_setup["model_grid"]
        model_name, model_subdir = model_names[index_of_range], model_subdirs[index_of_range]

        dt_string = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        results_name = f"{model_type}{_sep_out}{model_grid}{_sep_out}{bin_code}{_sep_out}{dt_string}.npz"

        # Load the data
        x_train, y_train, meta, wavelengths = load_data(filename_train_data, clean_dataset=True,
                                                        return_meta=True, return_wavelengths=True,
                                                        used_minerals=minerals_used, used_endmembers=endmembers_used,
                                                        grid_setup=grid_setup, filtering_setup=filtering_setup)

        method, K = gimme_method(maximum_splits=max_splits, len_data=len(x_train))
        if method == "K-fold":  # Shuffle the data (it is better to do it for K-fold)
            x_train, y_train, meta = shuffle_data(x_train, y_train, meta)

        normalised_at = wavelengths[is_constant(x_train, constant=1.0, axis=0)]

        grid_setup["wvl_grid"], grid_setup["wvl_norm"] = wavelengths, normalised_at
        model_setup["model_subdir"], model_setup["model_subdir"] = model_subdir, model_name

        info = gimme_info(taxonomy=False, model_option=(method, K, num_models), output_setup=output_setup,
                          grid_setup=grid_setup, filtering_setup=filtering_setup, model_setup=model_setup)

        y_pred = np.zeros(np.shape(y_train))

        # Splitting test indices
        _, _, indices, _ = zip(*[split_data_for_testing(np.arange(len(x_train)), y_train, (method, i, K)) for i in range(K)])
        indices = stack(indices)

        start, stop = 0, 0
        for i in tqdm(range(K)):
            # Split them to train and test parts
            x_train_part, y_train_part, x_test_part, y_test_part = split_data_for_testing(x_train, y_train, (method, i, K))

            # Create and train the neural network and save the model
            model_names_trained = [train(x_train, y_train, np.array([]), np.array([]), params=p,
                                         monitoring=comp_model_setup["monitoring"],
                                         model_subdir=model_subdir, model_name=model_name,
                                         metrics=comp_model_setup["metrics"]) for _ in range(num_models)]

            y_pred_part, accuracy_part = evaluate_test_data(model_names_trained, x_test_part, y_test_part,
                                                            proportiontocut=proportiontocut,
                                                            subfolder_model=model_subdir)
            start, stop = stop, stop + len(y_test_part)
            y_pred[start:stop] = y_pred_part

        save_results(results_name, spectra=x_train[indices], wavelengths=wavelengths,
                     y_true=y_train[indices], y_pred=y_pred, labels_key=labels_key,
                     metadata=meta.iloc[indices], metadata_key=metadata_key,
                     config_setup=info, subfolder=model_subdir)

        # One can get config with
        # data = load_npz(path.join(_path_accuracy_tests, results_name))
        # config = data[_config_name][()]

        print("\n-----------------------------------------------------")
        print_header(bin_code=bin_code)
        print_info(y_train[indices], y_pred, bin_code=bin_code, which=method)
        print("-----------------------------------------------------\n")
