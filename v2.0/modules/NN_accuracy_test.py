from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from typing import Any, Literal
from datetime import datetime
from sklearn.model_selection import KFold, LeaveOneOut
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path

from modules.utilities import check_dir
from modules.utilities_spectra import print_accuracy, print_accuracy_header, used_indices
from modules.control_plots import plot_scatter_plots, plot_error_evaluation, plot_error_density_plots
from modules.control_plots import plot_confusion_matrix

from modules.NN_losses_metrics_activations import my_rmse
from modules.NN_data import labels_to_categories
from modules.NN_train import train
from modules.NN_evaluate import evaluate_test_data

from modules.NN_config import minerals_all, minerals_used, endmembers_used
from modules.NN_config import num_minerals, endmembers_counts, use_pure_only, use_mix_of_the_pure_ones
from modules.NN_config_taxonomy import classes, metrics

from modules._constants import _path_data, _path_accuracy_test


def split_data_for_testing(x_data: np.ndarray, y_data: np.ndarray,
                           options: tuple[str, int, int]) -> tuple[np.ndarray, ...]:
    method = options[0]
    index = options[1]
    K = options[2]

    if method == "LOO":  # Leave-one-out
        # This can be written using KFold(n_splits=len(x_train)); that s definition of K for this case
        train_indices, test_indices = list(LeaveOneOut().split(x_data))[index]
    elif method == "K-fold":  # K-fold method
        train_indices, test_indices = list(KFold(n_splits=K).split(x_data))[index]
    else:
        raise ValueError('Method must be one of "LOO" and "K-fold".')

    x_data, x_test = deepcopy(x_data[train_indices, :]), deepcopy(x_data[test_indices, :])
    y_data, y_test = deepcopy(y_data[train_indices]), deepcopy(y_data[test_indices])

    return x_data, y_data, x_test, y_test


def gimme_info(filename_data: str, to_what_data: Literal["Itokawa", "Eros", "Didymos"] | None,
               taxonomical: bool) -> dict[str, Any]:
    if taxonomical:
        from modules.NN_config_taxonomy import p
    else:
        from modules.NN_config import p

    data = np.load("".join((_path_data, filename_data)), allow_pickle=True)

    if to_what_data is None:
        output = {"wavelengths": np.array(data["wavelengths"], dtype=np.float32)}
    else:
        if to_what_data == "Itokawa":  # Itokawa
            data_file = "".join((_path_data, "/Itokawa-denoised-norm.npz"))
        elif to_what_data == "Eros":  # Eros
            data_file = "".join((_path_data, "/Eros-denoised-norm.npz"))
        elif to_what_data == "Didymos":
            data_file = "".join((_path_data, "/Didymos_2022-denoised-norm.npz"))
        else:
            raise ValueError("".join(("Unknown resolution of ", to_what_data, ".")))

        data_new = np.load(data_file, allow_pickle=True)
        output = {"wavelengths": np.array(data_new["wavelengths"], dtype=np.float32)}

    if "metadata key" in data.files:
        output["metadata key"] = np.array(data["metadata key"], dtype=object)
    if "label metadata" in data.files:
        if taxonomical:
            output["label metadata"] = np.array(data["label metadata"], dtype=str)
        else:
            output["label metadata"] = np.array(data["label metadata"],
                                                dtype=str)[used_indices(minerals_used, endmembers_used)]

    if taxonomical:
        output["model info"] = {"params": p,
                                "used classes": classes}
    else:
        output["model info"] = {"params": p,
                                "used minerals": minerals_all,
                                "used end-members": endmembers_used,
                                "pure only": use_pure_only,
                                "pure and their mixtures": use_mix_of_the_pure_ones}

    return output


def save_results(final_name: str, x_data: np.ndarray, wavelengths: np.ndarray, y_data: np.ndarray,
                 y_predicted: np.ndarray, metadata: np.ndarray, model_info: dict[str, Any],
                 labels_key: np.ndarray | None = None, metadata_key: np.ndarray | None = None):

    check_dir("".join((_path_accuracy_test, final_name, ".npz")))
    filename = "".join((_path_accuracy_test, final_name))

    tmp = Path(filename)
    if tmp.suffix == "":
        filename += ".npz"

    # collect data and metadata
    data_and_metadata = {"spectra": np.array(x_data, dtype=np.float32),  # save spectra
                         "wavelengths": np.array(wavelengths, dtype=np.float32),  # save wavelengths
                         "labels true": np.array(y_data, dtype=np.float32),  # save labels
                         "labels predicted": np.array(y_predicted, dtype=np.float32),  # save labels
                         "metadata": np.array(metadata, dtype=object),  # save metadata
                         "model info": model_info}  # save model info

    if metadata_key is not None:
        data_and_metadata["metadata key"] = np.array(metadata_key, dtype=object)

    if labels_key is not None:
        data_and_metadata["label metadata"] = np.array(labels_key, dtype=str)

    with open(filename, "wb") as f:
        np.savez(f, **data_and_metadata)


if __name__ == "__main__":

    taxonomical = True

    if taxonomical:
        from modules.NN_data import load_taxonomical_data as load_data
        from modules.NN_config_taxonomy import model_subdir, model_name_suffix, interpolate_to
        from modules.NN_config_taxonomy import trim_mean_cut, model_grid, p

        if model_subdir != "accuracy_test":
            raise ValueError('Change "model_subdir" in NN_config_taxonomy.py to "accuracy_test".')

        filename_train_data = "asteroid_spectra-reduced-denoised-norm.npz"
        filename_suffix = "taxonomical"

    else:
        from modules.NN_data import load_compositional_data as load_data
        from modules.NN_config import model_subdir, model_name_suffix, interpolate_to
        from modules.NN_config import trim_mean_cut, model_grid, p

        if model_subdir != "accuracy_test":
            raise ValueError('Change "model_subdir" in NN_config.py to "accuracy_test".')

        filename_train_data = "combined-denoised-norm.npz"
        filename_suffix = "compositional"

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    results_name = "".join((dt_string, "_", model_name_suffix, "_", filename_suffix, ".npz"))

    method = "K-fold"
    num_models = 1

    # Load the data
    x_train, y_train, meta = load_data(filename_train_data, clean_dataset=True, return_meta=True)

    if method == "LOO":  # If LOO then maximum training size
        K = len(x_train)
    elif method == "K-fold":
        K = 100

        # Shuffle the data (it is better to do it for K-fold)
        rng = np.random.default_rng(seed=42)  # to always get the same permutation
        idx = rng.permutation(len(x_train))
        x_train = x_train[idx]
        y_train = y_train[idx]
        meta = meta[idx]
    else:
        raise ValueError('Method must be one of "LOO" and "K-fold".')

    info = gimme_info(filename_train_data, interpolate_to, taxonomical)
    info["model info"][method] = K
    info["model info"]["num_models"] = num_models
    info["model info"]["trim_mean_cut"] = trim_mean_cut

    if taxonomical:
        # labels to categories
        y_train = labels_to_categories(y_train)

    y_pred = np.zeros(np.shape(y_train))
    y_true = np.zeros(np.shape(y_train))

    start, stop = 0, 0

    for i in tqdm(range(K)):
        # Split them to train and test parts
        x_train_part, y_train_part, x_test_part, y_test_part = split_data_for_testing(x_train, y_train, (method, i, K))

        # Create and train the neural network and save the model
        model_names = [train(x_train, y_train, np.array([]), np.array([]), p) for _ in range(num_models)]
        y_pred_part, accuracy_part = evaluate_test_data(model_names, x_test_part, y_test_part,
                                                        subfolder_model="accuracy_test")
        start, stop = stop, stop + len(y_test_part)
        y_pred[start:stop] = y_pred_part
        y_true[start:stop] = y_test_part

    if "metadata key" in info.keys():
        metadata_key = info["metadata key"]
    else:
        metadata_key = None

    if "label metadata" in info.keys():
        labels_key = info["label metadata"]
    else:
        labels_key = None

    save_results(results_name, x_data=x_train, wavelengths=info["wavelengths"], y_data=y_true, y_predicted=y_pred,
                 metadata=meta, model_info=info["model info"], labels_key=labels_key, metadata_key=metadata_key)

    # one can get model info with
    # data = np.load("".join((_path_accuracy_test, results_name)), allow_pickle=True)
    # model_info = data["model info"][()]

    if taxonomical:
        plot_confusion_matrix(y_true, y_pred, labels=np.array(list(classes.keys())),
                              quiet=False, suf="".join(("_", model_grid, "_accuracy_test")))

        accuracy = np.mean(metrics[0](y_true, y_pred))

        print("-----------------------------------------------------")
        print("{:20s}".format("Test accuracy:"), str("{:7.5f}").format(np.round(accuracy, 5)))
        print("-----------------------------------------------------")

    else:
        plot_scatter_plots(y_true, y_pred, used_minerals=minerals_used, used_endmembers=endmembers_used,
                           pure_only=use_pure_only, mix_of_the_pure_ones=use_mix_of_the_pure_ones,
                           quiet=False, suf="".join(("_", model_grid, "_accuracy_test")))
        plot_error_evaluation(y_true, y_pred, used_minerals=minerals_used, used_endmembers=endmembers_used,
                              quiet=False, suf="".join(("_", model_grid, "_accuracy_test")))
        plot_error_density_plots(y_true, y_pred, used_minerals=minerals_used, used_endmembers=endmembers_used,
                                 quiet=False, suf="".join(("_", model_grid, "_accuracy_test")))

        accuracy = my_rmse(num_minerals, all_to_one=False)(y_true, y_pred).numpy()

        print("-----------------------------------------------------")
        print_accuracy_header(minerals_used, endmembers_used)
        print_accuracy(accuracy, method, minerals_used, endmembers_used, endmembers_counts)
        print("-----------------------------------------------------")
