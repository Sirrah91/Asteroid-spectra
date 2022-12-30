import numpy as np
from keras.models import load_model
from scipy.stats import trim_mean
from typing import Callable

from modules.NN_losses_metrics_activations import custom_objects

from modules.utilities import normalise_in_rows
from modules.utilities_spectra import print_accuracy, print_accuracy_header, gimme_indices

from modules.NN_config import num_minerals, endmembers_counts, minerals_used, endmembers_used

from modules._constants import _path_data, _path_model


def average_and_normalise(predictions: np.ndarray, per_partes: bool) -> np.ndarray:
    # loading needed values
    if per_partes:
        from modules.NN_config import trim_mean_cut
    else:
        from modules.NN_config_taxonomy import trim_mean_cut

    # Trimmed mean
    predictions = trim_mean(predictions, trim_mean_cut, axis=2)

    # Normalisations to 1
    if per_partes:
        for start, stop in gimme_indices(num_minerals, endmembers_counts):
            norm = np.sum(predictions[:, start:stop], axis=1)
            predictions[:, start:stop] = normalise_in_rows(predictions[:, start:stop], norm)
    else:
        norm = np.sum(predictions, axis=1)
        predictions = normalise_in_rows(predictions, norm)

    return np.array(predictions, dtype=np.float32)


def evaluate(model_names: list[str], filename_or_data: str | np.ndarray,
             subfolder_model: str = "compositional") -> np.ndarray:
    # This function evaluate the mean model on new a dataset

    if not model_names:
        raise ValueError('"model_names" is empty')

    # loading needed values
    if "taxonomical" in subfolder_model:
        from modules.NN_config_taxonomy import verb
    else:
        from modules.NN_config import verb

    quiet = verb == 0

    if isinstance(filename_or_data, str):
        # Import the test dataset
        if not quiet:
            print("Loading dataset")

        data_file = "".join((_path_data, filename_or_data))
        data = np.load(data_file, allow_pickle=True)
        data = data["spectra"]
    else:
        data = np.array(filename_or_data)

    if np.ndim(data) == 1:
        data = np.reshape(data, (1, len(data)))
    data = np.array(data, dtype=np.float32)

    if not quiet:
        print("Evaluating the neural network")

    # Calc average prediction over the models
    for idx, model_name in enumerate(model_names):
        model = load_model("".join((_path_model, subfolder_model, "/", model_name)),
                           custom_objects=custom_objects)

        if idx == 0:
            predictions = np.zeros((len(data), model.output_shape[1], len(model_names)))

        # Evaluate model on test data
        predictions[:, :, idx] = model.predict(data, verbose=0)

    # Trimmed means and normalisations to 1
    predictions = average_and_normalise(predictions, "taxonomical" not in subfolder_model)
    print("-----------------------------------------------------")

    return predictions


def evaluate_test_data(model_names: list[str], x_test: np.ndarray, y_test: np.ndarray,
                       x_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
                       x_train: np.ndarray | None = None, y_train: np.ndarray | None = None,
                       subfolder_model: str = "compositional") -> tuple[np.ndarray, ...]:
    if not model_names:
        raise ValueError('"model_names" is empty')

    # loading needed values
    if "taxonomical" in subfolder_model:
        from modules.NN_config_taxonomy import verb, show_result_plot, show_control_plot

    elif "compositional" in subfolder_model or "accuracy_test" in subfolder_model:
        # the same options in NN_config and NN_config_taxonomy
        from modules.NN_config import verb, show_result_plot, show_control_plot
    else:  # "range_test"
        from modules.NN_config_range_test import verb, show_result_plot, show_control_plot

    from modules.NN_config_taxonomy import metrics  # needed to distinguish between taxonomical and compositional

    quiet = verb == 0

    if not quiet:
        print("Evaluating the neural network on the test data")
    predictions = np.zeros((*np.shape(y_test), len(model_names)))

    do_train, do_val = y_train is not None and len(y_train) > 0, y_val is not None and len(y_val) > 0

    if do_train:
        predictions_train = np.zeros((*np.shape(y_train), len(model_names)))
    if do_val:
        predictions_val = np.zeros((*np.shape(y_val), len(model_names)))

    # Calc average prediction over the models
    for idx, model_name in enumerate(model_names):
        model = load_model("".join((_path_model, subfolder_model, "/", model_name)),
                           custom_objects=custom_objects)

        # Evaluate model on test data
        predictions[:, :, idx] = model.predict(x_test, verbose=0)
        if do_train:
            predictions_train[:, :, idx] = model.predict(x_train, verbose=0)
        if do_val:
            predictions_val[:, :, idx] = model.predict(x_val, verbose=0)

        if idx == 0:
            taxonomical = np.any([metrics[0].__name__ in metric.name for metric in model.metrics])

    # Trimmed means and normalisations to 1
    predictions = average_and_normalise(predictions, not taxonomical)
    if do_train:
        predictions_train = average_and_normalise(predictions_train, not taxonomical)
    if do_val:
        predictions_val = average_and_normalise(predictions_val, not taxonomical)

    if not quiet:
        print(model_names)

    # these two are result plots
    if show_result_plot:
        print("Result plots")
        _result_plots(y_test, predictions, taxonomical)

    if show_control_plot:
        print("Control plots")
        if do_val:
            _control_plots(y_val, predictions_val, taxonomical)
        if do_train:
            _control_plots(y_train, predictions_train, taxonomical)

    # Evaluate the accuracy (this is always printed)
    print("\n-----------------------------------------------------")

    if taxonomical:
        metric = metrics[0]

    else:
        from modules.NN_losses_metrics_activations import my_rmse

        print_accuracy_header(minerals_used, endmembers_used)
        metric = my_rmse(num_minerals, all_to_one=False)

    if do_train:
        _print_info(metric, y_train, predictions_train, taxonomical, "train")
    if do_val:
        _print_info(metric, y_val, predictions_val, taxonomical, "validation")
    acc = _print_info(metric, y_test, predictions, taxonomical, "test")

    print("-----------------------------------------------------\n")

    return predictions, acc


def gimme_suf(num_labels: int) -> str:
    # ad-hoc function for names of scatter plots for paper

    if num_labels == 2:
        return "_pure"

    if num_labels == 6:
        return "_mix"

    if num_labels == 11:
        return "_all1"

    if num_labels == 15:
        return "_all2"

    return ""


def _result_plots(y_true: np.ndarray, y_pred: np.ndarray, taxonomical: bool) -> None:
    quiet = True

    if taxonomical:
        from modules.control_plots import plot_confusion_matrix
        from modules.NN_config_taxonomy import classes

        plot_confusion_matrix(y_true, y_pred, labels=np.array(list(classes.keys())), quiet=quiet)

    else:
        from modules.control_plots import plot_scatter_plots, plot_error_evaluation
        from modules.NN_config import use_pure_only, use_mix_of_the_pure_ones, minerals_used, endmembers_used

        plot_scatter_plots(y_true, y_pred, used_minerals=minerals_used, used_endmembers=endmembers_used,
                           pure_only=use_pure_only, mix_of_the_pure_ones=use_mix_of_the_pure_ones, quiet=quiet)
        plot_error_evaluation(y_true, y_pred, used_minerals=minerals_used, used_endmembers=endmembers_used,
                              quiet=quiet)


def _control_plots(y_true: np.ndarray, y_pred: np.ndarray, taxonomical: bool) -> None:
    quiet = True

    if taxonomical:
        from modules.control_plots import plot_confusion_matrix
        from modules.NN_config_taxonomy import classes

        plot_confusion_matrix(y_true, y_pred, labels=np.array(list(classes.keys())), suf="_val", quiet=quiet)

    else:
        from modules.control_plots import plot_scatter_plots, plot_error_evaluation
        from modules.NN_config import use_pure_only, use_mix_of_the_pure_ones, minerals_used, endmembers_used

        plot_scatter_plots(y_true, y_pred, used_minerals=minerals_used, used_endmembers=endmembers_used,
                           pure_only=use_pure_only, mix_of_the_pure_ones=use_mix_of_the_pure_ones,
                           quiet=quiet, suf="_val")


def _print_info(metric: Callable, y_true: np.ndarray, y_pred: np.ndarray, taxonomical: bool, which: str) -> np.ndarray:
    if taxonomical:
        acc = np.mean(metric(y_true, y_pred).numpy())
        print("{:20s}".format("".join((which.capitalize(), " accuracy:"))), str("{:7.5f}").format(np.round(acc, 5)))

    else:
        acc = metric(y_true, y_pred).numpy()
        print_accuracy(acc, which, minerals_used, endmembers_used, endmembers_counts)

    return acc
