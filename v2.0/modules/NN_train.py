import numpy as np
import pandas as pd
from datetime import datetime
from keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau
from pprint import pprint
import json
import os
from typing import Literal

from keras_tuner.tuners import BayesianOptimization, RandomSearch
from contextlib import redirect_stdout
import keras_tuner as kt
from keras.models import Functional

from modules.control_plots import plot_model_history, plot_corr_matrix, plot_conv_kernels
from modules.utilities import check_dir

from modules.NN_models_taxonomy import metrics
from modules.NN_losses_metrics_activations import my_rmse

from modules.NN_HP import gimme_hyperparameters

from modules._constants import _path_model, _path_hp_tuning


def train(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
          params: dict[str, str | int | float | bool | list[int]]) -> str:

    # loading needed values
    if params["model_usage"] == "taxonomical":
        from modules.NN_models_taxonomy import MLP_model, CNN_model, objective, direction
        from modules.NN_config_taxonomy import show_control_plot, verb, model_subdir
    else:  # compositional is default
        from modules.NN_models import MLP_model, CNN_model
        from modules.NN_losses_metrics_activations import objective, direction
        from modules.NN_config import show_control_plot, verb, model_subdir


    quiet = verb == 0

    if not quiet:
        # empty line after loading the data
        print("\nSetting up the neural network")

    # Define model architecture
    if params["model_type"] == "MLP":  # fully connected
        model = MLP_model((x_train.shape[1],), params)
    elif params["model_type"] == "CNN":  # convolutional
        model = CNN_model((x_train.shape[1], 1), params)
    else:
        raise NameError('unknown p["model_type"]')

    monitoring = {"objective": objective, "direction": direction}
    model = fit_model(model, x_train, y_train, x_val, y_val, params, monitoring, verb)

    # Save model to project dir with a timestamp
    model_name, filename = gimme_model_name(params["model_usage"])
    check_dir(filename)
    model.save(filename)

    if not quiet:
        print("Model was saved to disk")

    if show_control_plot:
        plot_model_history(model, quiet=quiet)
        plot_conv_kernels(model_name, subfolder_model=model_subdir, layer="Conv1D", quiet=quiet)

    if not quiet:
        _print_info(model, x_train, y_train, x_val, y_val, params["model_usage"])

    return model_name


def fit_model(model: Functional, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
              params: dict[str, str | int | float | bool | list[int]], monitoring: dict[str, str],
              verbose: int = 2) -> Functional:
    # visualise the model
    # visualizer(model, filename="architecture", format="png", view=True)

    # Train model
    if verbose == 1:
        pprint(params)
        print()
        model.summary()

    if verbose > 0:
        print("Training...")

    if np.size(y_val) > 0:
        validation_data = (x_val, y_val)
    else:
        validation_data, monitoring["objective"] = None, monitoring["objective"].replace("val_", "")

    # parameters in early stopping and reduce LR
    coef, factor = 0.3, 1.
    callbacks = collect_callbacks(monitoring=monitoring, patience=params["num_epochs"] * coef, factor=factor)

    model.fit(x_train, y_train, validation_data=validation_data, epochs=params["num_epochs"],
              batch_size=params["batch_size"], validation_batch_size=len(y_val), shuffle=True,
              callbacks=callbacks, verbose=verbose)

    return model


def collect_callbacks(monitoring: dict[str, str], patience: float = 0., factor: float = 1.) -> list:
    callbacks = [TerminateOnNaN()]

    if patience > 0.:
        # Set early stopping monitor so the model will stop training if it does not improve anymore
        early_stopping_monitor = EarlyStopping(monitor=monitoring["objective"], mode=monitoring["direction"],
                                               patience=patience, restore_best_weights=True)
        callbacks.append(early_stopping_monitor)

    if 0. < factor < 1.:
        # set reduction learning rate callback
        reduce_lr = ReduceLROnPlateau(monitor=monitoring["objective"], mode=monitoring["direction"], factor=factor,
                                      patience=50, min_lr=0.)
        callbacks.append(reduce_lr)

    return callbacks


def _print_info(model: Functional, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                model_usage: Literal["compositional", "taxonomical"]) -> None:
    print("\nEvaluation on the data:")

    if model_usage == "taxonomical":
        from modules.NN_config_taxonomy import val_portion

        # Evaluate the train data
        acc = np.mean(metrics[0](y_train, model.predict(x_train, verbose=0)))
        print("{:20s}".format("Train accuracy:"), str("{:7.5f}").format(np.round(acc, 5)))

        if val_portion > 0:
            # Evaluate the validation data
            acc = np.mean(metrics[0](y_val, model.predict(x_val, verbose=0)))
            print("{:20s}".format("Validation accuracy:"), str("{:7.5f}").format(np.round(acc, 5)))

    else:
        from modules.NN_config import val_portion
        from modules.utilities_spectra import print_accuracy_header, print_accuracy
        from modules.NN_config import num_minerals, minerals_used, endmembers_used, endmembers_counts

        print_accuracy_header(minerals_used, endmembers_used)
        # Evaluate the train data
        acc = my_rmse(num_minerals, all_to_one=False)(y_train, model.predict(x_train, verbose=0)).numpy()
        print_accuracy(acc, "train", minerals_used, endmembers_used, endmembers_counts)

        if val_portion > 0:
            # Evaluate the validation data
            acc = my_rmse(num_minerals, all_to_one=False)(y_val, model.predict(x_val, verbose=0)).numpy()
            print_accuracy(acc, "validation", minerals_used, endmembers_used, endmembers_counts)

    print()  # Empty line


def gimme_model_name(model_usage: Literal["compositional", "taxonomical"]) -> tuple[str, str]:
    if model_usage == "taxonomical":
        from modules.NN_config_taxonomy import model_subdir, model_name_suffix
    else:
        from modules.NN_config import model_subdir, model_name_suffix

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    model_name = "".join((dt_string, "_", model_name_suffix, ".h5"))
    filename = "".join((_path_model, model_subdir, "/", model_name))

    return model_name, filename


def hp_tuner(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
             compositional_or_taxonomical: Literal["compositional", "taxonomical"]) -> list[str]:

    # loading needed values
    if compositional_or_taxonomical == "compositional":
        from modules.NN_models import MyHyperModel, metrics
        from modules.NN_losses_metrics_activations import objective, direction
        from modules.NN_config import model_subdir, model_name_suffix, model_grid
    elif compositional_or_taxonomical == "taxonomical":
        from modules.NN_models_taxonomy import MyHyperModel, objective, direction, metrics
        from modules.NN_config_taxonomy import model_subdir, model_name_suffix, model_grid
    else:
        raise ValueError('"compositional_or_taxonomical" must be "compositional" or " taxonomical"')

    params = gimme_hyperparameters(for_tuning=True)()

    method = params["tuning_method"]

    # parameters in early stopping and reduce LR
    coef, factor = 0.5, 1.
    monitoring = {"objective": objective, "direction": direction}
    callbacks = collect_callbacks(monitoring=monitoring, patience=params["num_epochs"] * coef, factor=factor)

    N = 5  # save N best models (if N < max_trials, N = max_trials)

    # input_shape_CNN is computed from this one
    hypermodel = MyHyperModel(input_shape_MLP=(x_train.shape[1],), params=params)

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    project_name = "_".join((dt_string, model_subdir, model_grid))

    directory = "".join((_path_hp_tuning, method))

    if method == "Bayes":
        max_trials = 300

        tuner = BayesianOptimization(hypermodel,
                                     objective=kt.Objective(objective, direction=direction),
                                     max_trials=max_trials,
                                     beta=5,
                                     num_initial_points=np.round(max_trials * 2. / 3.),
                                     directory=directory,
                                     project_name=project_name)
    elif method == "Random":
        max_trials = 1000

        tuner = RandomSearch(hypermodel,
                             objective=kt.Objective(objective, direction=direction),
                             max_trials=max_trials,
                             directory=directory,
                             project_name=project_name)
    else:
        raise ValueError('Unknown method. Must be "Bayes" or "Random"')

    # tuner.search_space_summary()

    tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=params["num_epochs"], shuffle=True,
                 validation_batch_size=len(y_val), callbacks=callbacks, verbose=2)

    # write results to a file (file does not look nice)
    filename = "".join((directory, "/", project_name, ".csv"))
    check_dir(filename)
    with open(filename, "w") as f:
        with redirect_stdout(f):
            tuner.results_summary(max_trials)

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=np.min((N, max_trials)))  # raise warnings
    model_names = ["".join((dt_string, "_", model_name_suffix, "_Tuner_", str(i), ".h5"))
                   for i in range(len(best_model))]

    for name, model in zip(model_names, best_model):
        # Save top models to project dir with a timestamp
        filename = "".join((_path_model, model_subdir, "/", name))
        check_dir(filename)
        model.save(filename)

    if params["plot_corr_mat"]:
        prepare_hp_for_corr_plot(project_name, method, metrics)

    return model_names


def prepare_hp_for_corr_plot(hp_dirname: str, method: str, metrics_list: list) -> None:
    folder = "".join((_path_hp_tuning, method, "/", hp_dirname, "/"))

    max_trials = len(next(os.walk(folder))[1])
    hps = max_trials * [0]
    loss_metrics = np.zeros((max_trials, 1 + len(metrics_list)))

    # to get keys
    json_name = lambda x: "".join((folder, "trial_", "".join(("{:0", f"{len(str(max_trials))}", "d}")).format(x),
                                   "/trial.json"))
    with open(json_name(0)) as f:
        stored_json = json.load(f)
    keys_loss_metrics = np.array(list(stored_json["metrics"]["metrics"].keys()))
    val_keys = ["val_" in key for key in keys_loss_metrics]
    keys_loss_metrics = keys_loss_metrics[val_keys]

    for i in range(max_trials):
        with open(json_name(i)) as f:
            stored_json = json.load(f)
            hps[i] = stored_json["hyperparameters"]["values"]
            for j, keys in enumerate(keys_loss_metrics):
                loss_metrics[i, j] = stored_json["metrics"]["metrics"][keys]["observations"][0]["value"][0]

    data = pd.DataFrame(hps)

    for i, keys in enumerate(keys_loss_metrics):
        data[keys] = loss_metrics[:, i]

    order = {"val_loss": 0}

    met = {"".join(("val_", metric.__name__)): i + 1 for i, metric in enumerate(metrics_list)}
    order = dict(order, **met)

    order["model_type"] = len(order)
    order["num_layers"] = len(order)

    units = np.sort(np.array(data.keys())[["num_units" in i for i in np.array(data.keys())]])
    units = {m: i + len(order) for i, m in enumerate(units)}
    order = dict(order, **units)

    filters = np.sort(np.array(data.keys())[["num_filters" in i for i in np.array(data.keys())]])
    filters = {m: i + len(order) for i, m in enumerate(filters)}
    order = dict(order, **filters)

    if "kernel_size" in data.keys():
        order["kernel_size"] = len(order)

    order["input_activation"] = len(order)
    order["output_activation"] = len(order)

    if "dropout_input_hidden" in data.keys():
        order["dropout_input_hidden"] = len(order)
    order["dropout_hidden_hidden"] = len(order)
    order["dropout_hidden_output"] = len(order)

    order["L1_trade_off"] = len(order)
    order["L2_trade_off"] = len(order)
    order["max_norm"] = len(order)

    order["optimizer"] = len(order)
    order["learning_rate"] = len(order)
    order["batch_size"] = len(order)
    order["batch_norm_before_activation"] = len(order)

    if "alpha" in data.keys():
        order["alpha"] = len(order)

    data = data[list(order.keys())]

    for model_type in ["MLP", "CNN"]:
        where_models = np.array(data["model_type"] == model_type)

        # construct the correlation matrix only if more than one realisation were computed
        if np.sum(where_models) > 1:
            data_part = data.iloc[:, np.where(data[where_models].sum())[0]]  # remove non-used HP
            data_corr = data_part.corr(numeric_only=True)
            data_keys = np.array(data_corr.keys(), dtype=str)

            plot_corr_matrix(data_keys, data_corr, "".join(("_Tuner_", method, "_", model_type)))
