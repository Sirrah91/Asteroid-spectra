from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.callbacks import TerminateOnNaN, ReduceLROnPlateau, ModelCheckpoint
from modules.NN_callbacks import ReturnBestEarlyStopping
from pprint import pprint
import json
import warnings
from os import path, walk
from typing import Literal
from itertools import chain

from keras_tuner.tuners import BayesianOptimization, RandomSearch
import keras_tuner as kt
from tensorflow.keras.models import Model

from modules.NN_models import MyHyperModel

from modules.control_plots import plot_model_history, plot_corr_matrix, plot_model_layer
from modules.utilities_spectra import gimme_model_specification, print_header, print_info, gimme_bin_code_from_name, \
    load_txt
from modules.utilities import check_dir, is_empty, sort_df_with_keys

from modules.NN_HP import gimme_hyperparameters

from modules._constants import (_path_model, _path_hp_tuning, _model_suffix, _sep_out, _sep_in, _quiet, _verbose,
                                _show_control_plot)

# defaults only
from modules.NN_config_composition import comp_model_setup
from modules.NN_config_taxonomy import tax_model_setup


def train(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
          params: dict[str, str | int | float | bool | list[int]],
          monitoring: dict | None = None,
          model_subdir: str | None = None,
          model_name: str | None = None,
          metrics: list[str] | None = None
          ) -> str:
    # loading needed values
    if params["model_usage"] == "taxonomy":
        if monitoring is None: monitoring = tax_model_setup["monitoring"]
        if model_subdir is None: model_subdir = tax_model_setup["model_subdir"]
        if model_name is None: model_name = tax_model_setup["model_name"]
        if metrics is None: metrics = tax_model_setup["metrics"]

    else:
        if monitoring is None: monitoring = comp_model_setup["monitoring"]
        if model_subdir is None: model_subdir = comp_model_setup["model_subdir"]
        if model_name is None: model_name = comp_model_setup["model_name"]
        if metrics is None: metrics = comp_model_setup["metrics"]

    if not _quiet:
        # empty line after loading the data
        print("\nSetting up the neural network")

    bin_code = gimme_bin_code_from_name(model_name=model_name)

    # Define model architecture
    if params["model_type"] in ["MLP", "CNN"]:
        hypermodel = MyHyperModel(input_shape_MLP=(np.shape(x_train)[1],),
                                  params=params,
                                  metrics=metrics,
                                  bin_code=bin_code)
        model = hypermodel.build(kt.HyperParameters())
    else:
        raise NameError('unknown p["model_type"]')

    if "accuracy_test" in model_subdir or "range_test" in model_subdir:
        show_control_plot, verbose = False, 0
    else:
        show_control_plot, verbose = _show_control_plot, _verbose

    model_name, model_filename = gimme_model_name(params["model_usage"], model_subdir, model_name)
    check_dir(model_filename)

    model = fit_model(model, x_train, y_train, x_val, y_val, params, monitoring, model_filename, verbose)

    # Save model to project dir with a timestamp
    if path.isfile(model_filename):
        # Model weights were saved by ModelCheckpoint; restore the best one here
        model.load_weights(model_filename)
    # else Model weights were set by EarlyStopping

    # save the model here
    model.save(model_filename)

    if not _quiet:
        print("Model was saved to disk")

    if show_control_plot:
        plot_model_history(model, quiet=_quiet)
        plot_model_layer(model_name, subfolder_model=model_subdir, layer="Conv1D", suf=f"{_sep_out}kernels",
                         quiet=_quiet)

    if not _quiet:
        print_header(bin_code=bin_code)
        print_info(y_train, model.predict(x_train, verbose=0), bin_code=bin_code, which="train")
        if not is_empty(y_val):
            print_info(y_val, model.predict(x_val, verbose=0), bin_code=bin_code, which="validation")

    return model_name


def fit_model(model: Model,
              x_train: np.ndarray, y_train: np.ndarray,
              x_val: np.ndarray, y_val: np.ndarray,
              params: dict[str, str | int | float | bool | list[int]],
              monitoring: dict[str, str],
              model_filename: str | None = None,
              verbose: int = 2) -> Model:
    # visualise the model
    # visualizer(model, filename="architecture", format="png", view=True)

    # Train model
    if verbose == 1:
        pprint(params)
        print()
        model.summary()

    if verbose > 0:
        print("Training...")

    if not is_empty(y_val):
        validation_data = (x_val, y_val)
    else:
        validation_data, monitoring["objective"] = None, monitoring["objective"].replace("val_", "")

    # parameters in early stopping and reduce LR
    stopping_patience = int(params["num_epochs"] * 0.3)
    lr_factor = 1.
    callbacks = collect_callbacks(monitoring=monitoring,
                                  model_filename=model_filename,
                                  stopping_patience=stopping_patience,
                                  reducelr_patience=stopping_patience // 2,
                                  lr_factor=lr_factor,
                                  verbose=verbose)

    model.fit(x_train, y_train, validation_data=validation_data, epochs=params["num_epochs"],
              batch_size=params["batch_size"], validation_batch_size=len(y_val), shuffle=True,
              callbacks=callbacks, verbose=verbose)

    return model


def collect_callbacks(monitoring: dict[str, str],
                      model_filename: str | None = None,
                      stopping_patience: int = 0,
                      reducelr_patience: int = 0,
                      lr_factor: float = 1.,
                      verbose: int = 2) -> list:
    callbacks = [TerminateOnNaN()]

    if model_filename is not None:  # good backup if something happens during training
        checkpoint = ModelCheckpoint(model_filename, monitor=monitoring["objective"], mode=monitoring["direction"],
                                     save_best_only=True, save_weights_only=True, verbose=verbose)
        callbacks.append(checkpoint)

    if stopping_patience > 0:
        # Set early stopping monitor so the model will stop training if it does not improve anymore
        early_stopping_monitor = ReturnBestEarlyStopping(monitor=monitoring["objective"], mode=monitoring["direction"],
                                                         patience=stopping_patience, restore_best_weights=True,
                                                         verbose=verbose)
        callbacks.append(early_stopping_monitor)

    if 0. < lr_factor < 1. and reducelr_patience > 0:
        # set reduction learning rate callback
        reduce_lr = ReduceLROnPlateau(monitor=monitoring["objective"], mode=monitoring["direction"], factor=lr_factor,
                                      patience=reducelr_patience, min_lr=0., verbose=verbose)
        callbacks.append(reduce_lr)

    return callbacks


def gimme_model_name(model_usage: Literal["composition", "taxonomy"], model_subdir: str | None,
                     model_name: str | None) -> tuple[str, str]:
    if model_usage == "taxonomy":
        if model_subdir is None: model_subdir = tax_model_setup["model_subdir"]
        if model_name is None: model_name = tax_model_setup["model_name"]

    else:
        if model_subdir is None: model_subdir = comp_model_setup["model_subdir"]
        if model_name is None: model_name = comp_model_setup["model_name"]

    model_name = model_name.replace(".", "_")  # there cannot be . in model_name

    dt_string = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    final_suffix = "" if _model_suffix == "SavedModel" else f".{_model_suffix}"
    model_name_full = f"{model_name}{_sep_out}{dt_string}{final_suffix}"

    filename = path.join(_path_model, model_subdir, model_name_full)

    return model_name_full, filename


def hp_tuner(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
             composition_or_taxonomy: Literal["composition", "taxonomy"],
             monitoring: dict | None = None,
             model_subdir: str | None = None,
             model_name: str | None = None,
             metrics: list[str] | None = None
             ) -> list[str]:
    # loading needed values
    if composition_or_taxonomy == "taxonomy":
        if monitoring is None: monitoring = tax_model_setup["monitoring"]
        if model_subdir is None: model_subdir = tax_model_setup["model_subdir"]
        if model_name is None: model_name = tax_model_setup["model_name"]
        if metrics is None: metrics = tax_model_setup["metrics"]

    elif composition_or_taxonomy == "composition":
        if monitoring is None: monitoring = comp_model_setup["monitoring"]
        if model_subdir is None: model_subdir = comp_model_setup["model_subdir"]
        if model_name is None: model_name = comp_model_setup["model_name"]
        if metrics is None: metrics = comp_model_setup["metrics"]

    else:
        raise ValueError('"composition_or_taxonomy" must be "composition" or " taxonomy"')

    params = gimme_hyperparameters(for_tuning=True)(composition_or_taxonomy)

    bin_code = gimme_bin_code_from_name(model_name=model_name)

    method = params["tuning_method"]

    # parameters in early stopping and reduce LR
    stopping_patience = int(params["num_epochs"] * 0.5)
    lr_factor = 1.
    callbacks = collect_callbacks(monitoring=monitoring,
                                  stopping_patience=stopping_patience,
                                  reducelr_patience=stopping_patience // 2,
                                  lr_factor=lr_factor,
                                  verbose=2)

    N = 5  # save N best models (if N < max_trials, N = max_trials)

    # input_shape_CNN is computed from this one
    hypermodel = MyHyperModel(input_shape_MLP=(np.shape(x_train)[1],),
                              params=params,
                              metrics=metrics,
                              bin_code=bin_code)

    dt_string = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    model_type = model_subdir[:model_subdir.find(path.sep)]

    model_specification = gimme_model_specification(model_name)

    project_name = f"{model_type}{_sep_out}{model_specification}{_sep_out}{dt_string}"
    directory = path.join(_path_hp_tuning, method)

    if method == "Bayes":
        max_trials = 300

        tuner = BayesianOptimization(hypermodel,
                                     objective=kt.Objective(monitoring["objective"], direction=monitoring["direction"]),
                                     max_trials=max_trials,
                                     beta=5.,
                                     num_initial_points=np.round(max_trials * 2. / 3.),
                                     directory=directory,
                                     project_name=project_name)
    elif method == "Random":
        max_trials = 1500

        tuner = RandomSearch(hypermodel,
                             objective=kt.Objective(monitoring["objective"], direction=monitoring["direction"]),
                             max_trials=max_trials,
                             directory=directory,
                             project_name=project_name)
    else:
        raise ValueError('Unknown method. Must be "Bayes" or "Random"')

    # tuner.search_space_summary()

    tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=params["num_epochs"], shuffle=True,
                 validation_batch_size=len(y_val), callbacks=callbacks, verbose=2)

    # write results to a file
    filename = path.join(directory, f"{project_name}.csv")
    check_dir(filename)

    trials = tuner.oracle.get_best_trials(num_trials=max_trials)
    metrics_all = list(trials[0].metrics.get_config()["metrics"].keys())

    # put metric and val_metric next to each other
    metrics_all = [metric for metric in metrics_all if "val_" in metric]
    metrics_all = list(chain.from_iterable((metric.replace("val_", ""), metric) for metric in metrics_all))

    df = pd.DataFrame(trial.hyperparameters.get_config()["values"] |
                      {metric: trial.metrics.get_last_value(metric) for metric in metrics_all}
                      for trial in trials)

    # sort keys before saving
    units = [f"num_units_{i}" for i in range(np.max(df["num_layers"]))]
    filters = [f"num_filters_{i}" for i in range(np.max(df["num_layers"]))]

    order = [*metrics_all, "model_type", "num_layers", *units, *filters, "kernel_size", "kernel_padding",
             "input_activation", "output_activation",
             "dropout_input_hidden", "dropout_hidden_hidden", "dropout_hidden_output",
             "L1_trade_off", "L2_trade_off", "max_norm", "optimizer", "learning_rate",
             "batch_size", "batch_norm_before_activation", "alpha", "use_weights"]

    df = sort_df_with_keys(df, order)
    df.to_csv(filename, sep="\t", index=False, na_rep="N/A")

    # Retrieve the best model
    with warnings.catch_warnings():  # This does not help to suppress the warnings
        warnings.simplefilter("ignore")
        best_model = tuner.get_best_models(num_models=np.min((N, max_trials)))  # raises warnings

    final_suffix = "" if _model_suffix == "SavedModel" else f".{_model_suffix}"
    model_names = [f"{model_name}{_sep_out}{dt_string}{_sep_out}Tuner{_sep_in}{i}{final_suffix}" for i in
                   range(len(best_model))]

    for name, model in zip(model_names, best_model):
        # Save top models to project dir with a timestamp
        filename = path.join(_path_model, model_subdir, name)
        check_dir(filename)
        model.save(filename)

    print("Models were saved to disk")

    if params["plot_corr_mat"]:
        do_corr_plots(df=df, method=method)

    return model_names


def prepare_hp_for_corr_plot(hp_dirname: str, method: str) -> None:
    file = path.join(_path_hp_tuning, method, f"{hp_dirname}.csv")

    if path.isfile(file):
        data = load_txt(file, sep="\t")

        # remove other keys and put metric and val_metric next to each other
        metrics_all = list(data.keys())
        metrics_all = [metric for metric in metrics_all if "val_" in metric]
        metrics_all = list(chain.from_iterable((metric.replace("val_", ""), metric) for metric in metrics_all))

    else:
        metrics_all = []

    if "loss" not in metrics_all:  # if loss is not there, the data is probably unstructured; read it from trials
        folder = path.join(_path_hp_tuning, method, hp_dirname)
        files = np.sort(next(walk(folder))[1])

        max_trials = len(files)
        hps = max_trials * [0]

        # to get keys
        json_name = lambda x: path.join(folder, f"{files[x]}", "trial.json")

        with open(json_name(0), "r") as f:
            stored_json = json.load(f)

        metrics_all = list(stored_json["metrics"]["metrics"].keys())

        # put metric and val_metric next to each other
        metrics_all = [metric for metric in metrics_all if "val_" in metric]
        metrics_all = list(chain.from_iterable((metric.replace("val_", ""), metric) for metric in metrics_all))

        loss_metrics = np.zeros((max_trials, len(metrics_all)))

        for i in range(max_trials):
            with open(json_name(i), "r") as f:
                stored_json = json.load(f)
            hps[i] = stored_json["hyperparameters"]["values"]
            for j, keys in enumerate(metrics_all):
                loss_metrics[i, j] = stored_json["metrics"]["metrics"][keys]["observations"][0]["value"][0]

        data = pd.DataFrame(hps)

        for i, keys in enumerate(metrics_all):
            data[keys] = loss_metrics[:, i]

    units = [f"num_units_{i}" for i in range(np.max(data["num_layers"]))]
    filters = [f"num_filters_{i}" for i in range(np.max(data["num_layers"]))]

    order = [*metrics_all, "model_type", "num_layers", *units, *filters, "kernel_size", "kernel_padding",
             "input_activation", "output_activation",
             "dropout_input_hidden", "dropout_hidden_hidden", "dropout_hidden_output",
             "L1_trade_off", "L2_trade_off", "max_norm", "optimizer", "learning_rate",
             "batch_size", "batch_norm_before_activation", "alpha", "use_weights"]

    data = sort_df_with_keys(data, order)

    do_corr_plots(df=data, method=method)


def do_corr_plots(df: pd.DataFrame, method: str = "Unknown") -> None:
    for model_type in np.unique(df["model_type"]):
        where_models = np.array(df["model_type"] == model_type)

        # construct the correlation matrix only if more than one realisation were computed
        if np.sum(where_models) > 1:
            data_part = df[where_models]

            # remove constant columns and single non-NaN columns
            data_part = data_part.loc[:, data_part.apply(pd.Series.nunique) > 1]

            data_corr = data_part.corr(numeric_only=True)
            data_keys = np.array(data_corr.keys(), dtype=str)

            plot_corr_matrix(data_keys, data_corr, suf=f"{_sep_out}Tuner{_sep_out}{method}{_sep_out}{model_type}")
