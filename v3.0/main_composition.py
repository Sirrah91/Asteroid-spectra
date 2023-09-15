# ----------------------------------------------------------------------- #
# Neural Network to Classify Asteroids Using Keras and Tensorflow Backend #
# by David Korda (david.korda@helsinki.fi)                                #
# ----------------------------------------------------------------------- #
# Run with python 3.10.6                                                  #
# Install: requirements.txt                                               #
# pip install -r requirements.txt                                         #
# ----------------------------------------------------------------------- #

"""
This code is provided under MIT licence (https://opensource.org/license/mit/).

Copyright 2023 David Korda and Tomáš Kohout (University of Helsinki and Institute of Geology of the Czech
Academy of Sciences).

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from modules.NN_data import load_composition_data as load_data
from modules.NN_data import split_composition_data_proportional as split_data_proportional
from modules.NN_train import train, hp_tuner
from modules.NN_evaluate import evaluate_test_data, evaluate
import numpy as np
from tqdm import tqdm

from modules.utilities_spectra import collect_all_models
from modules.NN_config_composition import (comp_model_setup, minerals_used, endmembers_used, comp_grid,
                                           comp_filtering_setup, comp_data_split_setup)
from modules._constants import _sep_in, _sep_out

train_new_model = True  # If you have a trained model, just run evaluate(model_names, filename_data_or_data)
tune_hyperparameters = False  # if you just want to tune hp


def pipeline(num_models: int = 1) -> np.ndarray:
    # pipeline(num_models) computes a trimmed mean of num_models and returns predictions or print best hyperparameters

    model_subdir, model_name = comp_model_setup["model_subdir"], comp_model_setup["model_name"]

    if train_new_model or tune_hyperparameters:
        # Name of the train data in _path_data
        filename_train_data = f"mineral{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz"

        # Load the data
        x_train, y_train = load_data(filename_train_data, clean_dataset=True,
                                     used_minerals=minerals_used, used_endmembers=endmembers_used,
                                     grid_setup=comp_grid, filtering_setup=comp_filtering_setup)

        # Split the data
        x_train, y_train, x_val, y_val, x_test, y_test = split_data_proportional(x_train, y_train,
                                                                                 val_portion=comp_data_split_setup["val_portion"],
                                                                                 test_portion=comp_data_split_setup["test_portion"],
                                                                                 used_minerals=minerals_used)

        if tune_hyperparameters:
            # Tuning of the hyperparameters defined in p_for_tuning dictionary
            model_names = hp_tuner(x_train, y_train, x_val, y_val, "composition",
                                   monitoring=comp_model_setup["monitoring"],
                                   model_subdir=model_subdir, model_name=model_name,
                                   metrics=comp_model_setup["metrics"])
        else:
            # Create, train, and save the neural network, evaluate it on the test data
            model_names = [train(x_train, y_train, x_val, y_val, params=comp_model_setup["params"],
                                 monitoring=comp_model_setup["monitoring"],
                                 model_subdir=model_subdir, model_name=model_name,
                                 metrics=comp_model_setup["metrics"]) for _ in range(num_models)]

        predictions, accuracy = evaluate_test_data(model_names, x_test, y_test, x_val=x_val, y_val=y_val,
                                                   x_train=x_train, y_train=y_train,
                                                   proportiontocut=comp_model_setup["trim_mean_cut"],
                                                   subfolder_model=model_subdir)
    else:
        # List of the models in ./_path_model/subfolder_model/
        model_names = collect_all_models(prefix=model_name, subfolder_model=model_subdir, full_path=False)

        if f"820{_sep_in}2080{_sep_in}20{_sep_in}1500" in model_subdir:
            filename_data = f"Itokawa{_sep_out}denoised{_sep_out}norm.npz"
        elif f"820{_sep_in}2360{_sep_in}20{_sep_in}1300" in model_subdir:
            filename_data = f"Eros{_sep_out}denoised{_sep_out}norm.npz"
        elif f"450{_sep_in}2450{_sep_in}5{_sep_in}550" in model_subdir:
            filename_data = f"mineral{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz"
        else:
            filename_data = ""

        # Evaluate the models; second argument can be a path to the data or the data itself
        predictions = evaluate(model_names, filename_data, proportiontocut=comp_model_setup["trim_mean_cut"],
                               subfolder_model=model_subdir)

    return predictions


if __name__ == "__main__":
    if tune_hyperparameters:
        pipeline()
    elif train_new_model:
        for _ in tqdm(range(10)):
            y_pred = pipeline()
    else:
        y_pred = pipeline()
