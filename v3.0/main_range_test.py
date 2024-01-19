# ----------------------------------------------------------------------- #
# Neural Network to Classify Asteroids Using Keras and TensorFlow Backend #
# by David Korda (david.korda@helsinki.fi)                                #
# ----------------------------------------------------------------------- #
# Run with Python 3.10.6                                                  #
# Install: requirements.txt                                               #
# pip install -r requirements.txt                                         #
# ----------------------------------------------------------------------- #

"""
This code is provided under the MIT licence (https://opensource.org/license/mit/).

Copyright 2023 David Korda (University of Helsinki).

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

from modules.NN_data import load_composition_data, load_taxonomy_data, labels_to_categories
from modules.NN_data import split_composition_data_proportional, split_taxonomy_data_proportional
from modules.NN_train import train
from modules.NN_evaluate import evaluate_test_data
import numpy as np
from functools import partial
from tqdm import tqdm

from modules.NN_config_range_test import model_subdirs, model_names, range_grids, taxonomy

from modules.NN_config_taxonomy import tax_model_setup, tax_filtering_setup, tax_data_split_setup, classes
from modules.NN_config_composition import (comp_model_setup, comp_filtering_setup, comp_data_split_setup,
                                           minerals_used, endmembers_used)

from modules._constants import _sep_in, _sep_out

# it taxonomy = True, you must modify "NN_classes.py" and force it to return the same classes (including the bin_to_cls)
if taxonomy:
    load_data = partial(load_taxonomy_data, used_classes=classes)
    split_data_proportional = split_taxonomy_data_proportional

    model_setup = tax_model_setup
    filtering_setup = tax_filtering_setup
    data_split_setup = tax_data_split_setup

    # Name of the train data in _path_data
    filename_train_data = f"asteroid{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz"

else:
    load_data = partial(load_composition_data, used_minerals=minerals_used, used_endmembers=endmembers_used)
    split_data_proportional = partial(split_composition_data_proportional, used_minerals=minerals_used)

    model_setup = comp_model_setup
    filtering_setup = comp_filtering_setup
    data_split_setup = comp_data_split_setup

    # Name of the train data in _path_data
    filename_train_data = f"mineral{_sep_in}spectra{_sep_out}denoised{_sep_out}norm.npz"


def pipeline(index_of_range: int, num_models: int = 1) -> np.ndarray:
    # pipeline(num_models) computes a trimmed mean of num_models and returns predictions or print best hyperparameters

    model_subdir, model_name = model_subdirs[index_of_range], model_names[index_of_range]
    grid_setup = range_grids[index_of_range]

    # Load the data
    x_train, y_train = load_data(filename_train_data, clean_dataset=True,
                                 grid_setup=grid_setup, filtering_setup=filtering_setup)

    # Split the data
    x_train, y_train, x_val, y_val, x_test, y_test = split_data_proportional(x_train, y_train,
                                                                             val_portion=data_split_setup["val_portion"],
                                                                             test_portion=data_split_setup["test_portion"]
                                                                             )

    if taxonomy:
        # labels to categories
        y_train = labels_to_categories(y_train, used_classes=classes)
        y_val = labels_to_categories(y_val, used_classes=classes)
        y_test = labels_to_categories(y_test, used_classes=classes)

    # Create, train, and save the neural network
    model_names_trained = [train(x_train, y_train, x_val, y_val, params=model_setup["params"],
                                 monitoring=model_setup["monitoring"],
                                 model_subdir=model_subdir, model_name=model_name,
                                 metrics=model_setup["metrics"]) for _ in range(num_models)]

    # Evaluate it on the test data
    predictions, accuracy = evaluate_test_data(model_names_trained, x_test, y_test, x_val=x_val, y_val=y_val,
                                               x_train=x_train, y_train=y_train,
                                               proportiontocut=model_setup["trim_mean_cut"],
                                               subfolder_model=model_subdir)

    return predictions


if __name__ == "__main__":
    y_pred = np.array([pipeline(index_of_range=index) for index in tqdm(range(len(model_names)))])
