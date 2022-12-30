# ----------------------------------------------------------------------- #
# Neural Network to Classify Asteroids Using Keras and Tensorflow Backend #
# by David Korda (david.korda@helsinki.fi)                                #
# ----------------------------------------------------------------------- #
# Run with python 3.10.6                                                  #
# Install: numpy, tensorflow + dependencies, keras + dependencies         #
# ----------------------------------------------------------------------- #

"""
Copyright 2022 David Korda, Antti Penttila, Arto Klami, Tomas Kohout (University of Helsinki and Institute of Geology of
the Czech Academy of Sciences). Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the 'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of
the Software.
THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS
OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from os import environ
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from modules.NN_data import load_taxonomical_data as load_data
from modules.NN_data import split_taxonomical_data_proportional as split_data_proportional
from modules.NN_data import labels_to_categories
from modules.NN_train import train, hp_tuner
from modules.NN_evaluate import evaluate_test_data, evaluate
import numpy as np
from tqdm import tqdm

from modules.utilities_spectra import collect_all_models
from modules.paper_plots import plot_surface_spectra
from modules.NN_config_taxonomy import p, model_subdir, model_grid, model_name_suffix

train_new_model = True  # If you have a trained model, just run evaluate(model_names, filename_data_or_data)
tune_hyperparameters = False  # if you just want to tune hp


def pipeline(num_models: int = 1) -> np.ndarray:
    # pipeline(num_models) computes a trimmed mean of num_models and returns predictions or print best hyperparameters

    if model_subdir != "taxonomical":
        raise ValueError('Change "model_subdir" in NN_config_taxonomy.py to "taxonomical".')

    if train_new_model or tune_hyperparameters:
        # Name of the train data in ./Datasets/
        filename_train_data = "asteroid_spectra-reduced-denoised-norm.npz"

        # Load the data
        x_train, y_train = load_data(filename_train_data, clean_dataset=True)

        # Split the data
        x_train, y_train, x_val, y_val, x_test, y_test = split_data_proportional(x_train, y_train)

        # labels to categories
        y_train = labels_to_categories(y_train)
        y_val = labels_to_categories(y_val)
        y_test = labels_to_categories(y_test)

        if tune_hyperparameters:
            # Tuning of the hyperparameters defined in p_for_tuning dictionary
            model_names = hp_tuner(x_train, y_train, x_val, y_val, "taxonomical")
        else:
            # Create, train, and save the neural network, evaluate it on the test data
            model_names = [train(x_train, y_train, x_val, y_val, p) for _ in range(num_models)]

        predictions, accuracy = evaluate_test_data(model_names, x_test, y_test, x_val=x_val, y_val=y_val,
                                                   x_train=x_train, y_train=y_train, subfolder_model=model_subdir)
    else:
        # List of the models in ./Models/model_subdir/
        model_names = collect_all_models(suffix=model_name_suffix, subfolder_model=model_subdir, full_path=False)

    if model_grid in ["Itokawa", "Eros"]:
        filename_data = "".join((model_grid, "-denoised-norm.npz"))

        # Evaluate the models; second argument can be a path to the data or the data itself
        predictions = evaluate(model_names, filename_data, subfolder_model=model_subdir)

        plot_surface_spectra(predictions, filename_data, "taxonomy")

    return predictions


if __name__ == "__main__":
    if tune_hyperparameters:
        pipeline()
    elif train_new_model:
        for _ in tqdm(range(10)):
            y_pred = pipeline()
    else:
        y_pred = pipeline()
