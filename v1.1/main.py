# ----------------------------------------------------------------------- #
# Neural Network to Classify Asteroids Using Keras and Tensorflow Backend #
# by David Korda (david.korda@helsinki.fi)                                #
# ----------------------------------------------------------------------- #
# Run with python 3.8.10                                                  #
# Install: numpy, tensorflow + dependencies, keras + dependencies         #
# ----------------------------------------------------------------------- #

"""
Copyright 2022 David Korda, Antti Penttil, Arto Klami, Tomas Kohout (University of Helsinki and Institute of Geology of
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
environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from modules.NN_data import load_data, split_data_proportional
from modules.NN_train import train, tune_hp_talos, tune_hp_tuner
from modules.NN_evaluate import evaluate_test_data, evaluate
import numpy as np
from typing import Union

from modules.control_plots import plot_surface_spectra
from modules.NN_config import *

train_new_model = True  # If you have a trained model, just run evaluate(model_names, filename_data_or_data)
tune_hyperparameters = False  # if you just want to tune hp


def pipeline(n_models: int = 1) -> Union[np.ndarray, None]:
    # pipeline(n_models) computes a trimmed mean of n_models and returns predictions or print best hyperparameters

    if train_new_model or tune_hyperparameters:
        # Name of the train data in ./Datasets/
        filename_train_data = 'combined-denoised-norm.dat'

        # Load the data
        x_train, y_train = load_data(filename_train_data, clean_dataset=True)

        # Split the data
        x_train, y_train, x_val, y_val, x_test, y_test = split_data_proportional(x_train, y_train)

        if tune_hyperparameters:
            # Tuning of the hyper-parameters defined in p_for_tuning dictionary
            # tune_hp_talos(x_train, y_train, x_val, y_val)
            tune_hp_tuner(x_train, y_train, x_val, y_val, method='Random')  # method = ['Bayes', 'Random']

            return
        else:
            # Create, train, and save the neural network, evaluate it on the test data
            model_names = [train(x_train, y_train, x_val, y_val, p) for _ in range(n_models)]
            predictions, accuracy = evaluate_test_data(model_names, x_test, y_test, x_val=x_val, y_val=y_val,
                                                       x_train=x_train, y_train=y_train)
    else:
        # Names of the model in ./Models/chemical/
        model_names = ['20220330113805_CNN.h5']

    if interpolate_to in ["Itokawa", "Eros"]:
        # No-label datafile in ./Datasets/
        filename_data = interpolate_to + '-nolabel.dat'

        # Evaluate the models; second argument can be a path to the data or the data itself
        predictions = evaluate(model_names, filename_data)
        plot_surface_spectra(predictions, filename_data, 'mineralogy')

    return predictions


if __name__ == '__main__':
    for _ in range(10):
        y_pred = pipeline()

"""
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import numpy as np
import matplotlib.pyplot as plt

from modules.NN_data import load_data, split_data_proportional
from modules.NN_evaluate import evaluate_test_data, evaluate
from modules.tables import mean_asteroid_type, accuracy_table, quantile_table, mean_S_asteroid_type
from modules.NN_config import num_labels, use_minerals_all
from modules.control_plots import plot_ast_PC1_PC2, plot_Fa_vs_Fs_ast_only, plot_spectra
from modules.control_plots import plot_PCA_BAR, plot_PCA_BAR, plot_scatter_NN_BC

filename_train_data = 'combined-denoised-norm.dat'
x_train, y_train = load_data(filename_train_data, clean_dataset=True)
# plot_spectra(x_train, y_train)
x_train, y_train, x_val, y_val, x_test, y_true = split_data_proportional(x_train, y_train)


if num_labels == 2 and use_minerals_all[0]:  # OL only
    model_names = ['20220325105832_CNN_OL.h5']  # [4.5]

if num_labels == 2 and use_minerals_all[1]:  # OPX only, no Wo_OPX
    model_names = ['20220325114608_CNN_OPX.h5']  # [4.5]

if num_labels == 6:  # OL + OPX + OL_OPX_mix only, no low-iron OL, no Wo_OPX
    model_names = ['20220404141225_CNN_OL_OPX.h5']  # [4.7, 6.4, 5.2]

if num_labels == 11:  # all, no low-iron, no Wo_OPX, no PLG chemical
    model_names = ['20220329232107_CNN.h5']  # [15.1, 5.3, 6.6, 10.0]

if num_labels == 15: # all, no low-iron
    model_names = ['20220331112738_CNN.h5']  # [15.3, 5.3, 6.2, 10.8, 24.4]

if num_labels == 10:  # all, no low-iron, no PLG, no Wo_OPX  -- final model
    model_names = ['20220330113805_CNN.h5']  # [14.4, 5.7, 5.7, 10.7]

y_pred, accuracy = evaluate_test_data(model_names, x_test, y_true, x_val, y_val, x_train, y_train)
accuracy_table(y_true, y_pred)
quantile_table(y_true, y_pred)

filename_data = 'asteroid_spectra-denoised-norm-nolabel.dat'
y_pred = evaluate(model_names, filename_data)
mean_asteroid_type(y_pred)
plot_ast_PC1_PC2(y_pred)
plot_Fa_vs_Fs_ast_only()
plot_scatter_NN_BC()
plot_PCA_BAR()

filename_data = 'Chelyabinsk-denoised-norm-nolabel.dat'
y_pred = evaluate(model_names, filename_data)
"""
