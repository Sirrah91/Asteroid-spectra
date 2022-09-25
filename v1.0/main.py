# ----------------------------------------------------------------------- #
# Neural Network to Classify Asteroids Using Keras and Tensorflow Backend #
# by David Korda (david.korda@helsinki.fi)                                #
# ----------------------------------------------------------------------- #
# Run with python 3.8.10                                                  #
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
environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from modules.NN_data import load_data, split_data_proportional
from modules.NN_train import train, tune_hp_talos, tune_hp_tuner
from modules.NN_evaluate import evaluate_test_data, evaluate
import numpy as np
from typing import Union
from tqdm import tqdm  # For progress bar

from modules.paper_plots import plot_surface_spectra
from modules.NN_config import *

train_new_model = True  # If you have a trained model, just run evaluate(model_names, filename_data_or_data)
tune_hyperparameters = False  # if you just want to tune hp


def pipeline(n_models: int = 1) -> Union[np.ndarray, None]:
    # pipeline(n_models) computes a trimmed mean of n_models and returns predictions or print best hyperparameters

    if train_new_model or tune_hyperparameters:
        # Name of the train data in ./Datasets/
        filename_train_data = 'combined-denoised-norm.npz'

        # Load the data
        x_train, y_train = load_data(filename_train_data, clean_dataset=True)

        # Split the data
        x_train, y_train, x_val, y_val, x_test, y_test = split_data_proportional(x_train, y_train)

        if tune_hyperparameters:
            # Tuning of the hyperparameters defined in p_for_tuning dictionary
            if p_for_tuning['tuning_method'] == 'talos':
                tune_hp_talos(x_train, y_train, x_val, y_val)
            else:
                tune_hp_tuner(x_train, y_train, x_val, y_val, method=p_for_tuning['tuning_method'])

            return
        else:
            # Create, train, and save the neural network, evaluate it on the test data
            model_names = [train(x_train, y_train, x_val, y_val, p) for _ in range(n_models)]
            predictions, accuracy = evaluate_test_data(model_names, x_test, y_test, x_val=x_val, y_val=y_val,
                                                       x_train=x_train, y_train=y_train, subfolder_model=model_dir)
    else:
        # Names of the model in ./Models/compositional/
        model_names = ['20220330113805_CNN.h5']

    filename_data = 'Didymos-denoised-norm.npz'

    model_names = ['20220720211034_CNN.h5',
                   '20220720214154_CNN.h5',
                   '20220720220904_CNN.h5',
                   '20220720223812_CNN.h5',
                   '20220720225952_CNN.h5',
                   '20220720231511_CNN.h5',
                   '20220721102638_CNN.h5',
                   '20220721104846_CNN.h5',
                   '20220721111416_CNN.h5',
                   '20220721113820_CNN.h5']

    if interpolate_to in ["Itokawa", "Eros"]:
        # No-label datafile in ./Datasets/
        filename_data = interpolate_to + '.npz'

        if interpolate_to in "Itokawa":
            model_names = ['20220610105620_CNN.h5',
                           '20220610110552_CNN.h5',
                           '20220610112130_CNN.h5',
                           '20220610113735_CNN.h5',
                           '20220610114833_CNN.h5',
                           '20220610120451_CNN.h5',
                           '20220610122038_CNN.h5',
                           '20220610122908_CNN.h5',
                           '20220610124440_CNN.h5',
                           '20220610125501_CNN.h5']
        else:  # Eros
            model_names = ['20220610130321_CNN.h5',
                           '20220610131845_CNN.h5',
                           '20220610132756_CNN.h5',
                           '20220610133912_CNN.h5',
                           '20220610134912_CNN.h5',
                           '20220610140202_CNN.h5',
                           '20220610141042_CNN.h5',
                           '20220610163539_CNN.h5',
                           '20220610164818_CNN.h5',
                           '20220610165847_CNN.h5']

        # Evaluate the models; second argument can be a path to the data or the data itself
        predictions = evaluate(model_names, filename_data)
        plot_surface_spectra(predictions, filename_data, 'composition')

    return predictions


if __name__ == '__main__':
    if tune_hyperparameters:
        pipeline()
    elif train_new_model:
        for _ in tqdm(range(10)):
            y_pred = pipeline()
    else:
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
from modules.paper_plots import plot_ast_PC1_PC2, plot_Fa_vs_Fs_ast_only, plot_spectra, plot_mineralogy_histogram
from modules.paper_plots import plot_PCA_BAR, plot_scatter_NN_BC, plot_error_density_plots

filename_train_data = 'combined-denoised-norm.npz'
x_train, y_train = load_data(filename_train_data, clean_dataset=True)
# plot_mineralogy_histogram(y_train)
# plot_spectra(x_train, y_train)
x_train, y_train, x_val, y_val, x_test, y_true = split_data_proportional(x_train, y_train)

if num_labels == 2 and use_minerals_all[0]:  # OL only
    model_names = ['20220325105832_CNN_OL.h5']  # [4.4]

if num_labels == 2 and use_minerals_all[1]:  # OPX only, no Wo_OPX
    model_names = ['20220325114608_CNN_OPX.h5']  # [4.5]

if num_labels == 6:  # OL + OPX + OL_OPX_mix only, no low-iron OL, no Wo_OPX
    model_names = ['20220404141225_CNN_OL_OPX.h5']  # [4.7, 6.4, 5.2]

if num_labels == 11:  # all, no low-iron, no Wo_OPX, no PLG chemical
    model_names = ['20220329232107_CNN.h5']  # [8.6, 21.6, 19.4, 11.1, 5.3, 8.1, 8.7, 10.3, 10.6]

if num_labels == 15: # all, no low-iron
    model_names = ['20220331112738_CNN.h5']  # [8.5, 21.2, 19.7, 11.9, 5.2, 9.4, 9.8, 2.4, 9.8, 11.1, 11.3, 25.5, 29.5, 17.4]

if num_labels == 10:  # all, no low-iron, no PLG, no Wo_OPX  -- final model
    model_names = ['20220330113805_CNN.h5']  # [7.0, 18.2, 18.2, 5.6, 5.7, 8.5, 11.7, 11.6]

y_pred, accuracy = evaluate_test_data(model_names, x_test, y_true, x_val, y_val, x_train, y_train)
accuracy_table(y_true, y_pred)
quantile_table(y_true, y_pred)
plot_error_density_plots(y_true, y_pred)
plot_scatter_NN_BC()

filename_data = 'asteroid_spectra-denoised-norm.npz'
y_pred = evaluate(model_names, filename_data)
mean_asteroid_type(y_pred)
mean_S_asteroid_type(y_pred)
plot_ast_PC1_PC2(y_pred)
plot_Fa_vs_Fs_ast_only()
plot_PCA_BAR()

filename_data = 'Chelyabinsk-denoised-norm.npz'
y_pred = evaluate(model_names, filename_data)

filename_data = 'Kachr_ol_opx-denoised-norm.npz'
y_pred = evaluate(model_names, filename_data)
"""
