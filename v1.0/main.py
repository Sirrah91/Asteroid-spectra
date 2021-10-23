# ------------------------------------------------------------------------------------------- #
# Neural Network to Classify Asteroids Using Keras and Tensorflow Backend                     #
# by David Korda (david.korda@helsinki.fi)                                                    #
# ------------------------------------------------------------------------------------------- #
# Run with python 3.8.10                                                                      #
# Install: numpy, tensorflow + dependencies, keras + dependencies, see needed libraries below #
# ------------------------------------------------------------------------------------------- #

from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from modules.collect_data import collect_data_RELAB
from modules.NN_data import load_data, split_data, remove_nans
from modules.NN_train import train, tune_hp
from modules.NN_evaluate import evaluate_test_data, evaluate
import numpy as np
from typing import Union

from modules.NN_config import *

collect_RELAB_data = False  # Collect data from RELAB
train_new_model = True  # If you have a trained model, run just evaluate.py
tune_hyperparameters = False  # if you just want to tune hp (train_new_model = True is necessary)


def pipeline(n_models: int = 5) -> Union[np.ndarray, None]:
    # Computes an average of n_models

    if collect_RELAB_data:
        # the numbers come from SPECTRAL CATALOGUE
        first_line, last_line, name = (2,), (560,), ('comb_all',)
        collect_data_RELAB(first_line, last_line, name)
        filename_train_data = "".join((name, '.dat'))
    else:
        # Name of the train data in ./Datasets/RELAB/
        filename_train_data = 'min_and_mix.dat'
        # filename_train_data = 'synthetic_OL_OPX_CPX_PLG-norm-denoised.dat'

    if train_new_model:
        # Load the data
        x_train, y_train = load_data(filename_train_data)
        x_train, y_train = remove_nans(x_train, y_train)

        # Split the data
        x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_train, y_train)

        # this is just to compare synthetic mixtures and meteorites
        filename_train_data = 'OC-norm-denoised.dat'
        x_test, y_test = load_data(filename_train_data)

        if tune_hyperparameters:
            # Tuning of the hyper-parameters defined in p_for_tuning dictionary
            # tune_hp(x_train, y_train, x_val, y_val)
            tune_hp(x_train, y_train, x_test, y_test)

            return
        else:
            # Create, train, and save the neural network, evaluate it on the test data
            model_names = [train(x_train, y_train, x_val, y_val, p) for _ in range(n_models)]
            predictions, accuracy = evaluate_test_data(model_names, x_test, y_test)
    else:
        # Name of the model in ./Models/chemical/
        """ 
        # these models are for full range; based on real data
        model_names = ['20211018155023_FC.h5']
        """

        """ 
        # these models are for full range; based on synthetic OC
        model_names = ['20211020143237_FC.h5']
        """

        # these models are for Tuomas range; based on real data
        model_names = ['20211020205324_FC.h5']

        # Name of the new data in ./Datasets/RELAB/ or ./Datasets/Tuomas/
        filename_data = 'Tuomas_HB_spectra-norm-denoised_nolabel.dat'

        # Evaluate the models; second argument can be a path to the data or the data itself
        predictions = evaluate(model_names, filename_data)

    return predictions


if __name__ == '__main__':
    y_pred = pipeline(1)
