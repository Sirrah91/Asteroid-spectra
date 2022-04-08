# ------------------------------------------------------------------------------------------- #
# Neural Network to Classify Asteroids Using Keras and Tensorflow Backend                     #
# by David Korda (david.korda@helsinki.fi)                                                    #
# ------------------------------------------------------------------------------------------- #
# Run with python 3.7.4                                                                       #
# Install: numpy, tensorflow + dependencies, keras + dependencies, see needed libraries below #
# ------------------------------------------------------------------------------------------- #

from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from modules.NN_data_Tuomas import load_data, split_data
from modules.NN_train_Tuomas import train, tune_hp
from modules.NN_evaluate_Tuomas import evaluate_test_data, evaluate
import numpy as np
from typing import Union
import h5py

from modules.control_plots import plot_surface_spectra
from modules.NN_config_Tuomas import *

train_new_model = False  # If you have a trained model, run just evaluate.py
tune_hyperparameters = False  # if you just want to tune hp (train_new_model = True is necessary)


def pipeline(n_models: int = 1) -> Union[np.ndarray, None]:
    # Computes an average of n_models

    if train_new_model:
        # Name of the train data in ./Datasets/Tuomas/
        filename_train_data = 'MyVISNIR-simulated-HB-simplified-taxonomy.dat'

        # Load the data
        x_train, y_train = load_data(filename_train_data)

        # Split the data
        x_train, y_train, x_val, y_val, x_test, y_test = split_data(x_train, y_train)

        if tune_hyperparameters:
            # Tuning of the hyper-parameters defined in p_for_tuning dictionary
            tune_hp(x_train, y_train, x_val, y_val)

            return
        else:
            # Create, train, and save the neural network, evaluate it on the test data
            model_names = [train(x_train, y_train, x_val, y_val, p) for _ in range(n_models)]
            predictions, acc = evaluate_test_data(model_names, x_test, y_test)
    else:
        # Name of the model in ./Models/classification/
        if model_name_suffix == 'Itokawa':
            model_names = ['20210608121330_Itokawa.h5',
                           '20210608121523_Itokawa.h5',
                           '20210608121715_Itokawa.h5',
                           '20210608121907_Itokawa.h5',
                           '20210608122100_Itokawa.h5',
                           '20210608122252_Itokawa.h5',
                           '20210608122455_Itokawa.h5',
                           '20210608122717_Itokawa.h5',
                           '20210608122936_Itokawa.h5',
                           '20210608123154_Itokawa.h5'
                           ]
        elif model_name_suffix == 'Eros':
            model_names = ['20210608114828_Eros.h5',
                           '20210608115027_Eros.h5',
                           '20210608115220_Eros.h5',
                           '20210608115405_Eros.h5',
                           '20210608115553_Eros.h5',
                           '20210608115743_Eros.h5',
                           '20210608115926_Eros.h5',
                           '20210608120121_Eros.h5',
                           '20210608120317_Eros.h5',
                           '20210608120511_Eros.h5'
                           ]
        else:
            raise ValueError('"model_name_suffix" in the config file must be either "Itokawa" or "Eros"')

    # Name of the new data in ./Datasets/Tuomas/
    if model_name_suffix == 'Itokawa':
        filename = 'polysum.h5'
    elif model_name_suffix == 'Eros':
        filename = 'polysumeros1000.h5'
    else:
        raise ValueError('"model_name_suffix" in the config file must be either "Itokawa" or "Eros"')

    with h5py.File("".join((project_dir, 'Datasets/Tuomas/', filename)), 'r') as f:
        # Choosing only the spectra, first two indexes contain coordinates
        # Removing the coordinate info from the beginning of each sample
        data = np.array(f['d'][:, 2:])

    # Evaluate the models; second argument can be a path to the data or the data itself
    predictions = evaluate(model_names, data)
    plot_surface_spectra(predictions, filename)

    return predictions


if __name__ == '__main__':
    y_pred = pipeline()
