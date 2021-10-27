import numpy as np
from typing import List, Union, Tuple
from keras.models import load_model
from scipy.stats import trim_mean

from modules.control_plots import plot_confusion_matrix
from modules.NN_config_Tuomas import *


def average_and_normalise(predictions: np.ndarray) -> np.ndarray:
    # Trimmed mean
    predictions = trim_mean(predictions, trimmed, axis=2)

    # Normalisations to 1
    norm = np.sum(predictions, axis=1)
    predictions = np.transpose(np.divide(np.transpose(predictions), norm))

    return predictions


def evaluate(model_names: List[str], filename_or_data: Union[str, np.ndarray]) -> np.ndarray:
    # This function evaluate the mean model on new a dataset

    if isinstance(filename_or_data, str):
        # Import the test dataset
        print('Loading dataset')

        data_file = "".join((project_dir, '/Datasets/Tuomas/', filename_or_data))
        data = np.loadtxt(data_file, delimiter='\t')
    else:
        data = np.array(filename_or_data)

    print("Evaluating the neural network")
    predictions = np.zeros((len(data), num_labels, len(model_names)))

    # Calc average prediction over the models
    for idx, model_name in enumerate(model_names):
        model = load_model("".join((project_dir, '/Models/classification/', model_name)))

        # Evaluate model on test data
        predictions[:, :, idx] = model.predict(data)

    # Trimmed means and normalisations to 1
    predictions = average_and_normalise(predictions)

    print("-----------------------------------------------------")

    return predictions


def evaluate_test_data(model_names: List[str], x_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, ...]:
    print("Evaluating the neural network on the test data")
    predictions = np.zeros((len(x_test), num_labels, len(model_names)))

    # Calc average prediction over the models
    for idx, model_name in enumerate(model_names):
        model = load_model("".join((project_dir, '/Models/classification/', model_name)))

        # Evaluate model on test data
        predictions[:, :, idx] = model.predict(x_test)

    # Trimmed means and normalisations to 1
    predictions = average_and_normalise(predictions)

    # Evaluate the accuracy
    acc = np.mean(metrics[0](y_test, predictions).numpy())
    print('Test accuracy:', str("{:7.5f}").format(np.round(acc, 5)))

    if show_control_plot:
        plot_confusion_matrix(y_test, predictions)

    print("-----------------------------------------------------")

    return predictions, acc
