import numpy as np
from typing import List, Union, Tuple
from keras.models import load_model
from scipy.stats import trim_mean

from modules.NN_losses_metrics_activations import *
from modules.control_plots import plot_scatter_plots, plot_error_evaluation
from modules.NN_config import *
from modules.utilities import print_accuracy

custom_objects = {loss_name: loss, output_activation_name: output_activation,
                  mse_name: mse, rmse_name: rmse, quantile_name: quantile, mae_name: mae, Lp_norm_name: Lp_norm,
                  r2_name: r2, sam_name: sam}


def average_and_normalise(predictions: np.ndarray) -> np.ndarray:
    # Trimmed mean
    predictions = trim_mean(predictions, trimmed, axis=2)

    # Normalisations to 1
    # modals
    start, stop = 0, num_minerals
    if num_minerals > 1:
        norm = np.sum(predictions[:, start:stop], axis=1)
        predictions[:, start:stop] = np.transpose(np.divide(np.transpose(predictions[:, start:stop]), norm))

    # chemicals
    for k in range(len(subtypes)):
        start, stop = stop, stop + subtypes[k]
        norm = np.sum(predictions[:, start:stop], axis=1)
        predictions[:, start:stop] = np.transpose(np.divide(np.transpose(predictions[:, start:stop]), norm))

    predictions = np.asarray(predictions).astype(np.float32)

    return predictions


def evaluate(model_names: List[str], filename_or_data: Union[str, np.ndarray],
             subfolder_model: str = 'chemical') -> np.ndarray:
    # This function evaluate the mean model on new a dataset

    if isinstance(filename_or_data, str):
        # Import the test dataset
        print('Loading dataset')

        data_file = "".join((project_dir, '/Datasets/', filename_or_data))
        data = np.loadtxt(data_file, delimiter='\t')
    else:
        data = np.array(filename_or_data)

    data = data.astype(np.float32)

    print("Evaluating the neural network")
    predictions = np.zeros((len(data), num_labels, len(model_names)))

    # Calc average prediction over the models
    for idx, model_name in enumerate(model_names):
        model = load_model("".join((project_dir, '/Models/', subfolder_model, '/', model_name)),
                           custom_objects=custom_objects)

        # Evaluate model on test data
        predictions[:, :, idx] = model.predict(data)

    # Trimmed means and normalisations to 1
    predictions = average_and_normalise(predictions)

    print("-----------------------------------------------------")

    return predictions


def evaluate_test_data(model_names: List[str], x_test: np.ndarray, y_test: np.ndarray,
                       x_val: np.ndarray = None, y_val: np.ndarray = None,
                       x_train: np.ndarray = None, y_train: np.ndarray = None,
                       subfolder_model: str = 'chemical') -> Tuple[np.ndarray, ...]:
    print("Evaluating the neural network on the test data")
    predictions = np.zeros((len(x_test), num_labels, len(model_names)))

    if y_train is not None:
        predictions_train = np.zeros((len(x_train), num_labels, len(model_names)))
    if y_val is not None:
        predictions_val = np.zeros((len(x_val), num_labels, len(model_names)))

    # Calc average prediction over the models
    for idx, model_name in enumerate(model_names):
        model = load_model("".join((project_dir, '/Models/', subfolder_model, '/', model_name)),
                           custom_objects=custom_objects)

        # Evaluate model on test data
        predictions[:, :, idx] = model.predict(x_test)
        if y_train is not None:
            predictions_train[:, :, idx] = model.predict(x_train)
        if y_val is not None:
            predictions_val[:, :, idx] = model.predict(x_val)

    # Trimmed means and normalisations to 1
    predictions = average_and_normalise(predictions)
    if y_train is not None:
        predictions_train = average_and_normalise(predictions_train)
    if y_val is not None:
        predictions_val = average_and_normalise(predictions_val)

    print(model_name)

    # Evaluate the accuracy
    if y_train is not None:
        acc = rmse(y_train, predictions_train).numpy()
        print_accuracy(acc, 'train')
    if y_val is not None:
        acc = rmse(y_val, predictions_val).numpy()
        print_accuracy(acc, 'validation')
    acc = rmse(y_test, predictions).numpy()
    print_accuracy(acc, 'test')

    # these two are result plots
    plot_scatter_plots(y_test, predictions)
    plot_error_evaluation(y_test, predictions)

    if show_control_plot:
        if y_val is not None:
            plot_scatter_plots(y_val, predictions_val, suf='_val')
        if y_train is not None:
            plot_scatter_plots(y_train, predictions_train, suf='_train')

    print("-----------------------------------------------------")

    return predictions, acc
