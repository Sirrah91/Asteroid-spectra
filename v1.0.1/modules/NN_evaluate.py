import numpy as np
from typing import List, Union, Tuple
from keras.models import load_model
from scipy.stats import trim_mean

from modules.NN_losses_metrics_activations import my_loss_v1, my_loss_v2, my_softmax, my_r2_v1, my_r2_v2, my_rmse
from modules.control_plots import plot_scatter_plots
from modules.NN_config import *


# HERE ARE SELECTED THE LOSSES AND ACCURACIES
if num_minerals > 1:
    my_loss = my_loss_v1
    my_r2 = my_r2_v1
    my_loss_name, my_r2_name = 'my_loss_v1', 'my_r2_v1'
else:
    my_loss = my_loss_v2
    my_r2 = my_r2_v2
    my_loss_name, my_r2_name = 'my_loss_v2', 'my_r2_v2'


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


def evaluate(model_names: List[str], filename_or_data: Union[str, np.ndarray]) -> np.ndarray:
    # This function evaluate the mean model on new a dataset

    if isinstance(filename_or_data, str):
        # Import the test dataset
        print('Loading dataset')

        if use_rebal_data:
            dataset = 'RELAB'
        else:
            dataset = 'Tuomas'

        data_file = "".join((project_dir, '/Datasets/', dataset, '/', filename_or_data))
        data = np.loadtxt(data_file, delimiter='\t')
    else:
        data = np.array(filename_or_data)

    data = data.astype(np.float32)

    print("Evaluating the neural network")
    predictions = np.zeros((len(data), num_labels, len(model_names)))

    # Calc average prediction over the models
    for idx, model_name in enumerate(model_names):
        model = load_model("".join((project_dir, '/Models/chemical/', model_name)),
                           custom_objects={my_loss_name: my_loss, "my_softmax": my_softmax,
                                           'my_rmse': my_rmse, my_r2_name: my_r2})

        # Evaluate model on test data
        predictions[:, :, idx] = model.predict(data)

    # Trimmed means and normalisations to 1
    predictions = average_and_normalise(predictions)

    print("-----------------------------------------------------")

    return predictions


def evaluate_test_data(model_names: List[str], x_test: np.ndarray, y_test: np.ndarray) -> Tuple[np.ndarray, float]:
    print("Evaluating the neural network on the test data")
    predictions = np.zeros((len(x_test), num_labels, len(model_names)))

    # Calc average prediction over the models
    for idx, model_name in enumerate(model_names):
        model = load_model("".join((project_dir, '/Models/chemical/', model_name)),
                           custom_objects={my_loss_name: my_loss, "my_softmax": my_softmax,
                                           'my_rmse': my_rmse, my_r2_name: my_r2})

        # Evaluate model on test data
        predictions[:, :, idx] = model.predict(x_test)

    # Trimmed means and normalisations to 1
    predictions = average_and_normalise(predictions)

    # Evaluate the accuracy
    acc = my_rmse(y_test, predictions).numpy()
    print('Mean test RMSE:', str("{:7.5f}").format(np.round(np.mean(acc), 5)))

    if show_control_plot:
        plot_scatter_plots(y_test, predictions)

    print("-----------------------------------------------------")

    return predictions, acc
