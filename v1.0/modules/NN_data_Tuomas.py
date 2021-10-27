import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from copy import deepcopy
from keras.utils import np_utils

from modules.NN_convert_classes import convert_classes
from modules.NN_config_Tuomas import *


def load_data(filename_train_data: str) -> Tuple[np.ndarray, ...]:
    # This function load a data from a dataset

    print('Loading train file')

    data_file = "".join((project_dir, '/Datasets/Tuomas/', filename_train_data))
    data = pd.read_csv(data_file, sep='\t', header=None).to_numpy()  # to read the file

    # Select training data
    x_train, y_train = deepcopy(data[:, 1:].astype(np.float32)), deepcopy(data[:, 0])

    if model_name_suffix == 'Itokawa':
        # Matching the wl range with HB range (HB:770-2200 vs BDM:450-2450), [39:164]
        indices = range(39, 164)
    elif model_name_suffix == 'Eros':
        # Matching the wl range with HB range (Eros:810-2450 vs BDM:450-2450)
        indices = [37, 39, 41, 43, 45, 47, 49, 52, 54, 56, 58, 60, 62, 65, 67, 69, 71, 73, 75, 77, 80, 82, 109, 113,
                   117, 121, 126, 130, 134, 139, 143, 147, 156, 160, 164, 169, 173, 177, 182, 186, 191, 199]
    else:
        raise ValueError('"model_name_suffix" in the config file must be either "Itokawa" or "Eros"')

    x_train = x_train[:, indices]

    # The indices below follow the comments in Tuomas' code
    if model_name_suffix == 'Itokawa':
        # Normalize with reflectance at 1550
        index = 70  # lambda[index] = 1550
    else:  # Must be Eros due to the error above
        # Normalize with reflectance at 1600
        index = 23  # lambda[index] = 1600

    norm = x_train[:, index]
    x_train = np.transpose(np.divide(np.transpose(x_train), norm))

    y_train = convert_classes(y_train)

    # Split test data into distinct class labels
    y_train = np_utils.to_categorical(y_train, num_labels)

    # Set data into numpy arrays
    x_train, y_train = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.float32)

    return x_train, y_train


def split_data(x_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, ...]:
    # This function splits the training data

    if val_portion > 0:
        print('Creating validation data')
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_portion)
    else:
        x_val, y_val = np.array([]), np.array([])

    if test_portion > 0:
        print('Creating test data')
        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                            test_size=test_portion / (1 - val_portion))
    elif val_portion > 0:  # If test portion is zero then use validation data
        x_test, y_test = deepcopy(x_val), deepcopy(y_val)
    else:  # If even val portion is zero, use train data (just for visualisation purposes)
        x_test, y_test = deepcopy(x_train), deepcopy(y_train)

    return x_train, y_train, x_val, y_val, x_test, y_test
