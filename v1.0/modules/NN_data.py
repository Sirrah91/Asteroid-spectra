import numpy as np
import pandas as pd  # terality
from typing import Tuple
from sklearn.model_selection import train_test_split
from copy import deepcopy

from modules.NN_config import *


def load_data(filename_train_data: str) -> Tuple[np.ndarray, ...]:
    # This function load a data from a dataset

    print('Loading train file')

    data_file = "".join((project_dir, '/Datasets/RELAB/', filename_train_data))
    data = pd.read_csv(data_file, sep='\t', header=None).to_numpy()

    # Select training data
    x_train, y_train = deepcopy(data[:, :-num_labels_all].astype(np.float32)), deepcopy(data[:, -num_labels_all:].astype(np.float32))

    # this is for testing of ideal range
    # 350 -- 2550 -> RMSE = 12.5
    # 450 -- 2450 -> RMSE = 11.8
    # 550 -- 2350 -> RMSE = 10.6
    # 600 -- 2300 -> RMSE = 12.0
    # 650 -- 2250 -> RMSE = 12.0
    # 500 -- 1200 + 1600 -- 2400 -> RMSE = 10.0
    # 500 -- 1150 + 1600 -- 2200 -> RMSE = 13.4
    # 700 -- 1150 + 1700 -- 2300 -> RMSE = 11.4

    # x_train = np.concatenate([x_train[:, 70:161], x_train[:, 270:391]], axis=1)

    if not use_rebal_data:
        # Tuomas' data are from 450 nm to 2450 nm, so take 20:-20
        x_train = x_train[:, 20:-20]

    # remove unwanted minerals
    ind_to_remove = np.where(used_indices == False)[0]
    # remove
    y_train = np.delete(y_train, ind_to_remove, axis=1)

    # x_train, y_train = remove_redundant_spectra(x_train, y_train)

    stop = num_minerals
    # renormalise modal
    if 0 < num_minerals < len(use_minerals):  # one or more minerals are missing, normalise modal to unit sum
        start = 0
        norm = np.sum(y_train[:, start:stop], axis=1)
        # normalise only where sum of numbers is non-zero (should not happen)
        non_zero = np.where(norm > 0)[0]
        y_train[non_zero, start:stop] = np.transpose(
            np.divide(np.transpose(y_train[non_zero, start:stop]), norm[non_zero]))

    # renormalise chemical
    for k in range(len(subtypes)):
        start, stop = stop, stop + subtypes[k]
        if subtypes[k] != len(subtypes_all_used[k]):
            norm = np.sum(y_train[:, start:stop], axis=1)
            # normalise only where sum of numbers is non-zero (should not happen)
            non_zero = np.where(norm > 0)[0]
            y_train[non_zero, start:stop] = np.transpose(
                np.divide(np.transpose(y_train[non_zero, start:stop]), norm[non_zero]))

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


def remove_nans(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # NaNs in numbers
    inds = np.unique(np.where(np.isnan(y_data))[0])

    x_data = np.delete(x_data, inds, axis=0)
    y_data = np.delete(y_data, inds, axis=0)

    return x_data, y_data


def remove_redundant_spectra(x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    useful_spectra = np.nansum(y_data, axis=1) > 0

    return x_data[useful_spectra], y_data[useful_spectra]
