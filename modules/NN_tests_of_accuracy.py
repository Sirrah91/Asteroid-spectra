from os import environ

environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from typing import Tuple
from sklearn.model_selection import KFold, LeaveOneOut
import numpy as np
from copy import deepcopy

from modules.NN_data import load_data
from modules.NN_train import train
from modules.NN_evaluate import evaluate_test_data
from modules.NN_config import *


def split_data(x_train: np.ndarray, y_train: np.ndarray, options: Tuple) -> Tuple[np.ndarray, ...]:
    method = options[0]
    index = options[1]
    K = options[2]

    # Leave-one-out
    if method == "LOO":
        # This can be written usingKFold(n_splits=K) too when K = len(x_train)
        train_indices, test_indices = list(LeaveOneOut().split(x_train))[index]

    # K-fold method
    if method == "K-fold":
        train_indices, test_indices = list(KFold(n_splits=K).split(x_train))[index]

    x_train, x_test = deepcopy(x_train[train_indices, :]), deepcopy(x_train[test_indices, :])
    y_train, y_test = deepcopy(y_train[train_indices]), deepcopy(y_train[test_indices])

    # This must be done due to keras
    x_train = np.asarray(x_train).astype(np.float32)
    x_test = np.asarray(x_test).astype(np.float32)
    y_train = np.asarray(y_train).astype(np.float32)
    y_test = np.asarray(y_test).astype(np.float32)

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':

    filename = '/RELAB/comb.dat'

    # Load the data
    x_tr, y_tr = load_data(filename)

    # Shuffle the data
    np.random.seed(42)  # to always get the same permutation
    idx = np.random.permutation(len(x_tr))
    x_tr = x_tr[idx]
    y_tr = y_tr[idx]

    method = "K-fold"
    K = 10  # just to be sure that something is here

    # If LOO then maximum training size
    if method == "LOO":
        K = len(x_tr)

    # If K-fold then 10
    if method == "K-fold":
        K = 10

    accuracy = [0] * K

    for i in range(K):
        # Split them to train and test parts
        X_tr, Y_tr, x_te, y_te = split_data(x_tr, y_tr, (method, i, K))

        # Create and train the neural network and save the model
        model_name = train(X_tr, Y_tr, x_te, y_te, p)

        _, acc = evaluate_test_data([model_name], x_te, y_te)
        accuracy[i] = acc

    final_accuracy = np.mean(accuracy)

    print("-----------------------------------------------------")
    print('Final accuracy:', str("{:7.5f}").format(np.round(final_accuracy, 5)))
    print("-----------------------------------------------------")
