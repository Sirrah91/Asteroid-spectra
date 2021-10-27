from typing import Tuple, Union, Dict
import numpy as np
from datetime import datetime
import talos
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, History, TerminateOnNaN
from keras import regularizers
from pprint import pprint
from keras_visualizer import visualizer

from modules.NN_losses_metrics_activations import *
from modules.control_plots import plot_model_history
from modules.NN_config import *
from modules.utilities import *


# HERE ARE SELECTED THE LOSSES AND ACCURACIES
if num_minerals > 1:
    my_loss = my_loss_v1
    my_r2 = my_r2_v1
else:
    my_loss = my_loss_v2
    my_r2 = my_r2_v2

# RMSE as a major metric
my_acc_name = 'val_my_rmse'  # important for hp tuning and early stopping


def FC_model(x_train: np.ndarray, params: Dict) -> Sequential:
    #
    # ADD BATCH NORMALISATION?
    #

    model = Sequential()
    # Adding the input layer to the model
    model.add(Dropout(params['dropout_input'], input_shape=(x_train.shape[1],)))

    # Adding the hidden layers
    for i in range(params['n_layers']):
        model.add(Dense(params['n_nodes'][i], activation=params['input_activation'],
                        kernel_regularizer=regularizers.l1(l1=params['lambda1'])))
        # Dropout layer for stabilisation of the network
        model.add(Dropout(params['dropout_hidden']))

    # Number of nodes in output layer
    model.add(Dense(num_labels, activation=my_softmax))

    # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
    model.compile(loss=my_loss, optimizer=params['optimizer'](learning_rate=params['learning_rate']),
                  metrics=[my_rmse, my_r2])

    return model


def CNN_model(x_train: np.ndarray, params: Dict) -> Sequential:
    # input_shape = (params['batch_size'], np.shape(x_train)[1], 1)
    input_shape = (np.shape(x_train)[1], 1)

    model = Sequential()
    # Adding the input layer to the model
    model.add(Dropout(params['dropout_input'], input_shape=input_shape))

    # Adding the hidden layers
    for i in range(params['n_layers']):
        model.add(Conv1D(filters=params['n_nodes'][i], kernel_size=params['kern_size'],
                         activation=params['input_activation'], padding='same'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(params['dropout_hidden']))

    # Flattening after the last CONV layer
    model.add(Flatten())

    # Number of nodes in output layer
    model.add(Dense(num_labels, activation=my_softmax))

    # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
    model.compile(loss=my_loss, optimizer=params['optimizer'](learning_rate=params['learning_rate']),
                  metrics=[my_rmse, my_r2])

    return model


def train(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, params: Dict) -> Union[
    Tuple[History, Sequential], str]:

    if params['tuning']:
        for_tuning, quiet = True, True
    else:
        for_tuning, quiet = False, False

    if not quiet:
        print()  # empty line after loading the data
        print('Setting up the neural network')

    # Define model architecture
    if params['model_type'] == 'FC':  # fully connected
        model = FC_model(x_train, params)
    else:  # convolutional
        model = CNN_model(x_train, params)

    # visualise the model
    # visualizer(model, filename='architecture', format='png', view=True)

    # Train model
    if not quiet:
        print("Training...")

    if val_portion > 0:
        # parameter in early stopping
        if for_tuning:
            coef = 1
        else:
            coef = 0.3

        # Set early stopping monitor so the model will stop training if it does not improve anymore
        early_stopping_monitor = EarlyStopping(monitor=my_acc_name, mode='min', patience=params['num_epochs'] * coef,
                                               restore_best_weights=True)

        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=params['num_epochs'],
                            batch_size=params['batch_size'], callbacks=[TerminateOnNaN(), early_stopping_monitor],
                            verbose=verb)
    else:
        history = model.fit(x_train, y_train, validation_data=None, epochs=params['num_epochs'],
                            batch_size=params['batch_size'], callbacks=[TerminateOnNaN()], verbose=verb)

    if for_tuning:
        return history, model

    # Save model to project dir with a timestamp
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")
    model_name = "".join((dt_string, '_', model_name_suffix, '.h5'))

    filename = "".join((project_dir, '/Models/chemical/', model_name))
    check_dir(filename)
    model.save(filename)

    if not quiet:
        print("Model was saved to disk")

    if show_control_plot:
        plot_model_history(model)

    # Evaluate the train data
    if not quiet:
        acc = my_rmse(y_train, model.predict(x_train)).numpy()
        print('\nEvaluation on the data:')
        print('Mean train RMSE:', str("{:7.5f}").format(np.round(np.mean(acc), 5)))

        if val_portion > 0:
            # Evaluate the validation data
            acc = my_rmse(y_val, model.predict(x_val)).numpy()
            print('Mean validation RMSE:', str("{:7.5f}").format(np.round(np.mean(acc), 5)))
        print()  # Empty line

    return model_name


def tune_hp(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> None:
    scan_object = talos.Scan(x_train, y_train, params=p_for_tuning, model=train, x_val=x_val, y_val=y_val,
                             experiment_name='tuning_HP')

    best = scan_object.data.sort_values(my_acc_name, ascending=True).iloc[0][-len(p_for_tuning):]
    pprint(best)
