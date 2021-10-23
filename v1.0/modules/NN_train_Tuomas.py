from typing import Tuple, Union, Dict
import numpy as np
from datetime import datetime
import talos
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, History, TerminateOnNaN
from keras import regularizers
from pprint import pprint
# from keras_visualizer import visualizer

from modules.control_plots import plot_model_history
from modules.NN_config_Tuomas import *


def FC_model(x_train: np.ndarray, params: Dict) -> Sequential:
    #
    # ADD BATCH NORMALISATION?
    #

    model = Sequential()
    # Adding the input layer to the model
    model.add(Dropout(params['dropout_input'], input_shape=(x_train.shape[1],)))

    model.add(Dense(params['nodes_1'], activation=params['input_activation'],
                    kernel_regularizer=regularizers.l1(l1=params['lambda1'])))

    # Dropout layer for stabilisation of the network
    model.add(Dropout(params['dropout_hidden']))

    # Number of nodes in output layer (same as classes)
    model.add(Dense(num_labels, activation=params['output_activation']))

    # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
    model.compile(loss=loss, optimizer=params['optimizer'](learning_rate=params['learning_rate']),
                  metrics=metrics)

    return model


def CNN_model(x_train: np.ndarray, params: Dict) -> Sequential:
    input_shape = (np.reshape(x_train, (*np.shape(x_train), 1)).shape[1], 1)

    model = Sequential()

    # Adding the initial and the first layers to the model
    model.add(Conv1D(filters=params['nodes_1'], kernel_size=params['kern_size_1'],
                     activation=params['input_activation'], padding='same', 
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(params['dropout_hidden']))

    model.add(Conv1D(filters=params['nodes_2'], kernel_size=params['kern_size_2'],
                     padding='same', activation=params['input_activation']))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(params['dropout_hidden']))

    # Flattening after the last CONV layer
    model.add(Flatten())

    # Number of nodes in output layer (same as classes)
    model.add(Dense(num_labels, activation=params['output_activation']))

    # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
    model.compile(loss=loss, optimizer=params['optimizer'](learning_rate=params['learning_rate']),
                  metrics=metrics)

    return model


def train(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, params: dict) -> Union[
    Tuple[History, Sequential], str]:
    # If list, then for tuning

    if params['tuning']:
        for_tuning, quiet = True, True
    else:
        for_tuning, quiet = False, False

    if not quiet:
        print()  # empty line after loading the data
        print('Setting up the neural network')

    # Define model architecture
    if model_type == 'FC':  # fully connected
        model = FC_model(x_train, params)
    else:  # convolutional
        model = CNN_model(x_train, params)

    # visualise the model
    # visualizer(model, filename='architecture', format='png', view=True)

    # Train model
    if not quiet:
        print("Training...")

    if val_portion > 0:
        # Set early stopping monitor so the model stops training when it won't improve anymore
        early_stopping_monitor = EarlyStopping(monitor='val_loss', patience=params['num_epochs'] / 3,
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
    model.save("".join((project_dir, '/Models/classification/', model_name)))

    if not quiet:
        print("Model was saved to disk")

    if show_control_plot:
        plot_model_history(model)

    # Evaluate the train data
    if not quiet:
        acc = np.mean(metrics[0](y_train, model.predict(x_train)).numpy())
        print('\nEvaluation on the data:')
        print('Train accuracy:', str("{:7.5f}").format(np.round(acc, 5)))

        if val_portion > 0:
            # Evaluate the validation data
            acc = np.mean(metrics[0](y_val, model.predict(x_val)).numpy())
            print('Validation accuracy:', str("{:7.5f}").format(np.round(acc, 5)))
        print()  # Empty line

    return model_name


def tune_hp(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> None:
    scan_object = talos.Scan(x_train, y_train, params=p_for_tuning, model=train, x_val=x_val, y_val=y_val,
                             experiment_name='tuning_HP', reduction_method='correlation', fraction_limit=0.5)

    best = scan_object.data.sort_values('val_' + metrics[0], ascending=False).iloc[0][-len(p_for_tuning):]
    pprint(best)
