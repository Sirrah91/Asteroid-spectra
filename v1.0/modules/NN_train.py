from typing import Tuple, Union, Dict
import numpy as np
from datetime import datetime
import talos
from keras.models import Functional
from keras.callbacks import EarlyStopping, History, TerminateOnNaN
from pprint import pprint
from keras_visualizer import visualizer
from keras.utils.vis_utils import plot_model

from modules.NN_models import *
from modules.control_plots import plot_model_history
from modules.NN_config import *
from modules.utilities import *


def train(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, params: Dict) -> Union[
    Tuple[History, Functional], str]:
    if params['tuning']:
        for_tuning, quiet, verbose = True, True, 0
    else:
        for_tuning, quiet, verbose = False, False, verb

    if not quiet:
        # empty line after loading the data
        print('\nSetting up the neural network')

    # Define model architecture
    if params['model_type'] == 'FC':  # fully connected
        model = FC_model(x_train, params)
    elif params['model_type'] == 'CNN':  # convolutional
        model = CNN_model(x_train, params)
    # elif params['model_type'] == 'RNN':  # recurrent
    #     model = RNN_model(x_train, params)
    # elif params['model_type'] == 'RCNN':  # recurrent convolutional
    #     model = RCNN_model(x_train, params)
    elif params['model_type'] == 'CNN_FC':  # combined
        model = CNN_FC_model(x_train, params)
    elif params['model_type'] == 'two_thread':  # combined
        model = two_thread_output_model(x_train, params)
    else:
        raise NameError("unknown p['model_type']")

    # visualise the model
    # visualizer(model, filename='architecture', format='png', view=True)

    # Train model
    if not quiet:
        if verbose == 1:
            model.summary()
        print("Training...")

    if val_portion > 0:
        # parameter in early stopping
        coef = 0.5 if for_tuning else 0.35

        # Set early stopping monitor so the model will stop training if it does not improve anymore
        early_stopping_monitor = EarlyStopping(monitor=main_acc_name, mode='min', patience=params['num_epochs'] * coef,
                                               restore_best_weights=True)

        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=params['num_epochs'],
                            batch_size=params['batch_size'], validation_batch_size=len(y_val),
                            callbacks=[TerminateOnNaN(), early_stopping_monitor], verbose=verbose)
    else:
        history = model.fit(x_train, y_train, validation_data=None, epochs=params['num_epochs'],
                            batch_size=params['batch_size'], callbacks=[TerminateOnNaN()], verbose=verbose)

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
        print('\nEvaluation on the data:')
        acc = rmse(y_train, model.predict(x_train)).numpy()
        print_accuracy(acc, 'train')

        if val_portion > 0:
            # Evaluate the validation data
            acc = rmse(y_val, model.predict(x_val)).numpy()
            print_accuracy(acc, 'validation')
        print()  # Empty line

    return model_name


def tune_hp(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    scan_object = talos.Scan(x_train, y_train,
                             params=p_for_tuning, model=train,
                             x_val=x_val, y_val=y_val,
                             # does the reduction work well?
                             reduction_method='correlation',
                             reduction_interval=50,
                             reduction_window=25,
                             reduction_threshold=0.2,
                             reduction_metric=main_acc_name,
                             minimize_loss=True,
                             #
                             fraction_limit=0.9999999,  # to make the search random
                             experiment_name='tuning_HP')

    data_all = scan_object.data
    best = data_all.sort_values(main_acc_name, ascending=True).iloc[0][-len(p_for_tuning):]
    pprint(best)

    # calc and plot correlation matrix
    """
    # if you load a csv (you MUST have the headers in correct columns)
    import pandas as pd
    data_all = pd.read_csv('./tuning_HP/021022092317.csv')
    """

    keep_values = np.where(data_all.keys() == 'val_loss')[0][0]
    data = data_all.to_numpy()[:, keep_values:]
    keys = data_all.keys().to_numpy()[keep_values:]

    # filter out non-numerical values (how to add some of these?)
    indices = np.logical_not(np.logical_or.reduce((keys == 'tuning', keys == 'model_type',
                                                   keys == 'input_activation', keys == 'output_activation',
                                                   keys == 'optimizer', keys == 'n_nodes', keys == 'n_nodes.1',
                                                   keys == 'n_nodes.2')))
    data, keys = data[:, indices].astype(np.float32), keys[indices].astype(np.str)

    # if only one option in a parameter, omit it
    indices = np.std(data, axis=0) > 1e-5
    data, keys = data[:, indices], keys[indices]

    corr_matrix = np.corrcoef(np.transpose(data))

    fig, ax = plt.subplots(figsize=(15, 15))
    cax = ax.matshow(corr_matrix)

    plt.xticks(np.arange(0, len(keys), 1.0), rotation=90)
    plt.yticks(np.arange(0, len(keys), 1.0))

    try:  # may end up in error if you are in latex mode and use '_'
        ax.set_xticklabels(keys)
        ax.set_yticklabels(keys)
    except:
        keys = np.char.replace(keys, '_', '\_')
        ax.set_xticklabels(keys)
        ax.set_yticklabels(keys)

    fig.colorbar(cax)

    for ix in range(len(keys)):
        for iy in range(len(keys)):
            ax.text(iy, ix, "".join('{:.2f}'.format(corr_matrix[ix, iy])), ha="center", va="center", color="r")
