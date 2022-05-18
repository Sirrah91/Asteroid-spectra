from typing import Tuple, Union, Dict
import numpy as np
from datetime import datetime
from keras.callbacks import EarlyStopping, History, TerminateOnNaN
from pprint import pprint
from keras_visualizer import visualizer
from keras.utils.vis_utils import plot_model

from keras_tuner.tuners import BayesianOptimization, RandomSearch
from contextlib import redirect_stdout

import talos
import keras_tuner as kt

from modules.control_plots import plot_corr_matrix
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
    if params['model_type'] == 'MLP':  # fully connected
        model = MLP_model((x_train.shape[1],), params)
    elif params['model_type'] == 'CNN':  # convolutional
        model = CNN_model((x_train.shape[1], 1), params)
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
        early_stopping_monitor = EarlyStopping(monitor=main_acc_name, mode=direction,
                                               patience=params['num_epochs'] * coef, restore_best_weights=True)

        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=params['num_epochs'],
                            batch_size=params['batch_size'], validation_batch_size=len(y_val), shuffle=True,
                            callbacks=[TerminateOnNaN(), early_stopping_monitor], verbose=verbose)
    else:
        history = model.fit(x_train, y_train, validation_data=None, epochs=params['num_epochs'], shuffle=True,
                            batch_size=params['batch_size'], callbacks=[TerminateOnNaN()], verbose=verbose)

    if for_tuning:  # needed for Talos, not Tuner
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

    if not quiet:
        print('\nEvaluation on the data:')

        # Evaluate the train data
        acc = rmse(y_train, model.predict(x_train)).numpy()
        print_accuracy(acc, 'train')

        if val_portion > 0:
            # Evaluate the validation data
            acc = rmse(y_val, model.predict(x_val)).numpy()
            print_accuracy(acc, 'validation')
        print()  # Empty line

    return model_name


plot_correlation_matrix = True


def tune_hp_talos(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray) -> None:
    scan_object = talos.Scan(x_train, y_train,
                             params=p_for_tuning, model=train,
                             x_val=x_val, y_val=y_val,
                             # does the reduction work well?
                             reduction_method='correlation',
                             reduction_interval=50,
                             reduction_window=25,
                             reduction_threshold=0.2,
                             reduction_metric=main_acc_name,
                             minimize_loss=direction == 'min',
                             #
                             fraction_limit=0.9999999,  # to make the search random
                             experiment_name='tuning_HP_Talos')

    data_all = scan_object.data
    best = data_all.sort_values(main_acc_name, ascending=direction == 'min').iloc[0][-len(p_for_tuning):]
    pprint(best)

    if plot_correlation_matrix:
        # collect data for correlation matrix and plot it
        """
        # if you load a csv (you MUST have the headers in correct columns)
        import pandas as pd
        data_all = pd.read_csv('./tuning_HP_Talos/021022092317.csv')
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
        indices = np.nanstd(data, axis=0) > 1e-5
        data, keys = np.transpose(data[:, indices]), keys[indices]

        plot_corr_matrix(keys, data, '_Talos')


def tune_hp_tuner(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                  method: str = 'Bayes') -> None:
    # parameter in early stopping
    coef = 0.5

    # Set early stopping monitor so the model will stop training if it does not improve anymore
    early_stopping_monitor = EarlyStopping(monitor=main_acc_name, mode=direction,
                                           patience=p_for_tuning['num_epochs'][0] * coef, restore_best_weights=True)

    hypermodel = MyHyperModel(input_shape_MLP=(x_train.shape[1],))  # input_shape_CNN is computed from this one

    now = datetime.now()
    dt_string = now.strftime("%Y%m%d%H%M%S")

    if method == 'Bayes':
        directory = 'tuning_HP_Bayes'
        max_trials = 50

        tuner = BayesianOptimization(hypermodel,
                                     objective=kt.Objective(main_acc_name, direction=direction),
                                     max_trials=max_trials,
                                     directory=directory,
                                     project_name=dt_string)
    elif method == 'Random':
        directory = 'tuning_HP_Random'
        max_trials = 100

        tuner = RandomSearch(hypermodel,
                             objective=kt.Objective(main_acc_name, direction=direction),
                             max_trials=max_trials,
                             directory=directory,
                             project_name=dt_string)
    else:
        raise ValueError('Unknown method. Must be "Bayes" or "Random"')

    # tuner.search_space_summary()

    tuner.search(x_train, y_train, validation_data=(x_val, y_val), epochs=p_for_tuning['num_epochs'][0], shuffle=True,
                 validation_batch_size=len(y_val), callbacks=[TerminateOnNaN(), early_stopping_monitor], verbose=2)

    # write results to a file (file does not look nice)
    with open(project_dir + directory + '/' + dt_string + '.csv', 'w') as f:
        with redirect_stdout(f):
            tuner.results_summary(max_trials)

    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]  # raise warnings

    # Save model to project dir with a timestamp
    model_name = "".join((dt_string, '_', model_name_suffix, '.h5'))

    filename = "".join((project_dir, '/Models/chemical/', model_name))
    check_dir(filename)
    best_model.save(filename)

    if plot_correlation_matrix:
        # collect data for correlation matrix and plot it
        # to get val_loss and val_metrics
        models = tuner.get_best_models(num_models=max_trials)  # raise warnings
        hps = tuner.get_best_hyperparameters(max_trials)
        hps = [hps[i].values for i in range(len(hps))]

        '''
        # if you load a stored JSON, use this
        import json
        import os
        folder = project_dir + '/tuning_HP_Bayes/20220425113504/'
        max_trials = len(next(os.walk(folder))[1])
        hps = max_trials * [0]
        loss_metrics = np.zeros((max_trials, 1 + len(metrics)))
    
        # to get keys
        with open(folder + 'trial_' + str("{:0" + f"{len(str(max_trials))}" + "d}").format(0) + '/trial.json') as f:
            stored_json = json.load(f)
        keys_loss_metrics = np.array(list(stored_json['metrics']['metrics'].keys()))
        val_keys = ['val_' in key for key in keys_loss_metrics]
        keys_loss_metrics = keys_loss_metrics[val_keys]
    
        for i in range(max_trials):
            with open(folder + 'trial_' + str("{:0" + f"{len(str(max_trials))}" + "d}").format(i) + '/trial.json') as f:
                stored_json = json.load(f)
                hps[i] = stored_json['hyperparameters']['values']
                for j in range(len(keys_loss_metrics)):
                    loss_metrics[i, j] = stored_json['metrics']['metrics'][
                        keys_loss_metrics[j]]['observations'][0]['value'][0]
        '''

        for model_type in ['MLP', 'CNN']:
            where_models = np.array([hps[i]['model_type'] == model_type for i in range(max_trials)])

            # construct the correlation matrix only if more than one realisation were computed
            if np.sum(where_models) > 1:
                if 'stored_json' not in locals():  # you don't read JSON
                    keys_loss_metrics = np.concatenate((np.array(['val_loss']), np.array(['val_' + metric.__name__
                                                                                          for metric in metrics])))
                    loss_metrics = np.zeros((max_trials, 1 + len(metrics)))
                    for i in np.where(where_models)[0]:
                        y_pred = models[i].predict(x_val)

                        if num_minerals > 0 and subtypes:
                            loss_used = loss_Bayes(hps[i]['alpha'])
                        else:
                            loss_used = loss

                        loss_metrics[i, 0] = np.mean(loss_used(y_val, y_pred))
                        loss_metrics[i, 1:] = np.array([np.mean(metric(y_val, y_pred)) for metric in metrics])
                loss_metrics_part = loss_metrics[np.where(where_models)[0]]

                data = np.array([np.array(list(hps[i].values()), dtype=object) for i in range(max_trials)
                                 if hps[i]['model_type'] == model_type])

                # get keys (key with maximum length - maximum layers)
                which_keys = np.argmax([len(np.array(list(hps[i].keys()))) for i in np.where(where_models)[0]])
                keys = np.array(list(hps[np.where(where_models)[0][which_keys]].keys()))

                # add NaN to nodes if a layer in not there, if no drop_hid_hid is present, if no alpha is present
                ind_n_layers = np.where('n_layers' == keys)[0]

                # ideally make this one separately, unless the model makes redundant combinations
                ind_drop_hid_hid = np.where('dropout_hidden_hidden' == keys)[0]

                max_layers = np.max([data[i][ind_n_layers] for i in range(len(data))])
                for i in range(len(data)):
                    # fill it with NaNs (some might be set even though the nodes should be NaNs)
                    if len(data[i]) != len(keys):
                        data[i] = np.append(data[i], (len(keys) - len(data[i]))*[np.nan])

                    # NaNs for nodes which are not present
                    if model_type == 'MLP':
                        nan_per_layer = 1  # units
                    elif model_type == 'CNN':
                        nan_per_layer = 2  # filters and kernels
                    actual_layers = data[i][ind_n_layers]
                    if max_layers > actual_layers:
                        inds_to_be_nan = (actual_layers - max_layers) * nan_per_layer  # from back
                        data[i][inds_to_be_nan[0]:] = np.nan

                    # NaNs for drop_hid_hid if only 1 hidden layer is present
                    if data[i][ind_n_layers] == 1:
                        data[i][ind_drop_hid_hid] = np.nan

                # ony numeric values
                indices = np.array([np.isreal(data[0][i]) for i in range(len(data[0]))])
                data = np.array([data[i][indices].astype(np.float) for i in range(len(data))])
                keys = keys[indices]

                # add losses and matrices to data and keys
                data = np.concatenate((loss_metrics_part, data), axis=1)
                keys = np.concatenate((keys_loss_metrics, keys))

                # if only one option in a parameter, omit it
                indices = np.nanstd(data, axis=0) > 1e-5
                data, keys = np.transpose(data[:, indices]), keys[indices]

                plot_corr_matrix(keys, data, '_Tuner_' + method + model_type)
