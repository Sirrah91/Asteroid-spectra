import numpy as np
from typing import Dict, Tuple
from keras import backend as K
from keras.layers import Input, Dense, Conv1D, LSTM, ConvLSTM1D, Lambda
from keras.layers import MaxPooling1D, AveragePooling1D, Flatten, BatchNormalization, Dropout
from keras.models import Sequential, Model, Functional
from keras import regularizers
import tensorflow_probability as tfp
from collections import Counter
from keras_tuner import HyperModel
import keras_tuner as kt
import tensorflow.keras.optimizers as opt

from modules.NN_losses_metrics_activations import *
from modules.NN_config import *

tfd = tfp.distributions
tfpl = tfp.layers

# metrics = [main_acc, mse, rmse, mae, quantile, Lp_norm, r2, sam]
metrics = [main_acc]

# make set of metrics unique
counter = Counter(metrics)
metrics = list(counter.keys())


def return_optimizer(params: Dict):
    # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
    if params['optimizer'] == 'Adam':
        optimizer = opt.Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'SGD':
        optimizer = opt.SGD(learning_rate=params['learning_rate'], nesterov=True)
    else:
        raise ValueError('Unknown optimizer. Add it to return_optimizer in NN_models.py')
        # add other optimisers if needed

    return optimizer


def MLP_model(input_shape: Tuple[int, ...], params: Dict) -> Functional:
    # Input layer
    inputs = Input(shape=input_shape)
    bs_norm = BatchNormalization()(inputs)

    # Dropout between input and the first hidden
    drop = Dropout(params['dropout_input_hidden'])(bs_norm)

    # Adding the hidden layers
    for i in range(params['n_layers']):
        dense = Dense(params['n_nodes'][i], activation=params['input_activation'],
                      kernel_regularizer=regularizers.l1(l1=params['L1_trade_off']),
                      bias_regularizer=regularizers.l1(l1=params['L1_trade_off'])
                      )(drop)
        bs_norm = BatchNormalization()(dense)
        # Dropout layer for stabilisation of the network
        if i < params['n_layers'] - 1:  # Las layer has different dropout
            drop = Dropout(params['dropout_hidden_hidden'])(bs_norm)
        else:
            drop = Dropout(params['dropout_hidden_output'])(bs_norm)

    # Number of nodes in output layer
    outputs = Dense(num_labels, activation=output_activation)(drop)

    # normalisation of the results
    # outputs = Lambda(lambda t: t/(tf.linalg.norm(t, ord=1) + K.epsilon()))(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
    optimizer = return_optimizer(params)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def CNN_model(input_shape: Tuple[int, ...], params: Dict) -> Functional:
    # Input layer
    inputs = Input(shape=input_shape)
    bs_norm = BatchNormalization()(inputs)

    # Dropout between input and the first hidden
    drop = Dropout(params['dropout_input_hidden'])(bs_norm)

    # Adding the CNN hidden layers
    for i in range(params['n_layers']):
        conv = Conv1D(filters=params['n_nodes'][i], kernel_size=params['kern_size'], padding='same',
                      activation=params['input_activation'],
                      kernel_regularizer=regularizers.l1(l1=params['L1_trade_off']),
                      bias_regularizer=regularizers.l1(l1=params['L1_trade_off'])
                      )(drop)
        bs_norm = BatchNormalization()(conv)
        # pool = MaxPooling1D(pool_size=2)(bs_norm)
        pool = bs_norm
        # Dropout layer for stabilisation of the network
        if i < params['n_layers'] - 1:  # Las layer has different dropout
            drop = Dropout(params['dropout_hidden_hidden'])(pool)

    flat = Flatten()(pool)
    drop = Dropout(params['dropout_hidden_output'])(flat)

    # Number of nodes in output layer
    outputs = Dense(num_labels, activation=output_activation)(drop)

    model = Model(inputs=inputs, outputs=outputs)

    # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
    optimizer = return_optimizer(params)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


class MyHyperModel(HyperModel):
    def __init__(self, input_shape_MLP: Tuple[int, ...]):
        self.input_shape_MLP = input_shape_MLP
        self.input_shape_CNN = input_shape_MLP + (1,)

    def common_hp(self, hp):
        self.model_type = hp.Choice('model_type', values=p_for_tuning['model_type'], default=p['model_type'])

        self.n_layers = hp.Int('n_layers', min_value=min(p_for_tuning['n_layers']),
                               max_value=max(p_for_tuning['n_layers']), default=p['n_layers'], step=1)

        self.drop_in_hid = hp.Float('dropout_input_hidden', min_value=min(p_for_tuning['dropout_input_hidden']),
                                    max_value=max(p_for_tuning['dropout_input_hidden']), step=0.05,
                                    default=p['dropout_input_hidden'])
        # if self.n_layers > 1:  # this does not work, why?
        self.drop_hid_hid = hp.Float('dropout_hidden_hidden', min_value=min(p_for_tuning['dropout_hidden_hidden']),
                                     max_value=max(p_for_tuning['dropout_hidden_hidden']), step=0.05,
                                     default=p['dropout_hidden_hidden'])
        self.drop_hid_out = hp.Float('dropout_hidden_output', min_value=min(p_for_tuning['dropout_hidden_output']),
                                     max_value=max(p_for_tuning['dropout_hidden_output']), step=0.05,
                                     default=p['dropout_hidden_output'])

        self.l1 = hp.Float('L1_trade_off', min_value=min(p_for_tuning['L1_trade_off']),
                           max_value=max(p_for_tuning['L1_trade_off']), default=p['L1_trade_off'], sampling='log')

        self.activation_input = hp.Choice('input_activation', values=p_for_tuning['input_activation'],
                                          default=p['input_activation'])
        self.activation_output = hp.Choice('output_activation', values=p_for_tuning['output_activation'],
                                           default=p['output_activation'])

        self.opt_choice = hp.Choice('optimizer', values=p_for_tuning['optimizer'], default=p['optimizer'])
        self.lr = hp.Float('learning_rate', min_value=min(p_for_tuning['learning_rate']),
                           max_value=max(p_for_tuning['learning_rate']), default=p['learning_rate'], sampling='log')

        if num_minerals > 0 and subtypes:
            self.alpha = hp.Float('alpha', min_value=min(p_for_tuning['alpha']), max_value=max(p_for_tuning['alpha']),
                                  default=p['alpha'], sampling='log')

        self.batch_size = hp.Int('batch_size', min_value=min(p_for_tuning['batch_size']),
                                 max_value=max(p_for_tuning['batch_size']), default=p['batch_size'], step=4)

    def build_MLP(self, hp):
        # Input layer
        inputs = Input(shape=self.input_shape_MLP)
        bs_norm = BatchNormalization()(inputs)

        # Dropout between input and the first hidden
        drop = Dropout(rate=self.drop_in_hid)(bs_norm)

        # Adding hidden layers
        for i in range(self.n_layers):
            dense = Dense(
                units=hp.Int(f'units_{i}', min_value=min(p_for_tuning['n_nodes_tuner']),
                             max_value=max(p_for_tuning['n_nodes_tuner']), default=p['n_nodes'][i], step=4),
                activation=self.activation_input,
                kernel_regularizer=regularizers.l1(l1=self.l1),
                bias_regularizer=regularizers.l1(l1=self.l1)
            )(drop)

            bs_norm = BatchNormalization()(dense)

            # Dropout layer for stabilisation of the network
            if i < self.n_layers - 1:  # Las layer has different dropout
                drop = Dropout(rate=self.drop_hid_hid)(bs_norm)

        drop = Dropout(rate=self.drop_hid_out)(bs_norm)

        # Number of nodes in output layer
        outputs = Dense(num_labels, activation=self.activation_output)(drop)

        uncompiled_model = Model(inputs=inputs, outputs=outputs)

        return uncompiled_model

    def build_CNN(self, hp):
        # Input layer
        inputs = Input(shape=self.input_shape_CNN)
        bs_norm = BatchNormalization()(inputs)

        # Dropout between input and the first hidden
        drop = Dropout(rate=self.drop_in_hid)(bs_norm)

        # Adding the CNN hidden layers
        # the same kernel size for all conv layers
        self.kernel_size = hp.Int(f'kernel_size', min_value=min(p_for_tuning['kern_size']),
                                  max_value=max(p_for_tuning['kern_size']), default=p['kern_size'], step=2)
        for i in range(self.n_layers):
            conv = Conv1D(
                filters=hp.Int(f'num_filters_{i}', min_value=min(p_for_tuning['n_nodes_tuner']),
                               max_value=max(p_for_tuning['n_nodes_tuner']), default=p['n_nodes'][i], step=4),
                kernel_size=self.kernel_size,
                padding='same',
                activation=self.activation_input,
                kernel_regularizer=regularizers.l1(l1=self.l1),
                bias_regularizer=regularizers.l1(l1=self.l1)
            )(drop)

            bs_norm = BatchNormalization()(conv)

            # Dropout layer for stabilisation of the network
            if i < self.n_layers - 1:  # Las layer has different dropout
                drop = Dropout(rate=self.drop_hid_hid)(bs_norm)

        flat = Flatten()(bs_norm)
        drop = Dropout(rate=self.drop_hid_out)(flat)

        # Number of nodes in output layer
        outputs = Dense(num_labels, activation=self.activation_output)(drop)

        uncompiled_model = Model(inputs=inputs, outputs=outputs)

        return uncompiled_model

    def build(self, hp):
        self.common_hp(hp)

        if self.model_type == 'MLP':
            with hp.conditional_scope("model_type", ['MLP']):
                model = self.build_MLP(hp)
        elif self.model_type == 'CNN':
            with hp.conditional_scope("model_type", ['CNN']):
                model = self.build_CNN(hp)

        # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
        if self.opt_choice == 'Adam':
            with hp.conditional_scope("optimizer", ['Adam']):
                optimizer_tuner = opt.Adam(learning_rate=self.lr)
        elif self.opt_choice == 'SGD':
            with hp.conditional_scope("optimizer", ['SGD']):
                optimizer_tuner = opt.SGD(learning_rate=self.lr, nesterov=True)
        else:
            raise ValueError('Unknown optimizer. Add it to MyHyperModel.build in NN_models.py')
            # add other optimisers if needed

        if num_minerals > 1:
            loss_used = loss_Bayes(self.alpha)
        else:
            loss_used = loss

        model.compile(loss=loss_used, metrics=metrics, optimizer=optimizer_tuner)

        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            # Tune batch size
            batch_size=self.batch_size,
            **kwargs,
        )
