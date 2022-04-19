import numpy as np
from typing import Dict
from keras import backend as K
from keras.layers import Input, Dense, Conv1D, LSTM, ConvLSTM1D, Lambda
from keras.layers import MaxPooling1D, AveragePooling1D, Flatten, BatchNormalization, Dropout
from keras.models import Sequential, Model, Functional
from keras import regularizers
import tensorflow_probability as tfp

from modules.NN_losses_metrics_activations import *

tfd = tfp.distributions
tfpl = tfp.layers

# metrics = [mse, rmse, median, mae, Lp_norm, r2, sam]
metrics = [mse]


def FC_model(x_train: np.ndarray, params: Dict) -> Functional:
    # Input layer
    inputs = Input(shape=(x_train.shape[1],))
    bs_norm = BatchNormalization()(inputs)

    # Dropout between input and the first hidden
    drop = Dropout(params['dropout_input_hidden'])(bs_norm)

    # Adding the hidden layers
    for i in range(params['n_layers']):
        dense = Dense(params['n_nodes'][i], activation=params['input_activation'],
                      kernel_regularizer=regularizers.l1(l1=params['lambda1']),
                      bias_regularizer=regularizers.l1(l1=params['lambda1']))(drop)
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
    model.compile(loss=loss, optimizer=params['optimizer'](learning_rate=params['learning_rate']),
                  metrics=metrics)

    return model


def CNN_model(x_train: np.ndarray, params: Dict) -> Functional:
    # Input layer
    inputs = Input(shape=(x_train.shape[1], 1))
    bs_norm = BatchNormalization()(inputs)

    # Dropout between input and the first hidden
    drop = Dropout(params['dropout_input_hidden'])(bs_norm)

    # Adding the CNN hidden layers
    for i in range(params['n_layers']):
        conv = Conv1D(filters=params['n_nodes'][i], kernel_size=params['kern_size'], padding='valid',
                      activation=params['input_activation'],
                      kernel_regularizer=regularizers.l1(l1=params['lambda1']),
                      bias_regularizer=regularizers.l1(l1=params['lambda1']))(drop)
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
    model.compile(loss=loss, optimizer=params['optimizer'](learning_rate=params['learning_rate']),
                  metrics=metrics)

    return model


def RNN_model(x_train: np.ndarray, params: Dict) -> Functional:
    # Input layer
    inputs = Input(shape=(x_train.shape[1], 1))
    bs_norm = BatchNormalization()(inputs)

    # Dropout between input and the first hidden
    drop = Dropout(params['dropout_input_hidden'])(bs_norm)

    # Adding the hidden layers
    for i in range(params['n_layers']):
        recurrent = LSTM(params['n_nodes'][i], activation=params['input_activation'],
                         kernel_regularizer=regularizers.l1(l1=params['lambda1']))(drop)
        bs_norm = BatchNormalization()(recurrent)
        # Dropout layer for stabilisation of the network
        if i < params['n_layers'] - 1:  # Las layer has different dropout
            drop = Dropout(params['dropout_hidden_hidden'])(bs_norm)
        else:
            drop = Dropout(params['dropout_hidden_output'])(bs_norm)

    # Number of nodes in output layer
    outputs = Dense(num_labels, activation=output_activation)(drop)
    
    model = Model(inputs=inputs, outputs=outputs)

    # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
    model.compile(loss=loss, optimizer=params['optimizer'](learning_rate=params['learning_rate']),
                  metrics=metrics)

    return model


def RCNN_model(x_train: np.ndarray, params: Dict) -> Functional:
    # Input layer
    inputs = Input(shape=(x_train.shape[1], 1))
    bs_norm = BatchNormalization()(inputs)

    # Dropout between input and the first hidden
    drop = Dropout(params['dropout_input_hidden'])(bs_norm)

    # Adding the CNN hidden layers
    for i in range(params['n_layers']):
        rc = ConvLSTM1D(filters=params['n_nodes'][i], kernel_size=params['kern_size'],
                        activation=params['input_activation'], padding='same')(drop)
        bs_norm = BatchNormalization()(rc)
        pool = MaxPooling1D(pool_size=2)(bs_norm)
        # Dropout layer for stabilisation of the network
        if i < params['n_layers'] - 1:  # Las layer has different dropout
            drop = Dropout(params['dropout_hidden_hidden'])(pool)

    flat = Flatten()(pool)
    drop = Dropout(params['dropout_hidden_output'])(flat)

    # Number of nodes in output layer
    outputs = Dense(num_labels, activation=output_activation)(drop)

    # normalisation of the results
    # outputs = Lambda(lambda t: t/(tf.linalg.norm(t, ord=1) + K.epsilon()))(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
    model.compile(loss=loss, optimizer=params['optimizer'](learning_rate=params['learning_rate']),
                  metrics=metrics)

    return model


def CNN_FC_model(x_train: np.ndarray, params: Dict) -> Functional:
    # Input layer
    inputs = Input(shape=(x_train.shape[1], 1))
    bs_norm = BatchNormalization()(inputs)

    # Dropout between input and the first hidden
    drop = Dropout(params['dropout_input_hidden'])(bs_norm)

    # Adding the CNN hidden layers
    for i in range(params['n_layers'][0]):
        conv = Conv1D(filters=params['n_nodes'][i], kernel_size=params['kern_size'],
                      activation=params['input_activation'], padding='same')(drop)
        bs_norm = BatchNormalization()(conv)
        # pool = MaxPooling1D(pool_size=2)(bs_norm)
        pool = bs_norm
        # Dropout layer for stabilisation of the network
        if i < params['n_layers'][0] - 1:
            drop = Dropout(params['dropout_hidden_hidden'])(pool)

    flat = Flatten()(drop)
    drop = Dropout(params['dropout_hidden_hidden'])(flat)

    # Adding the FC hidden layers
    for i in range(params['n_layers'][1]):
        dense = Dense(params['n_nodes'][i], activation=params['input_activation'],
                      kernel_regularizer=regularizers.l1(l1=params['lambda1']))(drop)
        bs_norm = BatchNormalization()(dense)
        # Dropout layer for stabilisation of the network
        if i < params['n_layers'][1] - 1:  # Las layer has different dropout
            drop = Dropout(params['dropout_hidden_hidden'])(bs_norm)
        else:
            drop = Dropout(params['dropout_hidden_output'])(bs_norm)

    # Number of nodes in output layer
    outputs = Dense(num_labels, activation=output_activation)(drop)

    # normalisation of the results
    # outputs = Lambda(lambda t: t/(tf.linalg.norm(t, ord=1) + K.epsilon()))(outputs)

    model = Model(inputs=inputs, outputs=outputs)

    # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
    model.compile(loss=loss, optimizer=params['optimizer'](learning_rate=params['learning_rate']),
                  metrics=metrics)

    return model


def two_thread_output_model(x_train: np.ndarray, params: Dict) -> Functional:
    # Input layer
    inputs = Input(shape=(x_train.shape[1], 1))
    bs_norm = BatchNormalization()(inputs)

    # Dropout between input and the first hidden
    extract = Dropout(params['dropout_input_hidden'])(bs_norm)

    # modal output
    conv = Conv1D(filters=params['n_nodes'][0], kernel_size=params['kern_size'],
                  activation=params['input_activation'], padding='same')(extract)
    bs_norm = BatchNormalization()(conv)
    # pool = MaxPooling1D(pool_size=2)(bs_norm)
    pool = bs_norm
    flat = Flatten()(pool)
    drop = Dropout(params['dropout_hidden_output'])(flat)

    """
    dense = Dense(params['n_nodes'][0] // 4, activation=params['input_activation'],
                  kernel_regularizer=regularizers.l1(l1=params['lambda1']))(extract)
    bs_norm = BatchNormalization()(dense)
    drop = Dropout(params['dropout_hidden_output'])(bs_norm)
    """

    output1 = Dense(num_minerals, activation='softmax')(drop)

    # chemical output
    conv = Conv1D(filters=params['n_nodes'][0], kernel_size=params['kern_size'],
                  activation=params['input_activation'], padding='same')(extract)
    bs_norm = BatchNormalization()(conv)
    # pool = MaxPooling1D(pool_size=2)(bs_norm)
    pool = bs_norm
    flat = Flatten()(pool)
    drop = Dropout(params['dropout_hidden_output'])(flat)

    """
    dense = Dense(params['n_nodes'][0] // 2, activation=params['input_activation'],
                  kernel_regularizer=regularizers.l1(l1=params['lambda1']))(extract)
    bs_norm = BatchNormalization()(dense)
    drop = Dropout(params['dropout_hidden_output'])(bs_norm)
    """

    output2 = Dense(np.sum(subtypes), activation=output_activation)(drop)

    outputs = K.concatenate((output1, output2), axis=1)

    # normalisation of the results
    # outputs = Lambda(lambda t: t/(tf.linalg.norm(t, ord=1) + K.epsilon()))(outputs)

    # output
    model = Model(inputs=inputs, outputs=outputs)

    # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
    model.compile(loss=loss, optimizer=params['optimizer'](learning_rate=params['learning_rate']),
                  metrics=metrics)

    return model


# Prior - diagonal MVN ~ N(0, 1)
def prior(kernel_size: int, bias_size: int, dtype=None) -> Sequential:
    n = kernel_size + bias_size

    prior_model = tf.keras.Sequential([

        tfpl.DistributionLambda(
            # Note: Our prior is a non-trainable distribution
            lambda t: tfd.MultivariateNormalDiag(loc=tf.zeros(n), scale_diag=tf.ones(n)))
    ])

    return prior_model


# Posterior
def posterior(kernel_size: int, bias_size: int, dtype=None) -> Sequential:
    n = kernel_size + bias_size

    posterior_model = tf.keras.Sequential([

        tfpl.VariableLayer(tfpl.MultivariateNormalTriL.params_size(n), dtype=dtype),
        tfpl.MultivariateNormalTriL(n)
    ])

    return posterior_model


# Define the model
def aleatoric_model(x_train: np.ndarray, params: Dict) -> Functional:
    inputs = Input(shape=(x_train.shape[1],))
    bs_norm = BatchNormalization()(inputs)

    # Dropout between input and the first hidden
    drop = Dropout(params['dropout_input_hidden'])(bs_norm)

    # Adding the hidden layers
    for i in range(params['n_layers']):
        dense = tfpl.DenseVariational(params['n_nodes'][i], make_prior_fn=prior, make_posterior_fn=posterior,
                                      kl_weight=1 / x_train.shape[0],
                                      # Normalizing to scale the D_KL term in ELBO properly when using minibatches.
                                      kl_use_exact=False,  # could be `True` in this case, but we go for estimated value
                                      activation=params['input_activation'],
                                      activity_regularizer=regularizers.l1(l1=params['lambda1']))(drop)

        bs_norm = BatchNormalization()(dense)
        # Dropout layer for stabilisation of the network
        if i < params['n_layers'] - 1:  # Las layer has different dropout
            drop = Dropout(params['dropout_hidden_hidden'])(bs_norm)
        else:
            drop = Dropout(params['dropout_hidden_output'])(bs_norm)

    # Number of nodes in output layer
    outputs = tfpl.DenseVariational(num_labels, make_prior_fn=prior, make_posterior_fn=posterior,
                                    kl_weight=1 / params['n_nodes'][i],
                                    # Normalizing to scale the D_KL term in ELBO properly when using minibatches.
                                    kl_use_exact=False,  # could be `True` in this case, but we go for estimated value
                                    activation=output_activation)(drop)

    model = Model(inputs=inputs, outputs=outputs)

    # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
    model.compile(loss=loss, optimizer=params['optimizer'](learning_rate=params['learning_rate']),
                  metrics=metrics)

    return model
