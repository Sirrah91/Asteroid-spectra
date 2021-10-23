# This file contains global parameters defining the neural network
from tensorflow.keras.optimizers import Adam
import numpy as np
from modules.collect_data import flatten_list
import warnings

# NN_losses_metrics_activations.my_loss_v? -- uncomment forbidden regions if necessary

# choose between RELAB and Tuomas' data
use_rebal_data = False

use_minerals = np.array((True,  # olivine
                         True,  # orthopyroxene
                         True,  # clinopyroxene
                         True))  # plagioclase

subtypes_all = [[True, True],  # Fa, Fo; OL
                [True, True, False],  # Fs, En, Wo; OPX
                [True, True, True],  # Fs, En, Wo; CPX
                [False, False, False]]  # An, Ab, Or; PLG

val_portion = 0.15  # Set the percentage of data for validation
test_portion = 0.00  # Set the percentage of data for tests

# Hyper parameters
p = {
    'model_type': 'FC',  # convolutional (CNN) or fully connected (FC)
    'nodes_1': 25,  # Number of nodes in input layer (more than output, less than input)
    'kern_size_1': 7,  # Width of the first kernel (only if CNN)
    'lambda1': 0.0001,  # 0.001, #0.0001,  # Trade-off parameter
    'dropout_input': 0.0,  # Dropout
    'dropout_hidden': 0.0,  # Dropout
    'learning_rate': 0.0001,  # Learning rate
    'num_epochs': 500,  # Number of epochs
    'batch_size': 64,  # Bath size
    'optimizer': Adam,  # Optimizer for the network (adam, adamax, nadam, adadelta, adagrad, rmsprop, sgd)
    'input_activation': 'relu',  # Activation function of input layer
    'tuning': 0  # These are parameters for evaluation
}

p_for_tuning = {  # In talos format
    #
    # [list, of, samples] or
    # tuple (start, stop, steps); no last element, i.e. (20, 60, 4) = np.linspace(20, 60, 5)[:-1]
    #
    'model_type': ['FC', 'CNN'],  # convolutional (CNN) or fully connected (FC)
    'nodes_1': [20, 25],  # Number of nodes in input layer (more than output, less than input)
    'kern_size_1': [7, 15],  # Width of the first kernel (only if CNN)
    'lambda1': [0.0001, 0.001],  # Trade-off parameter
    'dropout_input': [0.0, 0.1, 0.2],  # Dropout
    'dropout_hidden': [0.0, 0.2, 0.4],  # Dropout
    'learning_rate': [0.0001, 0.001],  # Learning rate
    'num_epochs': [250],  # Number of epochs
    'batch_size': [32, 64],  # Bath size
    'optimizer': [Adam],  # Optimizer for the network
    'input_activation': ['relu'],  # Activation function of input layer
    'tuning': [1]  # These are parameters for tuning
}

trimmed = 0.2  # parameter of trim_mean in evaluation
alpha = 1  # trade off between modal and chemical misfits

project_dir = '/home/dakorda/Python/NN/'  # Directory which contains Datasets, Modules, etc.
model_name_suffix = "".join((p['model_type']))  # model_name = timestamp + _ + suffix + .h5

show_control_plot = True  # True for showing control plots, False for not
verb = 0  # Set value for verbose: 0=no print, 1=full print, 2=simple print

#
# DO NOT CHANGE THE PART BELOW
#

# if there is no mineral, return error
if np.sum(use_minerals) == 0:
    raise ValueError('There is no mineral in use_minerals.')

num_labels_all = len(flatten_list(subtypes_all)) + len(use_minerals)  # One for each number
minerals_all = np.array(['olivine', 'orthopyroxene', 'clinopyroxene', 'plagioclase'])

# these are needed due to ``abandon'' regions
if use_minerals[2]:
    cpx_position = np.where(minerals_all[use_minerals] == 'clinopyroxene')[0][0].astype(np.int8)
if use_minerals[3]:
    plg_position = np.where(minerals_all[use_minerals] == 'plagioclase')[0][0].astype(np.int8)

# if there is only one end-member for a given mineral, the information is redundant and worsen the optimisation
subtypes_for_usage = [subtypes_all[k] if (use_minerals[k] and np.sum(subtypes_all[k]) > 1) else
                      len(subtypes_all[k]) * [False] for k in range(len(use_minerals))]
subtypes = [np.sum(subtypes_for_usage[k]) for k in range(len(use_minerals))
            if (use_minerals[k] and np.sum(subtypes_all[k]) > 1)]

# if there is no end-member, return warning
if np.sum(subtypes) == 0:
    warnings.warn("Warning. There is no end-member in subtypes_all.", DeprecationWarning)

# these are needed elsewhere
subtypes_all_used = [subtypes_for_usage[k] for k in range(len(use_minerals)) if use_minerals[k]]
use_minerals_all = use_minerals[:]
minerals_used = minerals_all[np.where(np.array([np.sum(subtypes_for_usage[k]) for k in range(len(use_minerals))]) > 1)]

# if there is only one mineral, the modal information is redundant and worsen the optimisation
if np.sum(use_minerals) == 1:
    use_minerals = np.array([False] * len(use_minerals))

minerals = minerals_all[np.where(use_minerals)]

used_indices = np.concatenate((use_minerals, flatten_list(subtypes_for_usage)))
num_minerals = len(minerals)
num_labels = sum(subtypes) + num_minerals  # One for each number
