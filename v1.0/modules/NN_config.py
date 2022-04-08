# This file contains global parameters defining the neural network
from tensorflow.keras.optimizers import Adam
import numpy as np
from modules.utilities import flatten_list
import warnings

# load pure minerals only
use_pure_only = False

# load mixtures of the pure ones only
# eg use_pure_only = True and use_minerals = [True, True, False, False] and use_mix_of_the_pure_ones = True
# leads to load OL, OPX, and OL+OPX binary mixtures (ie not meteorites etc)
# if use_pure0only = False, this option is not used
# if np.all(use_minerals) == True, both of the options have no effect
use_mix_of_the_pure_ones = False

use_minerals = np.array((True,  # olivine
                         True,  # orthopyroxene
                         True,  # clinopyroxene
                         False))  # plagioclase

subtypes_all = [[True, True],  # Fa, Fo; OL
                [True, True, False],  # Fs, En, Wo; OPX
                [True, True, True],  # Fs, En, Wo; CPX
                [False, False, False]]  # An, Ab, Or; PLG

# lower limits of iron in samples (to have reasonable absorption bands)
Fa_lim, Fs_lim = 3, 5
red_thresh = 5  # threshold for normalised reflectance (redder spectra are deleted)
lim_vol_part = 0.65  # samples with at least 65 vol% of wanted minerals

val_portion = 0.20  # Set the percentage of data for validation
test_portion = 0.20  # Set the percentage of data for tests

modal_in_wt_percent = False  # NOT FULLY IMPLEMENTED YET

# Hyper parameters
p = {
    'model_type': 'CNN',  # Convolutional (CNN) or fully connected (FC)
    'n_layers': 2,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
    'n_nodes': [24, 8],  # Number of nodes in hidden layers / number of filters for CNN
    'kern_size': 5,  # Width of the kernel (only if CNN)
    'lambda1': 0.005,  # L1 trade-off parameter
    'alpha': 0.1,  # Trade off between modal and chemical misfits (modal + alpha x chemical)
    'dropout_input_hidden': 0.0,  # Dropout
    'dropout_hidden_hidden': 0.3,  # Dropout
    'dropout_hidden_output': 0.4,  # Dropout
    'learning_rate': 0.0005,  # Learning rate
    'num_epochs': 5000,  # Number of epochs
    'batch_size': 8,  # Bath size
    'optimizer': Adam,  # Optimizer for the network
    'input_activation': 'relu',  # Activation function of input layer
    'output_activation': 'sigmoid',  # Activation function of output layer (relu, softmax, sigmoid)
    'tuning': 0  # This is parameters for evaluation; do not change
}

p_for_tuning = {  # In talos format
    #
    # [list, of, samples] or
    # tuple (start, stop, steps); no last element, i.e. (20, 60, 4) = np.linspace(20, 60, 5)[:-1]
    #
    'model_type': ['CNN'],  # Convolutional (CNN) or fully connected (FC)
    'n_layers': [2, 3],  # Number of hidden layers
    # Number of nodes in hidden layers [[first, second, ...], [...], ...]
    'n_nodes': [[16, 8, 8], [24, 8, 8], [24, 16, 8], [8, 8, 4]],
    'kern_size': [3, 5],  # Width of the kernel (only if CNN)
    'lambda1': [0.01, 0.1, 1.0],  # L1 trade-off parameter
    'alpha': [0.1],  # Trade off between modal and chemical misfits (modal + alpha x chemical)
    'dropout_input_hidden': [0.0],  # Dropout
    'dropout_hidden_hidden': [0.0, 0.2],  # Dropout
    'dropout_hidden_output': [0.2, 0.3, 0.4],  # Dropout
    'learning_rate': [0.0005],  # Learning rate
    'num_epochs': [600],  # Number of epochs 600
    'batch_size': [8],  # Bath size
    'optimizer': [Adam],  # Optimizer for the network
    'input_activation': ['relu', 'elu'],  # Activation function of input layer
    'output_activation': ['sigmoid', 'softmax'],  # Activation function of output layer (relu, softmax, sigmoid)
    'tuning': [1]  # This is parameters for evaluation; do not change
}

# Can I ask for chemical if I don't ask for modal?
chem_without_modal = False  # doesn't work yet. Need to rewrite losses and other functions with 'w * z' terms

trimmed = 0.2  # parameter of trim_mean in evaluation

project_dir = '/home/dakorda/Python/NN/'  # Directory which contains Datasets, Modules, etc.
model_name_suffix = "".join((p['model_type'], ''))  # model_name = timestamp + _ + suffix + .h5

show_control_plot = True  # True for showing control plots, False for not
verb = 0  # Set value for verbose: 0=no print, 1=full print, 2=simple print

#
# DO NOT CHANGE THE PART BELOW
#

if not chem_without_modal:
    # if there is no mineral, return error
    if np.sum(use_minerals) == 0:
        raise ValueError('There is no mineral in use_minerals.')

num_labels_all = len(flatten_list(subtypes_all)) + len(use_minerals)  # One for each number
minerals_all = np.array(['olivine', 'orthopyroxene', 'clinopyroxene', 'plagioclase'])

# if there is only one end-member for a given mineral, the information is redundant and worsen the optimisation
if chem_without_modal:
    subtypes_for_usage = [subtypes_all[k] if np.sum(subtypes_all[k]) > 1 else
                          len(subtypes_all[k]) * [False] for k in range(len(use_minerals))]
    subtypes = [np.sum(subtypes_for_usage[k]) for k in range(len(use_minerals))
                if np.sum(subtypes_all[k]) > 1]
else:
    subtypes_for_usage = [subtypes_all[k] if (use_minerals[k] and np.sum(subtypes_all[k]) > 1) else
                          len(subtypes_all[k]) * [False] for k in range(len(use_minerals))]
    subtypes = [np.sum(subtypes_for_usage[k]) for k in range(len(use_minerals))
                if (use_minerals[k] and np.sum(subtypes_all[k]) > 1)]

# if there is no end-member, return warning
if np.sum(subtypes) == 0:
    warnings.warn("Warning. There is no end-member in subtypes_all.", DeprecationWarning)

# these are needed elsewhere
if chem_without_modal:
    subtypes_all_used = [subtypes_for_usage[k] for k in range(len(use_minerals))]
else:
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

# these are needed due to `abandon' regions
if subtypes_for_usage[1][2]:
    opx_position = np.where(minerals_all[use_minerals_all] == 'orthopyroxene')[0][0].astype(np.int8)
    use_opx_pen = 1
else:
    opx_position, use_opx_pen = 0, 0  # use first index which is always present

if subtypes_for_usage[2][2]:
    cpx_position = np.where(minerals_all[use_minerals_all] == 'clinopyroxene')[0][0].astype(np.int8)
    use_cpx_pen = 1
else:
    cpx_position, use_cpx_pen = 0, 0  # use first index which is always present

if np.array(subtypes_for_usage[3]).all():
    plg_position = np.where(minerals_all[use_minerals_all] == 'plagioclase')[0][0].astype(np.int8)
    use_plg_pen, use_plg_pen_idx = 1, (0, 1, 2)
else:
    plg_position, use_plg_pen, use_plg_pen_idx = 0, 0, (0, 0, 0)  # use first index which is always present

if subtypes:  # subtypes is not empty
    subtypes_pen = subtypes
else:  # subtypes is empty
    subtypes_pen = [0]

"""
print()

# only if OPX is in the set and Wo is in OPX set
if subtypes_for_usage[1][2]:
    print('OPX penalisation should be active.')

# only if CPX is in the set and Wo is in CPX set
if subtypes_for_usage[2][2]:
    print('CPX penalisation should be active.')

# only if PLG is in the set and An, Ab, Or is in PLG set (you should never run An + Or without Ab)
if np.array(subtypes_for_usage[3]).all():
    print('PLG penalisation should be active.')

print()
"""
