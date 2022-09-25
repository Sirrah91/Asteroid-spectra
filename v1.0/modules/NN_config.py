# This file contains global parameters defining the neural network
import numpy as np
from modules.utilities import flatten_list
import warnings
from modules.NN_HP import gimme_hyperparameters

# load pure minerals only
use_pure_only = False

# load mixtures of the pure ones only
# e.g. use_pure_only = True and use_minerals = [True, True, False, False] and use_mix_of_the_pure_ones = True
# leads to load OL, OPX, and OL+OPX binary mixtures (ie not meteorites etc.)
# if use_pure_only == False, this option is not used
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

# re-interpolate input data to different resolution (see reinterpolate_data in NN.data.py)
interpolate_to = None  # None for no change; other possibilities are "ASPECT", "Itokawa", "Eros", "Didymos"
new_wvl_grid, new_wvl_grid_normalisation = None, None  # None for no change; interpolate_to must be None to use this

# lower limits of iron in samples (to have reasonable absorption bands)
Fa_lim, Fs_lim = 3, 5
red_thresh = 5  # threshold for normalised reflectance (redder spectra are deleted)
lim_vol_part = 0.65  # samples with at least 65 vol% of wanted minerals

val_portion = 0.20  # Set the fraction of data for validation
test_portion = 0.20  # Set the fraction of data for tests

# Hyper parameters
p = gimme_hyperparameters(composition_or_taxonomy="composition", grid_option=interpolate_to)

p_for_tuning = {  # In talos format
    #
    # [list, of, samples] or
    # tuple (start, stop, steps); no last element, i.e. (20, 60, 4) = np.linspace(20, 60, 5)[:-1]
    #
    # kern_size, dropout_hidden_hidden, and input_activation are uniform for all layers;
    # modify similarly to n_nodes if needed
    #
    # Convolutional (CNN) or dense (MLP); correctly implemented in Bayes and Random
    'model_type': ['CNN', 'MLP'],  # use one option and not both in Talos, otherwise you get redundant models
    'n_layers': [1, 2, 3],  # Number of hidden layers
    # Number of nodes in hidden layers [[first, second, ...], [...], ...]
    'n_nodes': [[16, 8, 8], [24, 8, 8], [24, 16, 8], [8, 8, 4]],
    'n_nodes_tuner': [4, 32],  # should be commented if Talos is used
    'kern_size': [3, 5],  # Width of the kernel (only if CNN)
    'dropout_input_hidden': [0.0, 0.0],  # Dropout rate
    'dropout_hidden_hidden': [0.0, 0.2, 0.5],  # Dropout rate
    'dropout_hidden_output': [0.0, 0.2, 0.3, 0.4],  # Dropout rate
    'L1_trade_off': [0.0001, 0.01, 0.1],  # L1 trade-off parameter
    'input_activation': ['relu', 'tanh', 'sigmoid'],  # Activation function of input and hidden layers
    'output_activation': ['sigmoid', 'softmax'],  # Activation function of output layer (relu, softmax, sigmoid)
    # Optimizer for the network
    'optimizer': ['Adam', 'SGD'],  # see return_optimizer and MyHyperModel.build in NN_models.py for options
    'learning_rate': [0.00005, 0.0005, 0.05],  # Learning rate
    'alpha': [0.01, 0.1, 1],  # Trade off between modal and chemical misfits
    'batch_size': [1, 4, 8, 16, 128],  # Bath size
    'num_epochs': [500],  # Number of epochs 600
    'tuning': [1],  # This is parameters for evaluation; do not change
    'tuning_method': 'Bayes'  # 'talos', 'Bayes', 'Random'
}

# Can I ask for chemical if I don't ask for modal?
chem_without_modal = False  # doesn't work yet. Need to rewrite losses and other functions with 'w * z' terms
modal_in_wt_percent = False  # NOT FULLY IMPLEMENTED YET

trimmed = 0.2  # parameter of trim_mean in evaluation

project_dir = '/home/dakorda/Python/NN/'  # Directory which contains Datasets, Modules, etc.
model_dir = 'compositional'  # directory where to save models
# model_dir = 'accuracy_test'  # directory where to save models
# model_name = time_stamp + _ + model_type + _ + suffix + .h5
model_name_suffix = interpolate_to if interpolate_to is not None else 'full'

show_result_plot = True  # True for showing and saving of results plots, False for not
show_control_plot = True  # True for showing and saving of control plots, False for not
verb = 2  # Set value for verbose: 0=no print, 1=full print, 2=simple print

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
    warnings.warn("Warning. There is no end-member in subtypes_all.")

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
