# This file contains global parameters defining the neural network
import numpy as np
import warnings
from modules.NN_HP import gimme_hyperparameters

# load pure minerals only
use_pure_only = False

# load mixtures of the pure ones only
# e.g. use_pure_only = True and minerals_all = [True, True, False, False] and use_mix_of_the_pure_ones = True
# lead to load OL, OPX, and OL+OPX binary mixtures (i.e. not meteorites etc.)
# if use_pure_only == False, this option is not used
use_mix_of_the_pure_ones = False

minerals_all = np.array([True,  # olivine
                         True,  # orthopyroxene
                         True,  # clinopyroxene
                         False])  # plagioclase

endmembers_all = [[True, True],  # Fa, Fo; OL
                  [True, True, False],  # Fs, En, Wo; OPX
                  [True, True, True],  # Fs, En, Wo; CPX
                  [False, False, False]]  # An, Ab, Or; PLG

# re-interpolate input data to different resolution (see reinterpolate_data in utilities_spectra.py)
# None for no change; other possibilities are "ASPECT", "ASPECT_swir" "Itokawa", "Eros", "Didymos"
interpolate_to = "Eros"
new_wvl_grid, new_wvl_grid_normalisation = None, None  # None for no change; "interpolate_to" must be None to use this

Fa_lim, Fs_lim = 3., 5.  # lower limits of iron in samples (to have reasonable absorption bands)
red_thresh = 5.  # threshold for normalised reflectance (redder spectra are deleted)
lim_vol_part = 0.65  # samples with at least 65 vol% of wanted minerals

val_portion = 0.0  # Set the fraction of data for validation
test_portion = 0.0  # Set the fraction of data for tests

# Hyper parameters
p = gimme_hyperparameters(for_tuning=False)(composition_or_taxonomy="composition", grid_option=interpolate_to)

modal_in_wt_percent = False  # NOT FULLY IMPLEMENTED YET

trim_mean_cut = 0.2  # parameter of trim_mean in evaluation

model_subdir = "composition"  # directory where to save models
# model_subdir = "accuracy_test"  # directory where to save models

# model_name equal to time_stamp + _ + model_type + _ + suffix + .h5
model_grid = interpolate_to if interpolate_to is not None else "full"
model_name_suffix = "".join((p["model_type"], "_", model_grid))

show_result_plot = False  # True for showing and saving of results plots, False for not
show_control_plot = False  # True for showing and saving of control plots, False for not
verb = 0  # Set value for verbose: 0 = no print, 1 = full print, 2 = simple print

if model_subdir == "accuracy_test":
    val_portion = 0.0  # Set the fraction of data for validation
    show_result_plot = False  # True for showing and saving of results plots, False for not
    show_control_plot = False  # True for showing and saving of control plots, False for not
    verb = 0  # Set value for verbose: 0 = no print, 1 = full print, 2 = simple print

#
# DO NOT CHANGE THE PART BELOW (unless you know the consequences)
#

if np.sum(minerals_all) == 0:
    raise ValueError('There is no mineral in "minerals_all".')

mineral_names = ["olivine", "orthopyroxene", "clinopyroxene", "plagioclase"]
mineral_names_short = ["OL", "OPX", "CPX", "PLG"]
endmember_names = [["Fa", "Fo"], ["Fs (OPX)", "En (OPX)", "Wo (OPX)"],
                   ["Fs (CPX)", "En (CPX)", "Wo (CPX)"], ["An", "Ab", "Or"]]

# if there is only one end-member for a given mineral, the information is redundant and worsen the optimisation
endmembers_used = [endmember if (mineral and np.sum(endmember) > 1) else len(endmember) * [False]
                   for mineral, endmember in zip(minerals_all, endmembers_all)]
endmembers_counts = np.array([np.sum(endmember) for endmember in endmembers_used])

# if there is only one mineral, the modal information is redundant and worsen the optimisation
if np.sum(minerals_all) == 1:
    minerals_used = np.array([False] * len(minerals_all))
else:
    minerals_used = minerals_all

# if there is no end-member, return warning
if np.sum(endmembers_counts) == 0:
    warnings.warn('Warning. There is no valid end-member in "endmembers_all".')

num_minerals = int(np.sum(minerals_used))
num_labels = num_minerals + int(np.sum(endmembers_counts) ) # One for each number

if num_labels == 0:
    raise ValueError("There is no valid label.")
