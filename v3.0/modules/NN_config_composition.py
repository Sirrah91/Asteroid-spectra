# This file contains global parameters defining the neural network
import numpy as np
from os import path

from modules.utilities import safe_arange
from modules.NN_HP import gimme_hyperparameters
from modules.NN_config_parse import gimme_model_grid, gimme_used_quantities, config_check, used_to_bin, gimme_num_labels
from modules._constants import _sep_out

comp_output_setup = {
    "minerals": np.array([True,  # olivine
                          True,  # orthopyroxene
                          True,  # clinopyroxene
                          False]),  # plagioclase

    "endmembers": [[True, True],  # Fa, Fo; OL
                   [True, True, False],  # Fs, En, Wo; OPX
                   [True, True, True],  # Fs, En, Wo; CPX
                   [False, False, False]]  # An, Ab, Or; PLG
}

# re-interpolate input data to different resolution (see reinterpolate_data in load_data.py)
comp_grid_setup = {
    "instrument": None,  # "ASPECT_vis-nir1-nir2-swir_30",

    "interpolate_to": "full",  # "full", "Itokawa", "Eros"; only if instrument is None

    # used when instrument is None and interpolate_to is unknown
    "wvl_grid": safe_arange(650., 2450., 30., endpoint=True),
    "wvl_norm": 650.  # float, None, or "adaptive"
}

comp_filtering_setup = {
    "use_pure_only": False,  # load pure minerals only

    # load mixtures of the pure ones only
    # e.g. use_pure_only = True and minerals_all = [True, True, False, False] and use_mix_of_the_pure_ones = True
    # lead to load OL, OPX, and OL+OPX binary mixtures (i.e. not meteorites etc.)
    # if use_pure_only == False, this option is not used
    "use_mix_of_the_pure_ones": False,

    "lim_vol_part": 0.65,  # remove samples with at least x vol fraction of wanted minerals

    # LOWER limits of end-member amounts (to have reasonable absorption bands)
    # these are also used to remove samples with high-iron phase that is not in "minerals"
    "chem_limits": {"OL": {"Fa": 3.},  # lower limits of iron in olivine
                    "OPX": {"Fs (OPX)": 5.},  # lower limits of iron in orthopyroxene
                    "CPX": {"Fs (CPX)": 5.}},  # lower limits of iron in clinopyroxene

    "remove_high_iron_unwanted": True,  # more iron than the limit => remove it
    "keep_if_not_used": False,  # it the chemical composition is not used, ignore limits

    "red_thresh": 5.  # threshold for normalised reflectance (redder spectra are deleted)
}

comp_data_split_setup = {
    "val_portion": 0.2,  # Set the fraction of data for validation
    "test_portion": 0.0  # Set the fraction of data for tests
}

comp_model_setup = {
    "metrics": ["mse"],  # must be in custom_objects in custom_objects in NN_losses_metrics_activations.py

    # important for hp tuning and early stopping
    "monitoring": {"objective": "val_loss",  # if is not loss, must be included in custom_objects
                   "direction": "min"  # minimise or maximise the objective (for HP tuning)?
                   },

    "trim_mean_cut": 0.2,  # parameter of trim_mean in evaluation

    "model_subdir": "composition"  # subdirectory where to save models
}

#
# DO NOT CHANGE THE PART BELOW (unless you know the consequences)
#

# data grid
comp_grid = gimme_model_grid(**comp_grid_setup)
model_grid = comp_grid["model_grid"]

# used minerals and end-members
minerals_used, endmembers_used = gimme_used_quantities(**comp_output_setup)
bin_code = used_to_bin(used_minerals=minerals_used, used_endmembers=endmembers_used)
comp_output_setup["used_minerals"], comp_output_setup["used_endmembers"] = minerals_used, endmembers_used
comp_output_setup["num_labels"] = gimme_num_labels(used_minerals=minerals_used, used_endmembers=endmembers_used)
comp_output_setup["bin_code"] = bin_code

# hyperparameters
p = gimme_hyperparameters(for_tuning=False)(composition_or_taxonomy="composition", grid_option=model_grid)
comp_model_setup["params"] = p

# model name
comp_model_setup["model_subdir"] = path.join(comp_model_setup["model_subdir"], model_grid)
comp_model_setup["model_name"] = f"{p['model_type']}{_sep_out}{model_grid}{_sep_out}{bin_code}"

# names of possible labels
mineral_names = ["olivine", "orthopyroxene", "clinopyroxene", "plagioclase"]
mineral_names_short = ["OL", "OPX", "CPX", "PLG"]

endmember_names = [["Fa", "Fo"], ["Fs (OPX)", "En (OPX)", "Wo (OPX)"],
                   ["Fs (CPX)", "En (CPX)", "Wo (CPX)"], ["An", "Ab", "Or"]]

config_check(output_setup=comp_output_setup, grid_setup=comp_grid,
             data_split_setup=comp_data_split_setup, model_options=comp_model_setup)
