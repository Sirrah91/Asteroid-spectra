from os import path
from modules.NN_HP import gimme_hyperparameters
from modules.NN_config_parse import gimme_model_grid, gimme_classes, config_check, cls_to_bin
from modules.utilities import safe_arange
from modules._constants import _sep_out

tax_output_setup = {
    "use_unknown_class": False  # Add extra "unknown" class for weird spectra
}

# Re-interpolate input data to different resolutions (see reinterpolate_data in load_data.py)
tax_grid_setup = {
    "instrument": None, #"ASPECT_vis-nir1-nir2-swir_30",

    "interpolate_to": "full",  # "full", "Itokawa", "Eros"; only if "instrument" is None

    # used when "instrument" is None and interpolate_to is unknown
    "wvl_grid": safe_arange(650., 2450., 30., endpoint=True),
    "wvl_norm": 650.  # float, None, or "adaptive"
}

tax_filtering_setup = {}

tax_data_split_setup = {
    "val_portion": 0.2,  # Set the fraction of data for validation
    "test_portion": 0.0  # Set the fraction of data for tests
}

tax_model_setup = {
    "metrics": ["f1_score"],  # must be in custom_objects in custom_objects in NN_losses_metrics_activations.py

    # important for HP tuning and early stopping
    "monitoring": {"objective": "val_f1_score",  # if is not loss, must be included in custom_objects
                   "direction": "max"  # minimise or maximise the objective (for HP tuning)?
                   },

    "trim_mean_cut": 0.2,  # parameter of trim_mean in evaluation

    "model_subdir": "taxonomy"  # subdirectory where to save models
}

#
# DO NOT CHANGE THE PART BELOW (unless you know the consequences)
#

# data grid
tax_grid = gimme_model_grid(**tax_grid_setup)
model_grid = tax_grid["model_grid"]

# used classes
classes = gimme_classes(model_grid=model_grid, use_unknown_class=tax_output_setup["use_unknown_class"])
bin_code = cls_to_bin(used_classes=classes)
tax_output_setup["used_classes"] = classes
tax_output_setup["num_labels"] = len(classes)
tax_output_setup["bin_code"] = bin_code

# hyperparameters
p = gimme_hyperparameters(for_tuning=False)(composition_or_taxonomy="taxonomy", grid_option=model_grid)
tax_model_setup["params"] = p

# model name
tax_model_setup["model_subdir"] = path.join(tax_model_setup["model_subdir"], model_grid)
tax_model_setup["model_name"] = f"{p['model_type']}{_sep_out}{model_grid}{_sep_out}{bin_code}"

# part of the file names
num_labels_in_file = len(classes) - ("Other" in classes)

config_check(output_setup=tax_output_setup, grid_setup=tax_grid,
             data_split_setup=tax_data_split_setup, model_options=tax_model_setup)
