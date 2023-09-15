# THIS FILE CONTAINS VARIABLES COMMON TO ALL FUNCTIONS (PATHS, CONSTANTS, ETC.)
import numpy as np
from os import path

# Base directory of the project
_project_dir = "/home/dakorda/Python/NN"  # do not use relative path if you create files from "modules" folder

# subdirs in _project_dir (useful for backup)
_subdirs = {"modules": "modules",
            "models": "models",
            "HP_tuning": "tuning_HP",

            "datasets": "datasets",
            "accuracy_tests": "accuracy_tests",
            "asteroid_images": "asteroid_images",

            "web_app": "web_app",

            "figures": "figures",
            "RELAB_spectra": path.join("RELAB", "data"),
            "backup": "backup"
            }
_subdirs["RELAB"] = path.join(_subdirs["datasets"], "RELAB")  # sub-subdir

# other directories
_path_modules = path.join(_project_dir, _subdirs["modules"])  # path to models
_path_data = path.join(_project_dir, _subdirs["datasets"])  # path to datasets
_path_model = path.join(_project_dir, _subdirs["models"])  # path to models
_path_hp_tuning = path.join(_project_dir, _subdirs["HP_tuning"])  # path to HP tuning results
_path_accuracy_tests = path.join(_project_dir, _subdirs["accuracy_tests"])  # path to accuracy tests

_path_figures = path.join(_project_dir, _subdirs["figures"])  # path to figures
_path_asteroid_images = path.join(_project_dir, _subdirs["asteroid_images"])  # path to background images

_path_relab_spectra = path.join(_project_dir, _subdirs["RELAB_spectra"])  # path to RELAB spectra
_path_catalogues = path.join(_project_dir, _subdirs["RELAB"])  # path to sample and spectral catalogues
_relab_web_page = "http://www.planetary.brown.edu/relabdata/data/"  # RELAB database (this URL does not work anymore)

_path_web_app = path.join(_project_dir, _subdirs["web_app"])  # path to web application
_path_backup = path.join(_project_dir, _subdirs["backup"])  # path to back-up directory

# names of the files in *.npz
# if you change any of these, you need to re-save your data, e.g., using change_files_in_npz in modules.utilities.py
_spectra_name = "spectra"
_wavelengths_name = "wavelengths"
_metadata_name = "metadata"
_metadata_key_name = "metadata_key"
_label_name = "labels"
_label_key_name = "labels_key"

# additional names of the files in *.npz used in accuracy tests
# if you change any of these, you need to re-save your data, e.g., using change_files_in_npz in modules.utilities.py
_label_true_name = "labels_true"
_label_pred_name = "labels_predicted"
_config_name = "config"

# additional names used in saving asteroid spectra
_coordinates_name = "coordinates"

# numerical eps
_num_eps = 1e-5  # num_eps of float32 is 1e-7

_wp = np.float32  # working precision -> set by keras config
_model_suffix = "h5"  # suffix of saved models

_rnd_seed = 42  # to control reproducibility; can be int or None (None for "automatic" random seed)

if _model_suffix not in ["SavedModel", "h5"]:
    print('Unknown "_model_suffix". Set to "SavedModel".')
    _model_suffix = "SavedModel"

# verbose of the code
_show_result_plot = True  # True for showing and saving results plots, False for not
_show_control_plot = True  # True for showing and saving control plots, False for not
_verbose = 2  # Set value for verbose: 0 = no print, 1 = full print, 2 = simple print
_quiet = _verbose == 0

# separators
_sep_in = "-"  # separates units inside one logical structure
_sep_out = "_"  # separates logical structures
