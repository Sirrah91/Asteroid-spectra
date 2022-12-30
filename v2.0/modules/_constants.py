# THIS FILE CONTAINS FILES COMMON TO ALL FUNCTIONS (PATHS, CONSTANTS, ETC.)

# Base directory of the project
_project_dir = "/home/dakorda/Python/NN/"  # there are still some issues with relative path, use the full path

# other directories
_path_data = "".join((_project_dir, "/Datasets/"))  # path to datasets
_path_model = "".join((_project_dir, "/Models/"))  # path to models
_path_hp_tuning = "".join((_project_dir, "/tuning_HP/"))  # path to HP tuning results
_path_accuracy_test = "".join((_project_dir, "/accuracy_test/"))  # path to accuracy tests

_path_figures = "".join((_project_dir, "/figures/"))  # path to figures
_path_asteroid_images = "".join((_project_dir, "/Asteroid_images/"))  # path to background images

_path_relab_spectra = "".join((_project_dir, "/RELAB/data/"))  # path to RELAB spectra
_path_catalogues = "".join((_path_data, "/RELAB/"))  # path to sample and spectral catalogues
_relab_web_page = "http://www.planetary.brown.edu/relabdata/data/"  # RELAB database (this URL does not work anymore)

_path_backup = "".join((_project_dir, "/backup/"))  # path to back-up directory

_path_MGM = "/home/dakorda/MGM/david_mgm/input/"  # path to MGM

# numerical eps
_num_eps = 1e-5
