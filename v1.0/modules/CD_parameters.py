# Parameters for the data collection

# Only spectra within this range will be processes
lambda_min = 450  # Minimum value of lambda; nm
lambda_max = 2450  # Maximum value of lambda; nm
resolution_max = 15  # Maximum acceptable step in  wavelength resolution; nm

denoise = True  # Denoise the spectrum? (convolution with the given kernel)
normalise = True  # Normalise the spectrum?

resolution_final = 5  # Interpolated resolution; nm
normalised_at = 550  # nm

project_dir = '/home/dakorda/Python/NN/'  # Directory which contains Datasets, Modules, etc.
path_relab = "".join((project_dir, '/Datasets/RELAB/'))  # Path to RELAB folder
path_relab_raw = "".join((project_dir, '/RELAB/'))  # Path to RELAB folder
path_taxonomy = "".join((project_dir, '/Datasets/taxonomy/'))  # Path to Tuomas folder
path_ctape = "".join((project_dir, '/Datasets/C-Tape/'))  # Path to Tuomas folder
path_MGM = '/home/local/dakorda/MGM/david_mgm/input/'  # Path to MGM folder
web_page = 'http://www.planetary.brown.edu/relabdata/data/'

# olivine, orthopyroxene, clinopyroxene, plagioclase
subtypes_CD = 2, 3, 3, 3
use_minerals_CD = 1, 1, 1, 1

num_minerals_CD = sum(use_minerals_CD)
# One for each number
num_labels_CD = sum([subtypes_CD[i] for i in range(len(subtypes_CD)) if use_minerals_CD[i]]) + num_minerals_CD

