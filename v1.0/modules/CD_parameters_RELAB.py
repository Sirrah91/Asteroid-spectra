# Parameters for the data collection

# Only spectra within this range will be processes
lambda_min = 350  # Minimum value of lambda; nm
lambda_max = 2550  # Maximum value of lambda; nm
resolution_max = 15  # Maximum value of resolution; nm
denoise = True  # Denoise the spectrum? (convolution with the given kernel)
normalise = True  # Normalise the spectrum?

resolution_final = 5  # Interpolated resolution; nm
normalised_at = 550  # nm

project_dir = '/home/dakorda/Python/NN/'  # Directory which contains Datasets, Modules, etc.
path_relab = "".join((project_dir, '/Datasets/RELAB/'))  # Path to RELAB folder
path_tuomas = "".join((project_dir, '/Datasets/Tuomas/'))  # Path to Tuomas folder
path_MGM = '/home/local/dakorda/MGM/david_mgm/input/'  # Path to MGM folder
web_page = 'http://www.planetary.brown.edu/relabdata/data/'

# olivine, orthopyroxene, clinopyroxene, plagioclase
subtypes = 2, 3, 3, 3
use_minerals = 1, 1, 1, 1

num_minerals = sum(use_minerals)
num_labels = sum([subtypes[i] for i in range(len(subtypes)) if use_minerals[i]]) + num_minerals  # One for each number

show_histogram = False
