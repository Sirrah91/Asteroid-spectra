# Parameters for the data collection
import numpy as np

# Only spectra within this range will be processed
lambda_min = 450.  # Minimum value of lambda; nm
lambda_max = 2450.  # Maximum value of lambda; nm
resolution_max = 15.  # Maximum acceptable step in wavelength resolution; nm

denoise = True  # Denoise the spectrum? (convolution with the given kernel)
normalise = True  # Normalise the spectrum?

resolution_final = 5.  # Interpolated resolution; nm
wvl_norm = 550.  # nm
denoising_sigma = 7.  # nm

# olivine, orthopyroxene, clinopyroxene, plagioclase
endmembers_CD = 2, 3, 3, 3
minerals_CD = 1, 1, 1, 1

# columns of metadata and labels in sample_catalogue.xlsx
usecols = "A:AG"

num_minerals_CD = int(np.sum(minerals_CD))
num_endmembers_CD = int(np.sum([endmember for mineral, endmember in zip(minerals_CD, endmembers_CD) if mineral]))
num_labels_CD = num_minerals_CD + num_endmembers_CD
