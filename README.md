# Asteroid-spectra
This project is related to (REF) paper about modal and chemical composition of silicate-rich rocks, including meteorites and asteroids.

Python neural-network code is based on keras library and can be spit into 3 parts:
1) NN_data.py reads data from ./Datasets/ and prepare them for training.
2) NN_train.py takes the loaded data and train the NN model
  a) current models losses/metrices are stored in NN_models.py, and NN_losses_metrics_activations.py. You should specify the metrices here
  b) in NN_train.py there are also methods for tuning HP (based on Tuner and Talos)
3) NN_evaluate.py inputs the names of trained models and data for evaluation (either data of name of a file in ./Datasets/)

The bahaviour of the whole code is driven by NN_config.py. You should not change the part at the bottom of this file unless you know the onsequences for the rest of the code.

The whole code can be run via main.py.

In ./Modules/chemical/ and ./Datasets/ there are trained modes and datasets which were used in the paper.

For any question or comments, contant me at david.korda@helsinki.fi.
