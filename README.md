Python neural network code is based on the Keras library and can be split into 3 parts:
- NN_data.py reads data from ./Datasets/ and prepares them for training.
- NN_train.py takes the loaded data and trains the NN model
  - current models and losses and metrics are stored in NN_models.py, and NN_losses_metrics_activations.py. You should specify the metrics here
  - in NN_train.py there are also methods for tuning HP (based on Tuner and Talos)
- NN_evaluate.py inputs the names of trained models and data for evaluation (either data or name of a file in ./Datasets/)

The behaviour of the whole code is driven by NN_config.py. You should not change the part at the bottom of this file unless you know the consequences for the rest of the code.

The whole code can be run via main*.py.

Trained models for determining mineralogical properties can be found in ./Models/. See README.md in the folder for details about the models.

Used dataset and their metadata are stored in ./Datasets/. Mineral and elemental analytical data used as input during training, validation, and testing are in **combined-denoised-norm.npz**. Taxonomy data are in **asteroid_spectra-reduced-denoised-norm.npz**. The relevant metadata and detailed mineral and chemical analyses used to derive training input analytical data are stored in table **Sample_Catalogue.xlsx**. See the corresponding README.md for more details.

For any questions or comments, contact me at david.korda@helsinki.fi.
