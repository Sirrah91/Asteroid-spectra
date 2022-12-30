from keras.metrics import categorical_accuracy
from keras.losses import categorical_crossentropy
from modules.NN_HP import gimme_hyperparameters

loss = categorical_crossentropy  # loss function (must be a function, not string)
metrics = [categorical_accuracy]  # metrics (must be a function, not string)

# re-interpolate input data to different resolution (see reinterpolate_data in utilities_spectra.py)
# None for no change; other possibilities are "ASPECT", "ASPECT_swir" "Itokawa", "Eros", "Didymos"
interpolate_to = None
new_wvl_grid, new_wvl_grid_normalisation = None, None  # None for no change; "interpolate_to" must be None to use this

val_portion = 0.2  # Set the fraction of data for validation
test_portion = 0.2  # Set the fraction of data for tests

# Hyper parameters
p = gimme_hyperparameters(for_tuning=False)(composition_or_taxonomy="taxonomy", grid_option=interpolate_to)

# definition of classes (comment unwanted classes)
# some classes were merged or delete if you use "-reduced" taxonomy
list_of_classes = [
    "A",
    "B",
    "C",
    # "Cb",
    # "Cg",
    "Cgh",
    "Ch",
    "D",
    "K",
    "L",
    # "O",
    "Q",
    # "Qw",
    # "R",
    "S",
    # "Sa",
    # "Sq",
    # "Sq:",
    # "Sqw",
    "Sr",
    # "Srw",
    # "Sv",
    # "Svw",
    # "Sw",
    "T",
    # "U",
    "V",
    # "Vw",
    "X",
    # "Xc",
    "Xe",
    "Xk",
    # "Xn",
]

classes = {cls: i for i, cls in enumerate(list_of_classes)}
classes2 = {value: key for key, value in classes.items()}

num_labels = len(classes)  # Select the number of classifications

trim_mean_cut = 0.2  # parameter of trim_mean in evaluation

model_subdir = "taxonomical"  # directory where to save models
# model_subdir = "accuracy_test"  # directory where to save models

# model_name equal to time_stamp + _ + model_type + _ + suffix + .h5
model_grid = interpolate_to if interpolate_to is not None else "full"
model_name_suffix = "".join((p["model_type"], "_", model_grid))

show_result_plot = True  # True for showing and saving of results plots, False for not
show_control_plot = True  # True for showing control plots, False for not
verb = 2  # Set value for verbose: 0 = no print, 1 = full print, 2 = simple print

if model_subdir == "accuracy_test":
    val_portion = 0.0  # Set the fraction of data for validation
    show_result_plot = False  # True for showing and saving of results plots, False for not
    show_control_plot = False  # True for showing and saving of control plots, False for not
    verb = 0  # Set value for verbose: 0 = no print, 1 = full print, 2 = simple print
