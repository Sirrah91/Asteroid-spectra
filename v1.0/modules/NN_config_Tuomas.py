# This file contains global parameters defining the neural network
from tensorflow.keras.optimizers import Adam

loss = 'categorical_crossentropy'  # loss function
metrics = ['categorical_accuracy']  # metrics

# This selects between Itokawa and Eros
model_name_suffix = 'Eros'  # model_type = date_code + _ + suffix + .h5

# Hyper parameters
p = {
    'model_type': 'FC',  # Convolutional (CNN) or fully connected (FC)
    'n_layers': 1,  # Number of hidden layers
    'n_nodes': [30],  # Number of nodes in hidden layers
    'kern_size': 17,  # Width of the kernel (only if CNN)
    'lambda1': 0.001,  # L1 trade-off parameter (only if FC)
    'dropout_input': 0.0,  # Dropout
    'dropout_hidden': 0.0,  # Dropout
    'learning_rate': 0.001,  # Learning rate
    'num_epochs': 2000,  # Number of epochs
    'batch_size': 64,  # Bath size
    'optimizer': Adam,  # Optimizer for the network (adam, adamax, nadam, adadelta, adagrad, rmsprop, sgd)
    'input_activation': 'relu',  # Activation function of input layer
    'output_activation': 'softmax',   # Activation function of output layer
    'tuning': 0  # This is parameters for evaluation; do not change
}

p_for_tuning = {  # In talos format
    #
    # [list, of, samples] or
    # tuple (start, stop, steps); no last element, i.e. (20, 60, 4) = np.linspace(20, 60, 5)[:-1]
    #
    'model_type': ['FC', 'CNN'],  # Convolutional (CNN) or fully connected (FC)
    'n_layers': [1, 2],  # Number of hidden layers
    'n_nodes': [[48, 32], [32, 16], [24, 16]],  # Number of nodes in hidden layers [first, second, ...]
    'kern_size': [17, 33],  # Width of the kernel (only if CNN)
    'lambda1': [0.0001, 0.001, 0.01],  # L1 trade-off parameter (only if FC)
    'dropout_input': [0.0, 0.1, 0.2],  # Dropout
    'dropout_hidden': [0.0, 0.2, 0.4],  # Dropout
    'learning_rate': [0.0001, 0.001, 0.01],  # Learning rate
    'num_epochs': [100],  # Number of epochs
    'batch_size': [16, 32, 64],  # Bath size
    'optimizer': [Adam],  # Optimizer for the network
    'input_activation': ['relu'],  # Activation function of input layer
    'output_activation': ['softmax'],   # Activation function of output layer
    'tuning': [1]  # This is parameters for evaluation; do not change
}

# definition of classes
classes = {'A': 0,
           'B': 1,
           'C': 2,
           'Cgh': 3,
           'Ch': 4,
           'D': 5,
           'K': 6,
           'L': 7,
           'Q': 8,
           'S': 9,
           'Sr': 10,
           'T': 11,
           'V': 12,
           'X': 13,
           'Xe': 14,
           'Xk': 15
           }

# This is useful mostly in control plots
classes2 = {value: key for key, value in classes.items()}

num_labels = len(classes)   # Select the number of classifications

val_portion = 0.2  # Set the percentage of data for validation
test_portion = 0  # Set the percentage of data for tests

trimmed = 0.2  # parameter of trim_mean

project_dir = '/home/dakorda/Python/NN/'  # Directory which contains Datasets, Modules, etc.

show_control_plot = True  # True for showing control plots, False for not
verb = 0  # Set value for verbose: 0=no print, 1=full print, 2=simple print