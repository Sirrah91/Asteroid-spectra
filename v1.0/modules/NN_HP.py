from typing import Dict, Union


def gimme_hyperparameters(chem_or_class: str, grid_option: Union[str, None]) -> Dict:

    if "chem" in chem_or_class:
        if grid_option is None:
            p = {
                'model_type': 'CNN',  # Convolutional (CNN) or dense (MLP)
                'n_layers': 2,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
                'n_nodes': [24, 8, 4],  # Number of nodes in hidden layers / number of filters for CNN
                'kern_size': 5,  # Width of the kernel (only if CNN)
                'dropout_input_hidden': 0.0,  # Dropout
                'dropout_hidden_hidden': 0.3,  # Dropout
                'dropout_hidden_output': 0.4,  # Dropout
                'L1_trade_off': 0.005,  # L1 trade-off parameter
                'input_activation': 'relu',  # Activation function of input layer
                'output_activation': 'sigmoid',  # Activation function of output layer (relu, softmax, sigmoid)
                # Optimizer for the network
                'optimizer': 'Adam',  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                'learning_rate': 0.0005,  # Learning rate
                'alpha': 0.1,  # Trade off between modal and chemical misfits (modal + alpha x chemical)
                'batch_size': 8,  # Bath size
                'num_epochs': 5000,  # Number of epochs
                'tuning': 0  # This is parameters for evaluation; do not change
            }

        elif "ASPECT" in grid_option:
            pass

        elif "Itokawa" in grid_option:
            p = {
                'model_type': 'CNN',  # Convolutional (CNN) or dense (MLP)
                'n_layers': 2,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
                'n_nodes': [4, 4, 4],  # Number of nodes in hidden layers / number of filters for CNN
                'kern_size': 5,  # Width of the kernel (only if CNN)
                'dropout_input_hidden': 0.0,  # Dropout
                'dropout_hidden_hidden': 0.15,  # Dropout
                'dropout_hidden_output': 0.2,  # Dropout
                'L1_trade_off': 0.0001,  # L1 trade-off parameter
                'input_activation': 'sigmoid',  # Activation function of input layer
                'output_activation': 'sigmoid',  # Activation function of output layer (relu, softmax, sigmoid)
                # Optimizer for the network
                'optimizer': 'Adam',  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                'learning_rate': 0.003,  # Learning rate
                'alpha': 0.01,  # Trade off between modal and chemical misfits (modal + alpha x chemical)
                'batch_size': 48,  # Bath size
                'num_epochs': 5000,  # Number of epochs
                'tuning': 0  # This is parameters for evaluation; do not change
            }

        elif "Eros" in grid_option:
            p = {
                'model_type': 'CNN',  # Convolutional (CNN) or dense (MLP)
                'n_layers': 3,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
                'n_nodes': [8, 4, 4],  # Number of nodes in hidden layers / number of filters for CNN
                'kern_size': 5,  # Width of the kernel (only if CNN)
                'dropout_input_hidden': 0.0,  # Dropout
                'dropout_hidden_hidden': 0.1,  # Dropout
                'dropout_hidden_output': 0.0,  # Dropout
                'L1_trade_off': 0.0003,  # L1 trade-off parameter
                'input_activation': 'sigmoid',  # Activation function of input layer
                'output_activation': 'sigmoid',  # Activation function of output layer (relu, softmax, sigmoid)
                # Optimizer for the network
                'optimizer': 'Adam',  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                'learning_rate': 0.05,  # Learning rate
                'alpha': 0.3,  # Trade off between modal and chemical misfits (modal + alpha x chemical)
                'batch_size': 128,  # Bath size
                'num_epochs': 5000,  # Number of epochs
                'tuning': 0  # This is parameters for evaluation; do not change
            }

    elif "class" in chem_or_class:
        if grid_option is None:
            p = {
                'model_type': 'CNN',  # Convolutional (CNN) or dense (MLP)
                'n_layers': 1,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
                'n_nodes': [32, 8, 4],  # Number of nodes in hidden layers / number of filters for CNN
                'kern_size': 3,  # Width of the kernel (only if CNN)
                'dropout_input_hidden': 0.0,  # Dropout
                'dropout_hidden_hidden': 0.0,  # Dropout
                'dropout_hidden_output': 0.2,  # Dropout
                'L1_trade_off': 0.08,  # L1 trade-off parameter
                'input_activation': 'sigmoid',  # Activation function of input layer
                'output_activation': 'softmax',  # Activation function of output layer
                # Optimizer for the network
                'optimizer': 'Adam',  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                'learning_rate': 0.007,  # Learning rate
                'batch_size': 97,  # Bath size
                'num_epochs': 2000,  # Number of epochs
                'tuning': 0  # This is parameters for evaluation; do not change
            }

        elif "Itokawa" in grid_option:
            p = {
                'model_type': 'CNN',  # Convolutional (CNN) or dense (MLP)
                'n_layers': 1,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
                'n_nodes': [32, 8, 4],  # Number of nodes in hidden layers / number of filters for CNN
                'kern_size': 5,  # Width of the kernel (only if CNN)
                'dropout_input_hidden': 0.0,  # Dropout
                'dropout_hidden_hidden': 0.0,  # Dropout
                'dropout_hidden_output': 0.4,  # Dropout
                'L1_trade_off': 0.0001,  # L1 trade-off parameter
                'input_activation': 'sigmoid',  # Activation function of input layer
                'output_activation': 'softmax',  # Activation function of output layer
                # Optimizer for the network
                'optimizer': 'Adam',  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                'learning_rate': 0.0015,  # Learning rate
                'batch_size': 128,  # Bath size
                'num_epochs': 2000,  # Number of epochs
                'tuning': 0  # This is parameters for evaluation; do not change
            }

        elif "Eros" in grid_option:
            p = {
                'model_type': 'CNN',  # Convolutional (CNN) or dense (MLP)
                'n_layers': 1,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
                'n_nodes': [32, 8, 4],  # Number of nodes in hidden layers / number of filters for CNN
                'kern_size': 3,  # Width of the kernel (only if CNN)
                'dropout_input_hidden': 0.0,  # Dropout
                'dropout_hidden_hidden': 0.2,  # Dropout
                'dropout_hidden_output': 0.1,  # Dropout
                'L1_trade_off': 0.1,  # L1 trade-off parameter
                'input_activation': 'sigmoid',  # Activation function of input layer
                'output_activation': 'softmax',  # Activation function of output layer
                # Optimizer for the network
                'optimizer': 'Adam',  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                'learning_rate': 0.01,  # Learning rate
                'batch_size': 128,  # Bath size
                'num_epochs': 2000,  # Number of epochs
                'tuning': 0  # This is parameters for evaluation; do not change
            }
    else:
        raise ValueError('"chem_or_class" must contain "chem" or "class".')

    return p
