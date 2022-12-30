from typing import Callable, Literal


def gimme_hyperparameters(for_tuning: bool = False) -> Callable:
    # kern_size, dropout_hidden_hidden, and input_activation are uniform for all layers;
    # modify similarly to num_nodes if needed

    if for_tuning:
        return tuning
    return usage


def usage(composition_or_taxonomy: Literal["composition", "taxonomy"],
          grid_option: Literal["ASPECT", "Itokawa", "Eros", "Didymos"] | None) -> dict[str, str | int | float
                                                                                            | bool | list[int]]:
    if "composition" in composition_or_taxonomy:
        if grid_option is None:
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 2,  # Number of hidden layers
                "num_nodes": [24, 8],  # Number of nodes in hidden layers / number of filters for CNN
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "dropout_input_hidden": 0.0,  # Dropout
                "dropout_hidden_hidden": 0.3,  # Dropout
                "dropout_hidden_output": 0.4,  # Dropout
                "L1_trade_off": 0.005,  # L1 trade-off parameter
                "L2_trade_off": 0.00001,  # L2 trade-off parameter
                "max_norm": 4,  # max L2 norm of the weights for each layer
                "input_activation": "relu",  # Activation function of input layer
                "output_activation": "sigmoid",  # Activation function of output layer (relu, softmax, sigmoid)
                # Optimizer for the network
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                "learning_rate": 0.0005,  # Learning rate
                "batch_size": 8,  # Bath size
                "bs_norm_before_activation": False,
                "alpha": 0.1,  # Trade off between modal and chemical misfits (modal + alpha x chemical)
                "num_epochs": 2000,  # Number of epochs
                "model_usage": "compositional"  # type of model
            }

        elif "ASPECT" in grid_option:  # not determined
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 2,  # Number of hidden layers
                "num_nodes": [24, 8],  # Number of nodes in hidden layers / number of filters for CNN
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "dropout_input_hidden": 0.0,  # Dropout
                "dropout_hidden_hidden": 0.3,  # Dropout
                "dropout_hidden_output": 0.4,  # Dropout
                "L1_trade_off": 0.005,  # L1 trade-off parameter
                "L2_trade_off": 0.00001,  # L2 trade-off parameter
                "max_norm": 4,  # max L2 norm of the weights for each layer
                "input_activation": "relu",  # Activation function of input layer
                "output_activation": "sigmoid",  # Activation function of output layer (relu, softmax, sigmoid)
                # Optimizer for the network
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                "learning_rate": 0.0005,  # Learning rate
                "batch_size": 8,  # Bath size
                "bs_norm_before_activation": False,
                "alpha": 0.1,  # Trade off between modal and chemical misfits (modal + alpha x chemical)
                "num_epochs": 2000,  # Number of epochs
                "model_usage": "compositional"  # type of model
            }

        elif "Itokawa" in grid_option:
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 2,  # Number of hidden layers
                "num_nodes": [24, 8],  # Number of nodes in hidden layers / number of filters for CNN
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "dropout_input_hidden": 0.0,  # Dropout
                "dropout_hidden_hidden": 0.3,  # Dropout
                "dropout_hidden_output": 0.4,  # Dropout
                "L1_trade_off": 0.005,  # L1 trade-off parameter
                "L2_trade_off": 0.00001,  # L2 trade-off parameter
                "max_norm": 4,  # max L2 norm of the weights for each layer
                "input_activation": "relu",  # Activation function of input layer
                "output_activation": "sigmoid",  # Activation function of output layer (relu, softmax, sigmoid)
                # Optimizer for the network
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                "learning_rate": 0.0005,  # Learning rate
                "batch_size": 8,  # Bath size
                "bs_norm_before_activation": False,
                "alpha": 0.1,  # Trade off between modal and chemical misfits (modal + alpha x chemical)
                "num_epochs": 2000,  # Number of epochs
                "model_usage": "compositional"  # type of model
            }

        elif "Eros" in grid_option:
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 2,  # Number of hidden layers
                "num_nodes": [24, 8],  # Number of nodes in hidden layers / number of filters for CNN
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "dropout_input_hidden": 0.0,  # Dropout
                "dropout_hidden_hidden": 0.3,  # Dropout
                "dropout_hidden_output": 0.4,  # Dropout
                "L1_trade_off": 0.005,  # L1 trade-off parameter
                "L2_trade_off": 0.00001,  # L2 trade-off parameter
                "max_norm": 4,  # max L2 norm of the weights for each layer
                "input_activation": "relu",  # Activation function of input layer
                "output_activation": "sigmoid",  # Activation function of output layer (relu, softmax, sigmoid)
                # Optimizer for the network
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                "learning_rate": 0.0005,  # Learning rate
                "batch_size": 8,  # Bath size
                "bs_norm_before_activation": False,
                "alpha": 0.1,  # Trade off between modal and chemical misfits (modal + alpha x chemical)
                "num_epochs": 2000,  # Number of epochs
                "model_usage": "compositional"  # type of model
            }

        elif "Didymos" in grid_option:  # was not done yet
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 2,  # Number of hidden layers
                "num_nodes": [24, 8],  # Number of nodes in hidden layers / number of filters for CNN
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "dropout_input_hidden": 0.0,  # Dropout
                "dropout_hidden_hidden": 0.3,  # Dropout
                "dropout_hidden_output": 0.4,  # Dropout
                "L1_trade_off": 0.005,  # L1 trade-off parameter
                "L2_trade_off": 0.00001,  # L2 trade-off parameter
                "max_norm": 4,  # max L2 norm of the weights for each layer
                "input_activation": "relu",  # Activation function of input layer
                "output_activation": "sigmoid",  # Activation function of output layer (relu, softmax, sigmoid)
                # Optimizer for the network
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                "learning_rate": 0.0005,  # Learning rate
                "batch_size": 8,  # Bath size
                "bs_norm_before_activation": False,
                "alpha": 0.1,  # Trade off between modal and chemical misfits (modal + alpha x chemical)
                "num_epochs": 2000,  # Number of epochs
                "model_usage": "compositional"  # type of model
            }

        else:
            raise ValueError('Unknown "grid_option". See "NN_HP.py" for possible choices.')

    elif "taxonomy" in composition_or_taxonomy:
        if grid_option is None:
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 1,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
                "num_nodes": [24],  # Number of nodes in hidden layers / number of filters for CNN
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "dropout_input_hidden": 0.0,  # Dropout
                "dropout_hidden_hidden": 0.0,  # Dropout
                "dropout_hidden_output": 0.3,  # Dropout
                "L1_trade_off": 0.1,  # L1 trade-off parameter
                "L2_trade_off": 0.1,  # L2 trade-off parameter
                "max_norm": 2,  # max L2 norm of the weights for each layer
                "input_activation": "elu",  # Activation function of input layer
                "output_activation": "softmax",  # Activation function of output layer
                # Optimizer for the network
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models_taxonomy.py for options
                "learning_rate": 0.0032,  # Learning rate
                "batch_size": 144,  # Bath size
                "bs_norm_before_activation": False,
                "num_epochs": 1500,  # Number of epochs
                "model_usage": "taxonomical"  # type of model
            }

        elif "ASPECT" in grid_option:  # not determined
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 1,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
                "num_nodes": [24],  # Number of nodes in hidden layers / number of filters for CNN
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "dropout_input_hidden": 0.0,  # Dropout
                "dropout_hidden_hidden": 0.0,  # Dropout
                "dropout_hidden_output": 0.3,  # Dropout
                "L1_trade_off": 0.1,  # L1 trade-off parameter
                "L2_trade_off": 0.1,  # L2 trade-off parameter
                "max_norm": 2,  # max L2 norm of the weights for each layer
                "input_activation": "elu",  # Activation function of input layer
                "output_activation": "softmax",  # Activation function of output layer
                # Optimizer for the network
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models_taxonomy.py for options
                "learning_rate": 0.0032,  # Learning rate
                "batch_size": 144,  # Bath size
                "bs_norm_before_activation": False,
                "num_epochs": 1500,  # Number of epochs
                "model_usage": "taxonomical"  # type of model
            }

        elif "Itokawa" in grid_option:
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 1,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
                "num_nodes": [24],  # Number of nodes in hidden layers / number of filters for CNN
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "dropout_input_hidden": 0.05,  # Dropout
                "dropout_hidden_hidden": 0.0,  # Dropout
                "dropout_hidden_output": 0.3,  # Dropout
                "L1_trade_off": 0.01,  # L1 trade-off parameter
                "L2_trade_off": 0.01,  # L2 trade-off parameter
                "max_norm": 2,  # max L2 norm of the weights for each layer
                "input_activation": "elu",  # Activation function of input layer
                "output_activation": "softmax",  # Activation function of output layer
                # Optimizer for the network
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models_taxonomy.py for options
                "learning_rate": 0.0013,  # Learning rate
                "batch_size": 56,  # Bath size
                "bs_norm_before_activation": False,
                "num_epochs": 1500,  # Number of epochs
                "model_usage": "taxonomical"  # type of model
            }

        elif "Eros" in grid_option:
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 1,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
                "num_nodes": [24],  # Number of nodes in hidden layers / number of filters for CNN
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "dropout_input_hidden": 0.1,  # Dropout
                "dropout_hidden_hidden": 0.0,  # Dropout
                "dropout_hidden_output": 0.3,  # Dropout
                "L1_trade_off": 0.01,  # L1 trade-off parameter
                "L2_trade_off": 0.01,  # L2 trade-off parameter
                "max_norm": 2,  # max L2 norm of the weights for each layer
                "input_activation": "elu",  # Activation function of input layer
                "output_activation": "softmax",  # Activation function of output layer
                # Optimizer for the network
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models_taxonomy.py for options
                "learning_rate": 0.001,  # Learning rate
                "batch_size": 128,  # Bath size
                "bs_norm_before_activation": False,
                "num_epochs": 1500,  # Number of epochs
                "model_usage": "taxonomical"  # type of model
            }

        elif "Didymos" in grid_option:  # was not done yet
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 1,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
                "num_nodes": [24],  # Number of nodes in hidden layers / number of filters for CNN
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "dropout_input_hidden": 0.0,  # Dropout
                "dropout_hidden_hidden": 0.0,  # Dropout
                "dropout_hidden_output": 0.3,  # Dropout
                "L1_trade_off": 0.1,  # L1 trade-off parameter
                "L2_trade_off": 0.1,  # L2 trade-off parameter
                "max_norm": 2,  # max L2 norm of the weights for each layer
                "input_activation": "elu",  # Activation function of input layer
                "output_activation": "softmax",  # Activation function of output layer
                # Optimizer for the network
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models_taxonomy.py for options
                "learning_rate": 0.0032,  # Learning rate
                "batch_size": 144,  # Bath size
                "bs_norm_before_activation": False,
                "num_epochs": 1500,  # Number of epochs
                "model_usage": "taxonomical"  # type of model
            }

        else:
            raise ValueError('Unknown "grid_option". See "NN_HP.py" for possible choices.')
    else:
        raise ValueError('"composition_or_taxonomy" must contain "composition" or "taxonomy".')

    return p

def tuning() -> dict:
    p = {
        # kern_size, dropout_hidden_hidden, and input_activation are uniform for all layers;
        # modify similarly to n_nodes if needed

        "model_type": ["CNN", "MLP"],  # Convolutional (CNN) or dense (MLP)
        "num_layers": [1, 3],  # Number of hidden layers
        "num_nodes": [4, 32],  # Number of units/filters in the hidden layers
        "kern_size": [3, 7],  # Width of the kernel (CNN only)
        "dropout_input_hidden": [0.0, 0.3],  # Dropout rate
        "dropout_hidden_hidden": [0.0, 0.5],  # Dropout rate
        "dropout_hidden_output": [0.0, 0.5],  # Dropout rate
        "L1_trade_off": [0.00001, 1.0],  # L1 trade-off parameter
        "L2_trade_off": [0.00001, 1.0],  # L2 trade-off parameter
        "max_norm": [1, 5],  # max L2 norm of the weights for each layer
        "input_activation": ["relu", "tanh", "sigmoid", "elu"],  # Activation function of input and hidden layers
        "output_activation": ["sigmoid", "softmax"],  # Activation function of output layer
        # Optimizer for the network
        "optimizer": ["Adam", "SGD"],  # see return_optimizer and MyHyperModel.build in NN_models.py for options
        "learning_rate": [0.0001, 1.0],  # Learning rate
        "batch_size": [4, 512],  # Bath size
        "bs_norm_before_activation": [True, False],
        "alpha": [0.01, 10.0],  # Trade off between modal and chemical misfits
        "num_epochs": 500,  # Number of epochs
        "tuning_method": "Bayes",  # "Bayes", "Random"
        "plot_corr_mat": True,
    }

    return p
