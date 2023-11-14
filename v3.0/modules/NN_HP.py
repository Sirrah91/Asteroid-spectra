from typing import Callable, Literal
from warnings import warn
from modules._constants import _sep_in


def gimme_hyperparameters(for_tuning: bool = False) -> Callable:
    # kern_size, dropout_hidden_hidden, and input_activation are uniform for all layers;
    # modify similarly to num_nodes if needed

    if for_tuning:
        return tuning
    return usage


def usage(composition_or_taxonomy: Literal["composition", "taxonomy"],
          grid_option: str) -> dict[str, str | int | float | bool | list[int]]:
    if "composition" in composition_or_taxonomy:

        if f"450{_sep_in}2450{_sep_in}5" in grid_option:
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 2,  # Number of hidden layers
                "num_nodes": [24, 8],  # Number of units/filters in the hidden layers
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "kern_pad": "same",  # Kernel padding (CNN only)
                "input_activation": "relu",  # Activation function of the input and hidden layers
                "output_activation": "sigmoid",  # Activation function of the output layer (relu, softmax, sigmoid)
                "dropout_input_hidden": 0.0,  # Dropout rate
                "dropout_hidden_hidden": 0.3,  # Dropout rate
                "dropout_hidden_output": 0.4,  # Dropout rate
                "L1_trade_off": 0.005,  # L1 trade-off parameter
                "L2_trade_off": 0.00001,  # L2 trade-off parameter
                "max_norm": 4.,  # max L2 norm of the weights for each layer
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                "learning_rate": 0.0005,  # Learning rate
                "batch_size": 8,  # Bath size
                "bs_norm_before_activation": False,
                "alpha": 0.1,  # Trade-off between modal and chemical misfits (modal + alpha x chemical)
                "num_epochs": 2000,  # Number of epochs
                "model_usage": "composition"  # type of model
            }

        elif f"820{_sep_in}2080{_sep_in}20" in grid_option:
            p = usage(composition_or_taxonomy="composition", grid_option=f"450{_sep_in}2450{_sep_in}5")

        elif f"820{_sep_in}2360{_sep_in}20" in grid_option:
            p = usage(composition_or_taxonomy="composition", grid_option=f"450{_sep_in}2450{_sep_in}5")

        elif f"650{_sep_in}1600{_sep_in}25" in grid_option:
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 2,  # Number of hidden layers
                "num_nodes": [24, 8],  # Number of units/filters in the hidden layers
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "kern_pad": "same",  # Kernel padding (CNN only)
                "input_activation": "elu",  # Activation function of the input and hidden layers
                "output_activation": "sigmoid",  # Activation function of the output layer (relu, softmax, sigmoid)
                "dropout_input_hidden": 0.05,  # Dropout rate
                "dropout_hidden_hidden": 0.1,  # Dropout rate
                "dropout_hidden_output": 0.2,  # Dropout rate
                "L1_trade_off": 0.001,  # L1 trade-off parameter
                "L2_trade_off": 0.0,  # L2 trade-off parameter
                "max_norm": 4.,  # max L2 norm of the weights for each layer
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                "learning_rate": 0.0005,  # Learning rate
                "batch_size": 64,  # Bath size
                "bs_norm_before_activation": True,
                "alpha": 0.1,  # Trade-off between modal and chemical misfits (modal + alpha x chemical)
                "num_epochs": 2000,  # Number of epochs
                "model_usage": "composition"  # type of model
            }

        elif f"750{_sep_in}1350{_sep_in}25" in grid_option:
            p = usage(composition_or_taxonomy="composition", grid_option=f"650{_sep_in}1600{_sep_in}25")

        elif f"750{_sep_in}1550{_sep_in}20" in grid_option:
            p = usage(composition_or_taxonomy="composition", grid_option=f"650{_sep_in}1600{_sep_in}25")

        elif "ASPECT" in grid_option:
            p = usage(composition_or_taxonomy="composition", grid_option=f"650{_sep_in}1600{_sep_in}25")

        elif "HS-H" in grid_option:
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 1,  # Number of hidden layers
                "num_nodes": [12],  # Number of units/filters in the hidden layers
                "kern_size": 3,  # Width of the kernel (only if CNN)
                "kern_pad": "same",  # Kernel padding (CNN only)
                "input_activation": "relu",  # Activation function of the input and hidden layers
                "output_activation": "softmax",  # Activation function of the output layer (relu, softmax, sigmoid)
                "dropout_input_hidden": 0.0,  # Dropout rate
                "dropout_hidden_hidden": 0.0,  # Dropout rate
                "dropout_hidden_output": 0.4,  # Dropout rate
                "L1_trade_off": 0.001,  # L1 trade-off parameter
                "L2_trade_off": 0.0006,  # L2 trade-off parameter
                "max_norm": 4.,  # max L2 norm of the weights for each layer
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                "learning_rate": 0.0022,  # Learning rate
                "batch_size": 64,  # Bath size
                "bs_norm_before_activation": True,
                "alpha": 0.02,  # Trade-off between modal and chemical misfits (modal + alpha x chemical)
                "num_epochs": 2000,  # Number of epochs
                "model_usage": "composition"  # type of model
            }

        else:
            warn('Unknown "grid_option". Using default settings (see "NN_HP.py")')
            p = usage(composition_or_taxonomy="composition", grid_option=f"450{_sep_in}2450{_sep_in}5")


    elif "taxonomy" in composition_or_taxonomy:

        if f"450{_sep_in}2450{_sep_in}5" in grid_option:
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 1,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
                "num_nodes": [24],  # Number of units/filters in the hidden layers
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "kern_pad": "same",  # Kernel padding (CNN only)
                "input_activation": "elu",  # Activation function of the input and hidden layers
                "output_activation": "softmax",  # Activation function of the output layer
                "dropout_input_hidden": 0.0,  # Dropout rate
                "dropout_hidden_hidden": 0.0,  # Dropout rate
                "dropout_hidden_output": 0.3,  # Dropout rate
                "L1_trade_off": 0.1,  # L1 trade-off parameter
                "L2_trade_off": 0.1,  # L2 trade-off parameter
                "max_norm": 4.,  # max L2 norm of the weights for each layer
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                "learning_rate": 0.0032,  # Learning rate
                "batch_size": 144,  # Bath size
                "bs_norm_before_activation": False,
                "use_weights": True,  # Use weighted-class loss function or not
                "num_epochs": 1500,  # Number of epochs
                "model_usage": "taxonomy"  # type of model
            }

        elif f"820{_sep_in}2080{_sep_in}20" in grid_option:
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 1,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
                "num_nodes": [24],  # Number of units/filters in the hidden layers
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "kern_pad": "same",  # Kernel padding (CNN only)
                "input_activation": "elu",  # Activation function of the input and hidden layers
                "output_activation": "softmax",  # Activation function of the output layer
                "dropout_input_hidden": 0.05,  # Dropout rate
                "dropout_hidden_hidden": 0.0,  # Dropout rate
                "dropout_hidden_output": 0.3,  # Dropout rate
                "L1_trade_off": 0.01,  # L1 trade-off parameter
                "L2_trade_off": 0.01,  # L2 trade-off parameter
                "max_norm": 4.,  # max L2 norm of the weights for each layer
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                "learning_rate": 0.0013,  # Learning rate
                "batch_size": 56,  # Bath size
                "bs_norm_before_activation": False,
                "use_weights": True,  # Use weighted-class loss function or not
                "num_epochs": 1500,  # Number of epochs
                "model_usage": "taxonomy"  # type of model
            }

        elif f"820{_sep_in}2360{_sep_in}20" in grid_option:
            p = usage(composition_or_taxonomy="taxonomy", grid_option=f"820{_sep_in}2080{_sep_in}20")

        elif f"500{_sep_in}900{_sep_in}10" in grid_option:
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 1,  # Number of hidden layers (CNN, FC);  scalar for CNN or FC
                "num_nodes": [12],  # Number of units/filters in the hidden layers
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "kern_pad": "same",  # Kernel padding (CNN only)
                "input_activation": "elu",  # Activation function of the input and hidden layers
                "output_activation": "softmax",  # Activation function of the output layer
                "dropout_input_hidden": 0.0,  # Dropout rate
                "dropout_hidden_hidden": 0.0,  # Dropout rate
                "dropout_hidden_output": 0.3,  # Dropout rate
                "L1_trade_off": 0.01,  # L1 trade-off parameter
                "L2_trade_off": 0.04,  # L2 trade-off parameter
                "max_norm": 4.,  # max L2 norm of the weights for each layer
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                "learning_rate": 0.005,  # Learning rate
                "batch_size": 64,  # Bath size
                "bs_norm_before_activation": True,
                "use_weights": True,  # Use weighted-class loss function or not
                "num_epochs": 1500,  # Number of epochs
                "model_usage": "taxonomy"  # type of model
            }

        elif f"670{_sep_in}950{_sep_in}10" in grid_option:
            p = usage(composition_or_taxonomy="taxonomy", grid_option=f"500{_sep_in}900{_sep_in}10")

        elif f"650{_sep_in}1600{_sep_in}25" in grid_option:
            p = {
                "model_type": "CNN",  # Convolutional (CNN) or dense (MLP)
                "num_layers": 1,  # Number of hidden layers
                "num_nodes": [24],  # Number of units/filters in the hidden layers
                "kern_size": 5,  # Width of the kernel (only if CNN)
                "kern_pad": "same",  # Kernel padding (CNN only)
                "input_activation": "tanh",  # Activation function of the input and hidden layers
                "output_activation": "softmax",  # Activation function of the output layer
                "dropout_input_hidden": 0.15,  # Dropout rate
                "dropout_hidden_hidden": 0.0,  # Dropout rate
                "dropout_hidden_output": 0.2,  # Dropout rate
                "L1_trade_off": 0.0,  # L1 trade-off parameter
                "L2_trade_off": 0.002,  # L2 trade-off parameter
                "max_norm": 4.,  # max L2 norm of the weights for each layer
                "optimizer": "Adam",  # see return_optimizer and MyHyperModel.build in NN_models.py for options
                "learning_rate": 0.01,  # Learning rate
                "batch_size": 64,  # Bath size
                "bs_norm_before_activation": True,
                "use_weights": True,  # Use weighted-class loss function or not
                "num_epochs": 1500,  # Number of epochs
                "model_usage": "taxonomy"  # type of model
            }

        elif "ASPECT" in grid_option and "swir" not in grid_option:
            p = usage(composition_or_taxonomy="taxonomy", grid_option=f"650{_sep_in}1600{_sep_in}25")

        elif "ASPECT" in grid_option and "swir" in grid_option:
            p = usage(composition_or_taxonomy="taxonomy", grid_option=f"820{_sep_in}2080{_sep_in}20")

        elif "HS-H" in grid_option:
            p = usage(composition_or_taxonomy="taxonomy", grid_option=f"500{_sep_in}900{_sep_in}10")

        else:
            warn('Unknown "grid_option". Using default settings (see "NN_HP.py")')
            p = usage(composition_or_taxonomy="taxonomy", grid_option=f"450{_sep_in}2450{_sep_in}5")

    else:
        raise ValueError('"composition_or_taxonomy" must contain "composition" or "taxonomy".')

    return p

def tuning(model_usage: Literal["composition", "taxonomy"]) -> dict:
    p = {
        # kern_size, dropout_hidden_hidden, and input_activation are uniform for all layers;
        # modify similarly to num_nodes if needed

        "model_type": ["CNN", "MLP"],  # Convolutional (CNN) or dense (MLP)
        "num_layers": [1, 3],  # Number of hidden layers
        "num_nodes": [4, 32],  # Number of units/filters in the hidden layers
        "kern_size": [3, 7],  # Width of the kernel (CNN only)
        "kern_pad": ["same"],  # Kernel padding (CNN only)
        "input_activation": ["relu", "tanh", "sigmoid", "elu"],  # Activation function of the input and hidden layers
        "output_activation": ["sigmoid", "softmax"],  # Activation function of the output layer
        "dropout_input_hidden": [0.0, 0.3],  # Dropout rate
        "dropout_hidden_hidden": [0.0, 0.5],  # Dropout rate
        "dropout_hidden_output": [0.0, 0.5],  # Dropout rate
        "L1_trade_off": [0., 1.0],  # L1 trade-off parameter
        "L2_trade_off": [0., 1.0],  # L2 trade-off parameter
        "max_norm": [1., 5.],  # max L2 norm of the weights for each layer
        "optimizer": ["Adam", "SGD"],  # see return_optimizer and MyHyperModel.build in NN_models.py for options
        "learning_rate": [0.0001, 1.0],  # Learning rate
        "batch_size": [4, 512],  # Bath size
        "bs_norm_before_activation": [True, False],
        # IF YOU USE VAL_LOSS AS A MONITORING QUANTITY, YOU SHOULD NOT USE ALPHA AND USE_WEIGHTS IN HP TUNING
        "alpha": [0.1],  # Trade-off between modal and chemical misfits; composition model only
        "use_weights": [True, False],  # Use weighted-class loss function or not; taxonomy model only
        "num_epochs": 500,  # Number of epochs
        "model_usage": model_usage,  # "composition" or "taxonomy"
        "tuning_method": "Random",  # "Bayes", "Random"
        "plot_corr_mat": True,
    }

    return p
