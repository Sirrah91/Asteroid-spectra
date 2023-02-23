from keras.layers import Input, Dense, Conv1D
from keras.layers import Flatten, BatchNormalization, Dropout, Activation
from keras.models import Model, Functional
from keras import regularizers
from keras_tuner import HyperModel
import tensorflow.keras.optimizers as opt
from tensorflow.keras.constraints import MaxNorm

from modules.NN_config_taxonomy import metrics, loss, num_labels

# objective, direction = "".join(("val_", metrics[0].__name__)), "max"  # minimise or maximise the objective?
objective, direction = "val_loss", "min"  # minimise or maximise the objective?


def return_optimizer(params: dict[str, str]):
    if params["optimizer"] == "Adam":
        optimizer = opt.Adam(learning_rate=params["learning_rate"])
    elif params["optimizer"] == "SGD":
        optimizer = opt.SGD(learning_rate=params["learning_rate"], nesterov=True)
    else:
        raise ValueError("Unknown optimizer. Add it to return_optimizer in NN_models_taxonomy.py")
        # add other optimisers if needed

    return optimizer


def MLP_model(input_shape: tuple[int, ...], params: dict[str, str | int | float | bool | list[int]]) -> Functional:
    # Input layer
    inputs = Input(shape=input_shape)
    bs_norm = BatchNormalization()(inputs)

    # Dropout between input and the first hidden
    drop = Dropout(params["dropout_input_hidden"])(bs_norm)

    # Adding the hidden layers
    for i in range(params["num_layers"]):
        dense = Dense(params["num_nodes"][i],
                      kernel_constraint=MaxNorm(params["max_norm"]),
                      bias_constraint=MaxNorm(params["max_norm"]),
                      kernel_regularizer=regularizers.l1_l2(l1=params["L1_trade_off"], l2=params["L2_trade_off"]),
                      bias_regularizer=regularizers.l1_l2(l1=params["L1_trade_off"], l2=params["L2_trade_off"])
                      )(drop)

        if params["bs_norm_before_activation"]:
            x = BatchNormalization()(dense)
            x = Activation(params["input_activation"])(x)
        else:
            x = Activation(params["input_activation"])(dense)
            x = BatchNormalization()(x)

        # Dropout layer for stabilisation of the network
        if i < params["num_layers"] - 1:  # Last layer has different dropout
            drop = Dropout(params["dropout_hidden_hidden"])(x)

    drop = Dropout(params["dropout_hidden_output"])(x)

    # Number of nodes in output layer
    dense = Dense(num_labels)(drop)
    outputs = Activation(params["output_activation"])(dense)

    model = Model(inputs=inputs, outputs=outputs)

    # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
    optimizer = return_optimizer(params)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def CNN_model(input_shape: tuple[int, ...], params: dict[str, str | int | float | bool | list[int]]) -> Functional:
    # Input layer
    inputs = Input(shape=input_shape)
    bs_norm = BatchNormalization()(inputs)

    # Dropout between input and the first hidden
    drop = Dropout(params["dropout_input_hidden"])(bs_norm)

    # Adding the CNN hidden layers
    for i in range(params["num_layers"]):
        conv = Conv1D(filters=params["num_nodes"][i], kernel_size=params["kern_size"], padding="same",
                      kernel_constraint=MaxNorm(params["max_norm"]),
                      bias_constraint=MaxNorm(params["max_norm"]),
                      kernel_regularizer=regularizers.l1_l2(l1=params["L1_trade_off"], l2=params["L2_trade_off"]),
                      bias_regularizer=regularizers.l1_l2(l1=params["L1_trade_off"], l2=params["L2_trade_off"])
                      )(drop)

        if params["bs_norm_before_activation"]:
            x = BatchNormalization()(conv)
            x = Activation(params["input_activation"])(x)
        else:
            x = Activation(params["input_activation"])(conv)
            x = BatchNormalization()(x)

        # Dropout layer for stabilisation of the network
        if i < params["num_layers"] - 1:  # Last layer has different dropout
            drop = Dropout(params["dropout_hidden_hidden"])(x)

    flat = Flatten()(x)
    drop = Dropout(params["dropout_hidden_output"])(flat)

    # Number of nodes in output layer
    dense = Dense(num_labels)(drop)
    outputs = Activation(params["output_activation"])(dense)

    model = Model(inputs=inputs, outputs=outputs)

    # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
    optimizer = return_optimizer(params)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


class MyHyperModel(HyperModel):
    def __init__(self, input_shape_MLP: tuple[int, ...], params: dict[str, str | int | float | bool | list[int]]):
        self.input_shape_MLP = input_shape_MLP
        self.input_shape_CNN = input_shape_MLP + (1,)
        self.params = params

    def common_hp(self, hp):
        self.activation_input = hp.Choice("input_activation", values=self.params["input_activation"])
        self.activation_output = hp.Choice("output_activation", values=self.params["output_activation"])

        self.opt_choice = hp.Choice("optimizer", values=self.params["optimizer"])
        self.lr = hp.Float("learning_rate", min_value=min(self.params["learning_rate"]),
                           max_value=max(self.params["learning_rate"]), sampling="log")

        self.drop_in_hid = hp.Float("dropout_input_hidden", min_value=min(self.params["dropout_input_hidden"]),
                                    max_value=max(self.params["dropout_input_hidden"]), step=0.05)
        self.drop_hid_out = hp.Float("dropout_hidden_output", min_value=min(self.params["dropout_hidden_output"]),
                                     max_value=max(self.params["dropout_hidden_output"]), step=0.1)

        self.l1 = hp.Float("L1_trade_off", min_value=min(self.params["L1_trade_off"]),
                           max_value=max(self.params["L1_trade_off"]), sampling="log")
        self.l2 = hp.Float("L2_trade_off", min_value=min(self.params["L2_trade_off"]),
                           max_value=max(self.params["L2_trade_off"]), sampling="log")
        self.max_norm = hp.Float("max_norm", min_value=min(self.params["max_norm"]),
                                 max_value=max(self.params["max_norm"]))

        self.batch_size = hp.Int("batch_size", min_value=min(self.params["batch_size"]),
                                 max_value=max(self.params["batch_size"]), step=4)
        self.bs_norm_before_activation = hp.Choice("batch_norm_before_activation",
                                                   values=self.params["bs_norm_before_activation"])

    def mlp_hp(self, hp):
        self.num_layers = hp.Int("num_layers", min_value=min(self.params["num_layers"]),
                                 max_value=max(self.params["num_layers"]), step=1)

        if self.num_layers > 1:
            with hp.conditional_scope("num_layers", list(range(2, max(self.params["num_layers"]) + 1))):
                self.drop_hid_hid = hp.Float("dropout_hidden_hidden",
                                             min_value=min(self.params["dropout_hidden_hidden"]),
                                             max_value=max(self.params["dropout_hidden_hidden"]), step=0.1)

        self.units = {}
        for i in range(self.num_layers):
            if i < self.num_layers:
                with hp.conditional_scope("num_layers", list(range(i + 1, max(self.params["num_layers"]) + 1))):
                    self.units[f"num_units_{i}"] = hp.Int(f"num_units_{i}", min_value=min(self.params["num_nodes"]),
                                                          max_value=max(self.params["num_nodes"]), step=4)

    def cnn_hp(self, hp):
        self.num_layers = hp.Int("num_layers", min_value=min(self.params["num_layers"]),
                                 max_value=max(self.params["num_layers"]), step=1)

        if self.num_layers > 1:
            with hp.conditional_scope("num_layers", list(range(2, max(self.params["num_layers"]) + 1))):
                self.drop_hid_hid = hp.Float("dropout_hidden_hidden",
                                             min_value=min(self.params["dropout_hidden_hidden"]),
                                             max_value=max(self.params["dropout_hidden_hidden"]), step=0.1)

        self.kernel_size = hp.Int("kernel_size", min_value=min(self.params["kern_size"]),
                                  max_value=max(self.params["kern_size"]), step=2)

        self.filters = {}
        for i in range(self.num_layers):
            if i < self.num_layers:
                with hp.conditional_scope("num_layers", list(range(i + 1, max(self.params["num_layers"]) + 1))):
                    self.filters[f"num_filters_{i}"] = hp.Int(f"num_filters_{i}",
                                                              min_value=min(self.params["num_nodes"]),
                                                              max_value=max(self.params["num_nodes"]), step=4)

    def return_optimizer(self, hp):
        if self.opt_choice == "Adam":
            with hp.conditional_scope("optimizer", ["Adam"]):
                optimizer_tuner = opt.Adam(learning_rate=self.lr)
        elif self.opt_choice == "SGD":
            with hp.conditional_scope("optimizer", ["SGD"]):
                optimizer_tuner = opt.SGD(learning_rate=self.lr, nesterov=True)
        else:
            raise ValueError("Unknown optimizer. Add it to MyHyperModel.build in NN_models_taxonomy.py")
            # add other optimisers if needed

        return optimizer_tuner

    def build_MLP(self):
        # Input layer
        inputs = Input(shape=self.input_shape_MLP)
        bs_norm = BatchNormalization()(inputs)

        # Dropout between input and the first hidden layer
        drop = Dropout(rate=self.drop_in_hid)(bs_norm)

        # Adding hidden layers
        for i in range(self.num_layers):
            dense = Dense(
                units=self.units[f"num_units_{i}"],
                kernel_constraint=MaxNorm(self.max_norm),
                bias_constraint=MaxNorm(self.max_norm),
                kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
                bias_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2)
            )(drop)

            if self.bs_norm_before_activation:
                x = BatchNormalization()(dense)
                x = Activation(self.activation_input)(x)
            else:
                x = Activation(self.activation_input)(dense)
                x = BatchNormalization()(x)

            # Dropout layer for stabilisation of the network
            if i < self.num_layers - 1:  # Last layer has different dropout
                drop = Dropout(rate=self.drop_hid_hid)(x)

        drop = Dropout(rate=self.drop_hid_out)(x)

        # Number of nodes in output layer
        dense = Dense(num_labels)(drop)
        outputs = Activation(self.activation_output)(dense)

        uncompiled_model = Model(inputs=inputs, outputs=outputs)

        return uncompiled_model

    def build_CNN(self):
        # Input layer
        inputs = Input(shape=self.input_shape_CNN)
        bs_norm = BatchNormalization()(inputs)

        # Dropout between input and the first hidden layer
        drop = Dropout(rate=self.drop_in_hid)(bs_norm)

        # Adding hidden layers
        for i in range(self.num_layers):
            conv = Conv1D(
                filters=self.filters[f"num_filters_{i}"],
                kernel_size=self.kernel_size,
                padding="same",
                kernel_constraint=MaxNorm(self.max_norm),
                bias_constraint=MaxNorm(self.max_norm),
                kernel_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2),
                bias_regularizer=regularizers.l1_l2(l1=self.l1, l2=self.l2)
            )(drop)

            if self.bs_norm_before_activation:
                x = BatchNormalization()(conv)
                x = Activation(self.activation_input)(x)
            else:
                x = Activation(self.activation_input)(conv)
                x = BatchNormalization()(x)

            # Dropout layer for stabilisation of the network
            if i < self.num_layers - 1:  # Last layer has different dropout
                drop = Dropout(rate=self.drop_hid_hid)(x)

        flat = Flatten()(x)
        drop = Dropout(rate=self.drop_hid_out)(flat)

        # Number of nodes in output layer
        dense = Dense(num_labels)(drop)
        outputs = Activation(self.activation_output)(dense)

        uncompiled_model = Model(inputs=inputs, outputs=outputs)

        return uncompiled_model

    def build(self, hp):
        self.model_type = hp.Choice("model_type", values=self.params["model_type"])

        if self.model_type == "MLP":
            with hp.conditional_scope("model_type", ["MLP"]):
                self.common_hp(hp)
                self.mlp_hp(hp)
                model = self.build_MLP()
                optimizer_tuner = self.return_optimizer(hp)
        elif self.model_type == "CNN":
            with hp.conditional_scope("model_type", ["CNN"]):
                self.common_hp(hp)
                self.cnn_hp(hp)
                model = self.build_CNN()
                optimizer_tuner = self.return_optimizer(hp)

        # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
        model.compile(loss=loss, metrics=metrics, optimizer=optimizer_tuner)

        return model

    def fit(self, hp, model: Functional, *args, **kwargs):
        return model.fit(
            *args,
            # Tune batch size
            batch_size=self.batch_size,
            **kwargs,
        )
