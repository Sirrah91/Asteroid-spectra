from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
import tensorflow.keras.optimizers as opt
from tensorflow.keras.constraints import MaxNorm
from keras_tuner import HyperModel
import numpy as np

from modules.utilities import to_list
from modules.NN_losses_metrics_activations import (gimme_metrics, create_custom_objects, gimme_composition_loss,
                                                   gimme_taxonomy_loss)
from modules.NN_config_parse import gimme_endmember_counts, gimme_num_minerals, gimme_num_labels
from modules.NN_config_parse import bin_to_used, bin_to_cls

# defaults only
from modules.NN_config_composition import comp_model_setup, comp_output_setup
from modules.NN_config_taxonomy import tax_model_setup, tax_output_setup


class MyHyperModel(HyperModel):
    # hp = keras_tuner.HyperParameters()

    def __init__(self, input_shape_MLP: tuple[int, ...],
                 params: dict[str, str | int | float | bool | list[int]],
                 metrics: list[str] | None = None,
                 bin_code: str | None = None,
                 ):
        self._input_shape_MLP = input_shape_MLP
        self._input_shape_CNN = input_shape_MLP + (1,)
        self._params = params
        self._metrics = metrics

        self._for_tuning = "tuning_method" in params.keys()
        self._model_usage = params["model_usage"]

        if self._model_usage == "taxonomy":
            if self._metrics is None: self._metrics = tax_model_setup["metrics"]
            if bin_code is None: bin_code = tax_output_setup["bin_code"]

            self._used_minerals, self._used_endmembers = None, None
            self._num_labels = len(bin_to_cls(bin_code=bin_code))

        else:
            if self._metrics is None: self._metrics = comp_model_setup["metrics"]
            if bin_code is None: bin_code = comp_output_setup["bin_code"]

            self._used_minerals, self._used_endmembers = bin_to_used(bin_code=bin_code)
            self._num_labels = gimme_num_labels(self._used_minerals, self._used_endmembers)

    def _common_hp(self, hp):
        self._activation_input = hp.Choice("input_activation", values=to_list(self._params["input_activation"]))
        self._activation_output = hp.Choice("output_activation", values=to_list(self._params["output_activation"]))

        self._opt_choice = hp.Choice("optimizer", values=to_list(self._params["optimizer"]))

        # add learning rate to conditional_scope for optimizer?
        self._lr = hp.Float("learning_rate", min_value=np.min(self._params["learning_rate"]),
                            max_value=np.max(self._params["learning_rate"]), sampling="log")

        self._drop_in_hid = hp.Float("dropout_input_hidden", min_value=np.min(self._params["dropout_input_hidden"]),
                                     max_value=np.max(self._params["dropout_input_hidden"]), step=0.05)
        self._drop_hid_out = hp.Float("dropout_hidden_output", min_value=np.min(self._params["dropout_hidden_output"]),
                                      max_value=np.max(self._params["dropout_hidden_output"]), step=0.1)

        if self._for_tuning:
            L1_trade_off = np.clip(self._params["L1_trade_off"], a_min=K.epsilon(), a_max=None)
            L2_trade_off = np.clip(self._params["L2_trade_off"], a_min=K.epsilon(), a_max=None)
            sampling = "log"
        else:
            L1_trade_off = self._params["L1_trade_off"]
            L2_trade_off = self._params["L2_trade_off"]
            sampling = "linear"  # Allow for zero min_value

        self._l1 = hp.Float("L1_trade_off", min_value=np.min(L1_trade_off), max_value=np.max(L1_trade_off), sampling=sampling)
        self._l2 = hp.Float("L2_trade_off", min_value=np.min(L2_trade_off), max_value=np.max(L2_trade_off), sampling=sampling)

        self._max_norm = hp.Float("max_norm", min_value=np.min(self._params["max_norm"]),
                                  max_value=np.max(self._params["max_norm"]))

        self._batch_size = hp.Int("batch_size", min_value=np.min(self._params["batch_size"]),
                                  max_value=np.max(self._params["batch_size"]), step=4)
        self._bs_norm_before_activation = hp.Choice("batch_norm_before_activation",
                                                    values=to_list(self._params["bs_norm_before_activation"]))

    def _mlp_hp(self, hp):
        self._num_layers = hp.Int("num_layers", min_value=np.min(self._params["num_layers"]),
                                  max_value=np.max(self._params["num_layers"]), step=1)

        if self._num_layers > 1:
            with hp.conditional_scope("num_layers", list(range(2, np.max(self._params["num_layers"]) + 1))):
                self._drop_hid_hid = hp.Float("dropout_hidden_hidden",
                                              min_value=np.min(self._params["dropout_hidden_hidden"]),
                                              max_value=np.max(self._params["dropout_hidden_hidden"]), step=0.1)

        self._units = {}
        for i in range(self._num_layers):
            if i < self._num_layers:
                with hp.conditional_scope("num_layers", list(range(i + 1, np.max(self._params["num_layers"]) + 1))):
                    default = None if self._for_tuning else self._params["num_nodes"][i]
                    self._units[f"num_units_{i}"] = hp.Int(f"num_units_{i}",
                                                           default=default,
                                                           min_value=np.min(self._params["num_nodes"]),
                                                           max_value=np.max(self._params["num_nodes"]), step=4)

    def _cnn_hp(self, hp):
        self._num_layers = hp.Int("num_layers", min_value=np.min(self._params["num_layers"]),
                                  max_value=np.max(self._params["num_layers"]), step=1)

        if self._num_layers > 1:
            with hp.conditional_scope("num_layers", list(range(2, np.max(self._params["num_layers"]) + 1))):
                self._drop_hid_hid = hp.Float("dropout_hidden_hidden",
                                              min_value=np.min(self._params["dropout_hidden_hidden"]),
                                              max_value=np.max(self._params["dropout_hidden_hidden"]), step=0.1)

        self._kernel_size = hp.Int("kernel_size", min_value=np.min(self._params["kern_size"]),
                                   max_value=np.max(self._params["kern_size"]), step=2)
        self._kernel_padding = hp.Choice("kernel_padding", values=to_list(self._params["kern_pad"]))

        self._filters = {}
        for i in range(self._num_layers):
            if i < self._num_layers:
                with hp.conditional_scope("num_layers", list(range(i + 1, np.max(self._params["num_layers"]) + 1))):
                    default = None if self._for_tuning else self._params["num_nodes"][i]
                    self._filters[f"num_filters_{i}"] = hp.Int(f"num_filters_{i}",
                                                               default=default,
                                                               min_value=np.min(self._params["num_nodes"]),
                                                               max_value=np.max(self._params["num_nodes"]), step=4)

    def _return_optimizer(self, hp):
        # Is the conditional_scope needed here?
        if self._opt_choice == "Adam":
            with hp.conditional_scope("optimizer", ["Adam"]):
                optimizer_tuner = opt.Adam(learning_rate=self._lr)
        elif self._opt_choice == "SGD":
            with hp.conditional_scope("optimizer", ["SGD"]):
                optimizer_tuner = opt.SGD(learning_rate=self._lr, nesterov=True)
        else:
            raise ValueError(f'Unknown optimizer. Must be one of "Adam" or "SGD" but is "{self._opt_choice}". '
                             f'Add it to MyHyperModel._return_optimizer in NN_models.py.')

        return optimizer_tuner

    def _return_loss_and_metrics(self, hp):
        metrics = gimme_metrics(metrics=self._metrics, used_minerals=self._used_minerals,
                                used_endmembers=self._used_endmembers, cleaning=True, all_to_one=True)

        if self._model_usage == "composition":

            if (gimme_num_minerals(self._used_minerals) > 0 and
                    np.sum(gimme_endmember_counts(self._used_endmembers)) > 0):

                if self._for_tuning:
                    alpha = np.clip(self._params["alpha"], a_min=K.epsilon(), a_max=None)
                    sampling = "log"
                else:
                    alpha = self._params["alpha"]
                    sampling = "linear"  # Allow for zero min_value

                self._alpha = hp.Float("alpha", min_value=np.min(alpha), max_value=np.max(alpha), sampling=sampling)

            else:  # If not true, the loss does not use alpha. Single-value choice to simplify HP tuning
                self._alpha = hp.Choice("alpha", values=[1.])

            loss = gimme_composition_loss(used_minerals=self._used_minerals,
                                          used_endmembers=self._used_endmembers,
                                          alpha=self._alpha)

        elif self._model_usage == "taxonomy":
            self._use_weights = hp.Choice("use_weights", values=to_list(self._params["use_weights"]))

            loss = gimme_taxonomy_loss(use_weights=self._use_weights)

        else:
            raise ValueError(
                f'Unknown model usage. Must be one of "composition" or "taxonomy" but is "{self._model_usage}".\n'
                f'Add it to MyHyperModel._return_loss_and_metrics and '
                f'MyHyperModel._return_output_activation, both in NN_models.py.')

        return loss, metrics

    def _return_output_activation(self):
        if self._model_usage == "composition":
            if self._activation_output in ["softmax", "sigmoid", "relu", "plu"]:
                custom_objects = create_custom_objects(used_minerals=self._used_minerals,
                                                       used_endmembers=self._used_endmembers)

                return custom_objects[f"{self._activation_output}_norm"]

            else:
                raise ValueError(f'Unknown output activation. Must be one of "softmax", "sigmoid", "relu", or "plu"'
                                 f' but is "{self._activation_output}".\n'
                                 f'Add it to MyHyperModel._return_output_activation in NN_models.py.')

        elif self._model_usage == "taxonomy":
            return self._activation_output

        else:
            raise ValueError(
                f'Unknown model usage. Must be one of "composition" or "taxonomy" but is "{self._model_usage}".\n'
                f'Add it to MyHyperModel._return_loss_and_metrics and '
                f'MyHyperModel._return_output_activation, both in NN_models.py.')

    def _build_MLP(self):
        # Input layer
        inputs = Input(shape=self._input_shape_MLP)
        bs_norm = BatchNormalization()(inputs)

        # Dropout between input and the first hidden layer
        drop = Dropout(rate=self._drop_in_hid)(bs_norm)

        # Adding hidden layers
        for i in range(self._num_layers):
            dense = Dense(
                units=self._units[f"num_units_{i}"],
                kernel_constraint=MaxNorm(self._max_norm),
                bias_constraint=MaxNorm(self._max_norm),
                kernel_regularizer=regularizers.l1_l2(l1=self._l1, l2=self._l2),
                bias_regularizer=regularizers.l1_l2(l1=self._l1, l2=self._l2)
            )(drop)

            if self._bs_norm_before_activation:
                x = BatchNormalization()(dense)
                x = Activation(self._activation_input)(x)
            else:
                x = Activation(self._activation_input)(dense)
                x = BatchNormalization()(x)

            # Dropout layer for stabilisation of the network
            if i < self._num_layers - 1:  # Last layer has different dropout
                drop = Dropout(rate=self._drop_hid_hid)(x)

        drop = Dropout(rate=self._drop_hid_out)(x)

        # Number of nodes in the output layer
        dense = Dense(self._num_labels)(drop)
        outputs = Activation(self._return_output_activation())(dense)
        uncompiled_model = Model(inputs=inputs, outputs=outputs)

        return uncompiled_model

    def _build_CNN(self):
        # Input layer
        inputs = Input(shape=self._input_shape_CNN)
        bs_norm = BatchNormalization()(inputs)

        # Dropout between input and the first hidden layer
        drop = Dropout(rate=self._drop_in_hid)(bs_norm)

        # Adding hidden layers
        for i in range(self._num_layers):
            conv = Conv1D(
                filters=self._filters[f"num_filters_{i}"],
                kernel_size=self._kernel_size,
                padding=self._kernel_padding,
                kernel_constraint=MaxNorm(self._max_norm),
                bias_constraint=MaxNorm(self._max_norm),
                kernel_regularizer=regularizers.l1_l2(l1=self._l1, l2=self._l2),
                bias_regularizer=regularizers.l1_l2(l1=self._l1, l2=self._l2)
            )(drop)

            if self._bs_norm_before_activation:
                x = BatchNormalization()(conv)
                x = Activation(self._activation_input)(x)
            else:
                x = Activation(self._activation_input)(conv)
                x = BatchNormalization()(x)

            # Dropout layer for stabilisation of the network
            if i < self._num_layers - 1:  # Last layer has different dropout
                drop = Dropout(rate=self._drop_hid_hid)(x)

        flat = Flatten()(x)
        drop = Dropout(rate=self._drop_hid_out)(flat)

        # Number of nodes in the output layer
        dense = Dense(self._num_labels)(drop)
        outputs = Activation(self._return_output_activation())(dense)

        uncompiled_model = Model(inputs=inputs, outputs=outputs)

        return uncompiled_model

    def build(self, hp):
        self._model_type = hp.Choice("model_type", values=to_list(self._params["model_type"]))

        if self._model_type == "MLP":
            with hp.conditional_scope("model_type", ["MLP"]):
                self._common_hp(hp)
                self._mlp_hp(hp)
                model = self._build_MLP()
                optimizer_tuner = self._return_optimizer(hp)
        elif self._model_type == "CNN":
            with hp.conditional_scope("model_type", ["CNN"]):
                self._common_hp(hp)
                self._cnn_hp(hp)
                model = self._build_CNN()
                optimizer_tuner = self._return_optimizer(hp)
        else:
            raise ValueError(f'Unknown model type. Must be one of "MLP" or "CNN" but is "{self._model_type}".\n'
                             f'Add it to MyHyperModel.build in NN_models.py.')

        # Compiling the model to train. Declare the loss function and the optimizer (SGD, Adam, etc.)
        loss, metrics = self._return_loss_and_metrics(hp)
        model.compile(loss=loss, metrics=metrics, optimizer=optimizer_tuner)

        return model

    def fit(self, hp, model: Model, *args, **kwargs):
        return model.fit(
            *args,
            # Tune batch size
            batch_size=self._batch_size,
            **kwargs,
        )
