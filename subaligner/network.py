import os
import tensorflow as tf

from tensorflow.keras.layers import (
    Dense,
    Input,
    LSTM,
    Conv1D,
    Dropout,
    Activation,
    BatchNormalization,
    Bidirectional,
    Flatten,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.utils import plot_model
from tensorflow.python import debug as tf_debug
from .utils import Utils
Utils.suppress_lib_logs()


# Not thread safe since the session of keras_backend is global
class Network(object):
    """ Network factory creates immutable DNNs. Only factory methods are allowed
    when generating DNN objects.
    """

    __secret = object()
    __LSTM = "lstm"
    __BI_LSTM = "bi_lstm"
    __CONV_1D = "conv_1d"
    __UNKNOWN = "unknown"

    def __init__(
        self,
        secret,
        input_shape,
        n_type,
        front_layers,
        relu_layers,
        dropout,
        model_path=None,
    ):
        """ Network object initialiser used by factory methods.

        Arguments:
            secret {object} -- A hash only known by factory methods.
            input_shape {tuple} -- A shape tuple (integers), not including the batch size.
            n_type {string} -- Can be "lstm" or "conv_1d".

        Raises:
            NotImplementedError -- Thrown when any network attributes are modified.
        """

        assert (
            secret == Network.__secret
        ), "Only factory methods are supported when creating instances"

        # Set the number of inter/intra threads to the number of physical cores (experiment shows this is the best)
        self.__config = tf.compat.v1.ConfigProto(
            inter_op_parallelism_threads=4,
            intra_op_parallelism_threads=4,
            allow_soft_placement=False,
            # log_device_placement=True,
        )
        if not hasattr("self", "__graph"):
            self.__graph = tf.compat.v1.Graph()
        if not hasattr("self", "__session"):
            self.__session = tf.compat.v1.Session(config=self.__config, graph=self.__graph)

        # for tb_debug:
        # tf.compat.v1.keras.backend.set_session(
        #     tf_debug.TensorBoardDebugWrapperSession(
        #         self.__session, "localhost:6064"
        #     )
        # )

        # for cli_debug:
        # tf.compat.v1.keras.backend.set_session(
        #     tf_debug.LocalCLIDebugWrapperSession(session)
        # )

        # non-debug mode
        tf.compat.v1.keras.backend.set_session(self.__session)

        if n_type == Network.__LSTM:
            self.__input_shape = input_shape
            self.__model = self.__lstm(
                input_shape, front_layers, relu_layers, dropout
            )
        if n_type == Network.__BI_LSTM:
            self.__input_shape = input_shape
            self.__model = self.__lstm(
                input_shape, front_layers, relu_layers, dropout, is_bidirectional=True
            )
        if n_type == Network.__CONV_1D:
            self.__input_shape = input_shape
            self.__model = self.__conv1d(
                input_shape, front_layers, relu_layers, dropout
            )
        if (input_shape is None and n_type == Network.__UNKNOWN and model_path is not None):
            self.__model = load_model(model_path)
            self.__input_shape = self.__model.input_shape[1:]
        self.__n_type = n_type

        # freeze the object after creation
        def __setattr__(self, *args):
            raise NotImplementedError("Cannot modify the immutable object")

        def __delattr__(self, *args):
            raise NotImplementedError("Cannot modify the immutable object")

    @classmethod
    def get_lstm(
        cls, input_shape, front_layers=[64], relu_layers=[32, 16], dropout=0.2
    ):
        """Factory method for creating a LSTM network.

        Arguments:
            input_shape {tuple} -- A shape tuple (integers), not including the batch size.

        Returns:
            Network -- Constructed LSTM network object.
        """

        return cls(
            cls.__secret,
            input_shape,
            Network.__LSTM,
            front_layers,
            relu_layers,
            dropout,
        )

    @classmethod
    def get_bi_lstm(
            cls, input_shape, front_layers=[64], relu_layers=[32, 16], dropout=0.2
    ):
        """Factory method for creating a bidirectional LSTM network.

        Arguments:
            input_shape {tuple} -- A shape tuple (integers), not including the batch size.

        Returns:
            Network -- Constructed Bidirectional LSTM network object.
        """

        return cls(
            cls.__secret,
            input_shape,
            Network.__BI_LSTM,
            front_layers,
            relu_layers,
            dropout,
        )

    @classmethod
    def get_conv1d(
        cls, input_shape, front_layers=[12], relu_layers=[56, 28], dropout=0.2
    ):
        """Factory method for creating a 1D convolutional network.

        Arguments:
            input_shape {tuple} -- A shape tuple (integers), not including the batch size.

        Returns:
            Network -- Constructed 1D convolutional network object.
        """
        return cls(
            cls.__secret,
            input_shape,
            Network.__CONV_1D,
            front_layers,
            relu_layers,
            dropout,
        )

    @classmethod
    def get_from_model(cls, model_path):
        """Load model into a network object.

        Arguments:
            model_path {string} -- The path to the model file.
        """

        return cls(
            cls.__secret,
            None,
            Network.__UNKNOWN,
            None,
            None,
            None,
            model_path=model_path,
        )

    @classmethod
    def save_model_and_weights(
        cls, model_filepath, weights_filepath, combined_filepath
    ):
        """Combine model and weights and save to a file

        Arguments:
            model_filepath {string} -- The path to the model file.
            weights_filepath {string} -- The path to the weights file.
        """

        model = load_model(model_filepath)
        model.load_weights(weights_filepath)
        model.save(combined_filepath)

    @staticmethod
    def load_model_and_weights(model_filepath, weights_filepath):
        """Load weights to the Network model.

        Arguments:
            model_filepath {string} -- The model file path.
            weights_filepath {string} -- The weights file path.

        Returns:
            Network -- Reconstructed network object.
        """
        network = Network.get_from_model(model_filepath)
        network.__model.load_weights(weights_filepath)
        return network

    @property
    def input_shape(self):
        """Get the input shape of the network.

        Returns:
            tuple -- The input shape of the network.
        """

        return self.__input_shape

    @property
    def n_type(self):
        """Get the type of the network.

        Returns:
            string -- The type of the network.
        """

        return self.__n_type

    @property
    def summary(self):
        """Get the summary of the network.

        Returns:
            string -- The summary of the network.
        """

        return self.__model.summary()

    @property
    def layers(self):
        """Get the layers of the network.

        Returns:
            list -- The statck of layers contained by the network
        """

        return self.__model.layers

    def get_predictions(self, input_data, weights_filepath, verbose=1):
        """Get a Numpy array of predictions.

        Arguments:
            input_data {numpy.ndarray} -- The input data, as a Numpy array.
            weights_filepath {string} -- The weights file path.
            verbose {int} -- The verbosity mode of logging, either 0 (succinct) or 1 (verbose).

        Returns:
            numpy.ndarray -- The Numpy array of predictions.
        """
        self.__model.load_weights(weights_filepath)
        return self.__model.predict(input_data, verbose=verbose)

    def fit_and_get_history(
        self,
        train_data,
        labels,
        model_filepath,
        weights_filepath,
        logs_dir,
        epochs,
        training_log,
        resume,
    ):
        """Fit the training data to the network and save the network model as a HDF file.

        Arguments:
            train_data {numpy.array} -- The Numpy array of training data.
            labels {numpy.array} -- The Numpy array of training labels.
            model_filepath {string} -- The model file path.
            weights_filepath {string} -- The weights file path.
            logs_dir {string} -- The TensorBoard log file directory.
            epochs {int} -- The number of training epochs.
            training_log {string} -- The path to the log file of epoch results.
            resume {bool} -- True to continue with previous training result or False to start a new one (default: {False}).
        Returns:
            tuple -- A tuple contains validation losses and validation accuracies.
        """

        csv_logger = (
            CSVLogger(training_log)
            if not resume
            else CSVLogger(training_log, append=True)
        )
        checkpoint = ModelCheckpoint(
            filepath=weights_filepath,
            monitor="val_loss",
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
        )
        tensorboard = TensorBoard(
            log_dir=logs_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=True,
        )
        # earlyStopping = EarlyStopping(monitor="val_loss", min_delta=0.00001, verbose=1, mode="min", patience=20)
        callbacks_list = [
            checkpoint,
            tensorboard,
            csv_logger,
        ]  # , earlyStopping]
        if not resume:
            self.__model.compile(
                loss="binary_crossentropy",
                optimizer=Adam(lr=0.001),
                metrics=["accuracy"],
            )
        initial_epoch = 0
        if resume:
            assert os.path.isfile(training_log), "{} does not exist and is required by training resumption".format(
                training_log)
            training_log_file = open(training_log)
            initial_epoch += sum(1 for _ in training_log_file) - 1
            training_log_file.close()
            assert epochs > initial_epoch, "Existing model has been trained for {} epochs".format(initial_epoch)
        hist = self.__model.fit(
            train_data,
            labels,
            epochs=epochs,
            batch_size=32,
            shuffle=True,
            validation_split=0.25,
            verbose=1,
            callbacks=callbacks_list,
            initial_epoch=initial_epoch,
        )
        save_model(self.__model, model_filepath)

        return hist.history["val_loss"], hist.history["val_acc"] if int(tf.__version__.split(".")[0]) < 2 else hist.history["val_accuracy"]

    def fit_with_generator(
        self,
        data_label_generator,
        steps_per_epoch,
        model_filepath,
        weights_filepath,
        logs_dir,
        epochs,
        training_log,
        resume,
        tb_debug=False,
        cli_debug=False,
    ):
        """Fit the training data to the network.

        Arguments:
            train_data {numpy.array} -- The Numpy array of training data.
            labels {numpy.array} -- The Numpy array of training labels.
            model_filepath {string} -- The model file path.
            weights_filepath {string} -- The weights file path.
            logs_dir {string} -- The TensorBoard log file directory.
            epochs {int} -- The number of training epochs.
            training_log {string} -- The path to the log file of epoch results.
            resume {bool} -- True to continue with previous training result or False to start a new one (default: {False}).
            tb_debug {bool} -- True to turn on the Tensorboard debug mode.
            cli_debug {bool} -- True to turn on the CLI debug mode.

        Returns:
            tuple -- A tuple contains validation losses and validation accuracies.
        """

        csv_logger = CSVLogger(training_log)
        checkpoint = ModelCheckpoint(
            filepath=weights_filepath,
            monitor="val_loss",
            verbose=1,
            save_best_only=False,
            save_weights_only=True,
        )
        tensorboard = TensorBoard(
            log_dir=logs_dir,
            histogram_freq=0,
            write_graph=True,
            write_images=True,
        )
        # earlyStopping = EarlyStopping(monitor="val_loss", min_delta=0.00001, verbose=1, mode="min", patience=20)
        callbacks_list = [
            checkpoint,
            tensorboard,
            csv_logger,
        ]  # , earlyStopping]
        if not resume:
            self.__model.compile(
                loss="binary_crossentropy",
                optimizer=Adam(lr=0.001),
                metrics=["accuracy"],
            )
        initial_epoch = 0
        if resume:
            assert os.path.isfile(training_log), "{} does not exist and is required by training resumption".format(
                training_log)
            training_log_file = open(training_log)
            initial_epoch += sum(1 for _ in training_log_file) - 1
            training_log_file.close()
            assert epochs > initial_epoch, "Existing model has been trained for {} epochs".format(initial_epoch)
        hist = self.__model.fit_generator(
            data_label_generator,
            epochs=epochs,
            shuffle=True,
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks_list,
            initial_epoch=initial_epoch,
        )
        self.__model.save(model_filepath)

        return hist.history["loss"], hist.history["acc"] if int(tf.__version__.split(".")[0]) < 2 else hist.history["accuracy"]

    # To make this work, need to change model._network_nodes to model._container_nodes
    # in tensorflow/python/keras/_impl/keras/utils/vis_utils.py
    def plot_model(self, file_path):
        """Plot the network architecture in the dot format.

        Arguments:
            file_path {string} -- The path of the saved image.
        """

        plot_model(self.__model, to_file=file_path, show_shapes=True)

    # Clear Keras backend session due to https://github.com/tensorflow/tensorflow/issues/3388
    @staticmethod
    def clear_session():
        tf.compat.v1.keras.backend.clear_session()

    @staticmethod
    def __lstm(input_shape, front_layers, relu_layers, dropout, is_bidirectional=False):
        inputs = Input(shape=input_shape)
        hidden = BatchNormalization()(inputs)

        for nodes in front_layers:
            hidden = Bidirectional(LSTM(nodes))(hidden) if is_bidirectional else LSTM(nodes)(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = Activation("relu")(hidden)
            hidden = Dropout(dropout)(hidden)

        for nodes in relu_layers:
            hidden = Dense(nodes)(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = Activation("relu")(hidden)
            hidden = Dropout(dropout)(hidden)

        hidden = Dense(1)(hidden)
        outputs = Activation("sigmoid")(hidden)

        return Model(inputs, outputs)

    @staticmethod
    def __conv1d(input_shape, front_layers, relu_layers, dropout):
        inputs = Input(shape=input_shape)
        hidden = BatchNormalization()(inputs)

        for nodes in front_layers:
            hidden = Conv1D(filters=nodes, kernel_size=3, activation="relu")(
                hidden
            )

        for nodes in relu_layers:
            hidden = Dense(nodes)(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = Activation("relu")(hidden)
            hidden = Dropout(dropout)(hidden)

        hidden = Dense(1)(hidden)
        outputs = Activation("sigmoid")(hidden)

        return Model(inputs, outputs)
