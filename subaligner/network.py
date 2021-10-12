import os
import math
import importlib
import psutil
import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as tf_optimizers

from typing import Tuple, Optional, Any, List, Generator
from tensorflow.keras.layers import (
    Dense,
    Input,
    LSTM,
    Conv1D,
    MaxPooling1D,
    Dropout,
    Activation,
    BatchNormalization,
    Bidirectional,
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger,
)
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras import backend as K
from .utils import Utils
from .logger import Logger
from .exception import TerminalException
from .hyperparameters import Hyperparameters
Utils.suppress_lib_logs()


class Network(object):
    """ Network factory creates DNNs.
    Not thread safe since the session of keras_backend is global.
    Only factory methods are allowed when generating DNN objects.
    """

    LSTM = "lstm"
    BI_LSTM = "bi_lstm"
    CONV_1D = "conv_1d"
    TYPES = [LSTM, BI_LSTM, CONV_1D]

    __secret = object()
    __UNKNOWN = "unknown"

    def __init__(
        self,
        secret: Optional[object],
        input_shape: Tuple,
        hyperparameters: Hyperparameters,
        model_path: Optional[str] = None,
        backend: str = "tensorflow"
    ) -> None:
        """ Network object initialiser used by factory methods.

        Arguments:
            secret {object} -- A hash only known by factory methods.
            input_shape {tuple} -- A shape tuple (integers), not including the batch size.
            hyperparameters {Hyperparameters} -- A configuration for hyperparameters used for training.
            model_path {string} -- The path to the model file.
            backend {string} -- The tensor manipulation backend (default: {tensorflow}). Only tensorflow is supported
                                by TF 2 and this parameter is here only for a historical reason.
        Raises:
            NotImplementedError -- Thrown when any network attributes are modified.
        """
        assert (
            secret == Network.__secret
        ), "Only factory methods are supported when creating instances"

        Network.__set_keras_backend(backend)

        if (hyperparameters.network_type == Network.__UNKNOWN and model_path is not None):
            self.__model = load_model(model_path)
            self.__input_shape = self.__model.input_shape[1:]
        elif hyperparameters.network_type == Network.LSTM:
            self.__input_shape = input_shape
            self.__model = self.__lstm(
                input_shape, hyperparameters
            )
        elif hyperparameters.network_type == Network.BI_LSTM:
            self.__input_shape = input_shape
            self.__model = self.__lstm(
                input_shape, hyperparameters, is_bidirectional=True
            )
        elif hyperparameters.network_type == Network.CONV_1D:
            self.__input_shape = input_shape
            self.__model = self.__conv1d(
                input_shape, hyperparameters
            )
        else:
            raise ValueError("Unknown network type. Should be one of %s", str(Network.TYPES))

        self.__n_type = hyperparameters.network_type
        self.hyperparameters = hyperparameters
        self.__LOGGER = Logger().get_logger(__name__)

    @classmethod
    def get_network(cls, input_shape: Tuple, hyperparameters: Hyperparameters) -> "Network":
        """Factory method for creating a network.

        Arguments:
            input_shape {tuple} -- A shape tuple (integers), not including the batch size.
            hyperparameters {Hyperparameters} -- A configuration for hyperparameters used for training.

        Returns:
            Network -- A constructed network object.
        """

        return cls(
            cls.__secret,
            input_shape,
            hyperparameters
        )

    @classmethod
    def get_from_model(cls, model_path: str, hyperparameters: Hyperparameters) -> "Network":
        """Load model into a network object.

        Arguments:
            model_path {string} -- The path to the model file.
            hyperparameters {Hyperparameters} -- A configuration for hyperparameters used for training.
        """

        hp = hyperparameters.clone()
        hp.network_type = Network.__UNKNOWN
        return cls(
            cls.__secret,
            (),
            hp,
            model_path=model_path
        )

    @classmethod
    def save_model_and_weights(
        cls, model_filepath: str, weights_filepath: str, combined_filepath: str
    ) -> None:
        """Combine model and weights and save to a file

        Arguments:
            model_filepath {string} -- The path to the model file.
            weights_filepath {string} -- The path to the weights file.
        """

        model = load_model(model_filepath)
        model.load_weights(weights_filepath)
        model.save(combined_filepath)

    @staticmethod
    def load_model_and_weights(model_filepath: str, weights_filepath: str, hyperparameters: Hyperparameters) -> "Network":
        """Load weights to the Network model.

        Arguments:
            model_filepath {string} -- The model file path.
            weights_filepath {string} -- The weights file path.
            hyperparameters {Hyperparameters} -- A configuration for hyperparameters used for training.

        Returns:
            Network -- Reconstructed network object.
        """
        network = Network.get_from_model(model_filepath, hyperparameters)
        network.__model.load_weights(weights_filepath)
        return network

    @property
    def input_shape(self) -> Tuple:
        """Get the input shape of the network.

        Returns:
            tuple -- The input shape of the network.
        """

        return self.__input_shape

    @property
    def n_type(self) -> str:
        """Get the type of the network.

        Returns:
            string -- The type of the network.
        """

        return self.__n_type

    @property
    def summary(self) -> None:
        """Print out the summary of the network.
        """

        self.__model.summary()

    @property
    def layers(self) -> List[Any]:
        """Get the layers of the network.

        Returns:
            list -- The statck of layers contained by the network
        """

        return self.__model.layers

    def get_predictions(self, input_data: np.ndarray, weights_filepath: str) -> np.ndarray:
        """Get a Numpy array of predictions.

        Arguments:
            input_data {numpy.ndarray} -- The input data, as a Numpy array.
            weights_filepath {string} -- The weights file path.

        Returns:
            numpy.ndarray -- The Numpy array of predictions.
        """
        self.__model.load_weights(weights_filepath)
        return self.__model.predict_on_batch(input_data)

    def fit_and_get_history(
        self,
        train_data: np.ndarray,
        labels: np.ndarray,
        model_filepath: str,
        weights_filepath: str,
        logs_dir: str,
        training_log: str,
        resume: bool,
    ) -> Tuple[List[float], List[float]]:
        """Fit the training data to the network and save the network model as a HDF file.

        Arguments:
            train_data {numpy.array} -- The Numpy array of training data.
            labels {numpy.array} -- The Numpy array of training labels.
            model_filepath {string} -- The model file path.
            weights_filepath {string} -- The weights file path.
            logs_dir {string} -- The TensorBoard log file directory.
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
            monitor=self.hyperparameters.monitor,
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
        earlyStopping = EarlyStopping(monitor=self.hyperparameters.monitor, min_delta=self.hyperparameters.es_min_delta,
                                      mode=self.hyperparameters.es_mode, patience=self.hyperparameters.es_patience,
                                      verbose=1)
        callbacks_list = [
            checkpoint,
            tensorboard,
            csv_logger,
            earlyStopping,
        ]
        if not resume:
            Optimizer = getattr(tf_optimizers, self.hyperparameters.optimizer)
            self.__model.compile(
                loss=self.hyperparameters.loss,
                optimizer=Optimizer(learning_rate=self.hyperparameters.learning_rate),
                metrics=self.hyperparameters.metrics,
            )
        initial_epoch = 0
        if resume:
            assert os.path.isfile(training_log), "{} does not exist and is required by training resumption".format(
                training_log)
            training_log_file = open(training_log)
            initial_epoch += sum(1 for _ in training_log_file) - 1
            training_log_file.close()
            assert self.hyperparameters.epochs > initial_epoch, \
                "The existing model has been trained for {0} epochs. Make sure the total epochs are larger than {0}".format(initial_epoch)
        try:
            hist = self.__model.fit(
                train_data,
                labels,
                epochs=self.hyperparameters.epochs,
                batch_size=self.hyperparameters.batch_size,
                shuffle=True,
                validation_split=self.hyperparameters.validation_split,
                verbose=1,
                callbacks=callbacks_list,
                initial_epoch=initial_epoch,
            )
        except KeyboardInterrupt:
            self.__LOGGER.warning("Training interrupted by the user")
            raise TerminalException("Training interrupted by the user")
        finally:
            save_model(self.__model, model_filepath)
            self.__LOGGER.warning("Model saved to %s" % model_filepath)

        return hist.history["val_loss"], hist.history["val_acc"] if int(tf.__version__.split(".")[0]) < 2 else hist.history["val_accuracy"]

    def fit_with_generator(
            self,
            train_data_raw: np.ndarray,
            labels_raw: np.ndarray,
            model_filepath: str,
            weights_filepath: str,
            logs_dir: str,
            training_log: str,
            resume: bool,
    ) -> Tuple[List[float], List[float]]:
        """Fit the training data to the network and save the network model as a HDF file.

        Arguments:
            train_data_raw {list} -- The HDF5 raw training data.
            labels_raw {list} -- The HDF5 raw training labels.
            model_filepath {string} -- The model file path.
            weights_filepath {string} -- The weights file path.
            logs_dir {string} -- The TensorBoard log file directory.
            training_log {string} -- The path to the log file of epoch results.
            resume {bool} -- True to continue with previous training result or False to start a new one (default: {False}).
        Returns:
            tuple -- A tuple contains validation losses and validation accuracies.
        """

        initial_epoch = 0
        batch_size = self.hyperparameters.batch_size
        validation_split = self.hyperparameters.validation_split
        csv_logger = (
            CSVLogger(training_log)
            if not resume
            else CSVLogger(training_log, append=True)
        )
        checkpoint = ModelCheckpoint(
            filepath=weights_filepath,
            monitor=self.hyperparameters.monitor,
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
        earlyStopping = EarlyStopping(monitor=self.hyperparameters.monitor, min_delta=self.hyperparameters.es_min_delta,
                                      mode=self.hyperparameters.es_mode, patience=self.hyperparameters.es_patience,
                                      verbose=1)
        callbacks_list = [
            checkpoint,
            tensorboard,
            csv_logger,
            earlyStopping,
        ]
        if not resume:
            Optimizer = getattr(tf_optimizers, self.hyperparameters.optimizer)
            self.__model.compile(
                loss=self.hyperparameters.loss,
                optimizer=Optimizer(learning_rate=self.hyperparameters.learning_rate),
                metrics=self.hyperparameters.metrics,
            )
        if resume:
            assert os.path.isfile(training_log), "{} does not exist and is required by training resumption".format(
                training_log)
            training_log_file = open(training_log)
            initial_epoch += sum(1 for _ in training_log_file) - 1
            training_log_file.close()
            assert self.hyperparameters.epochs > initial_epoch, \
                "The existing model has been trained for {0} epochs. Make sure the total epochs are larger than {0}".format(initial_epoch)

        train_generator = self.__generator(train_data_raw, labels_raw, batch_size, validation_split, is_validation=False)
        test_generator = self.__generator(train_data_raw, labels_raw, batch_size, validation_split, is_validation=True)
        steps_per_epoch = math.ceil(float(train_data_raw.shape[0]) * (1 - validation_split) / batch_size)
        validation_steps = math.ceil(float(train_data_raw.shape[0]) * validation_split / batch_size)

        try:
            hist = self.__model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                validation_data=test_generator,
                validation_steps=validation_steps,
                epochs=self.hyperparameters.epochs,
                shuffle=False,
                callbacks=callbacks_list,
                initial_epoch=initial_epoch,
            )
        except KeyboardInterrupt:
            self.__LOGGER.warning("Training interrupted by the user")
            raise TerminalException("Training interrupted by the user")
        finally:
            self.__model.save(model_filepath)
            self.__LOGGER.warning("Model saved to %s" % model_filepath)

        return hist.history["val_loss"], hist.history["val_acc"] if int(tf.__version__.split(".")[0]) < 2 else hist.history["val_accuracy"]

    @classmethod
    def simple_fit(
        cls,
        input_shape: Tuple,
        train_data: np.ndarray,
        labels: np.ndarray,
        hyperparameters: Hyperparameters,
    ) -> Tuple[List[float], List[float]]:
        """Fit the training data to the network and save the network model as a HDF file.

        Arguments:
            input_shape {tuple} -- A shape tuple (integers), not including the batch size.
            train_data {numpy.array} -- The Numpy array of training data.
            labels {numpy.array} -- The Numpy array of training labels.
            hyperparameters {Hyperparameters} -- A configuration for hyperparameters used for training.

        Returns:
            tuple -- A tuple contains validation losses and validation accuracies.
        """

        network = cls(cls.__secret, input_shape, hyperparameters)
        Optimizer = getattr(tf_optimizers, hyperparameters.optimizer)
        network.__model.compile(
            loss=hyperparameters.loss,
            optimizer=Optimizer(learning_rate=hyperparameters.learning_rate),
            metrics=hyperparameters.metrics,
        )
        initial_epoch = 0
        hist = network.__model.fit(
            train_data,
            labels,
            epochs=hyperparameters.epochs,
            batch_size=hyperparameters.batch_size,
            shuffle=True,
            validation_split=hyperparameters.validation_split,
            verbose=1,
            initial_epoch=initial_epoch,
        )

        return hist.history["val_loss"], hist.history["val_acc"] if int(tf.__version__.split(".")[0]) < 2 else hist.history["val_accuracy"]

    @classmethod
    def simple_fit_with_generator(
            cls,
            input_shape: Tuple,
            train_data_raw: np.ndarray,
            labels_raw: np.ndarray,
            hyperparameters: Hyperparameters,
    ) -> Tuple[List[float], List[float]]:
        """Fit the training data to the network and save the network model as a HDF file.

        Arguments:
            input_shape {tuple} -- A shape tuple (integers), not including the batch size.
            train_data_raw {list} -- The HDF5 raw training data.
            labels_raw {list} -- The HDF5 raw training labels.
            hyperparameters {Hyperparameters} -- A configuration for hyperparameters used for training.
        Returns:
            tuple -- A tuple contains validation losses and validation accuracies.
        """

        network = cls(cls.__secret, input_shape, hyperparameters)
        initial_epoch = 0
        batch_size = hyperparameters.batch_size
        validation_split = hyperparameters.validation_split
        Optimizer = getattr(tf_optimizers, hyperparameters.optimizer)
        network.__model.compile(
            loss=hyperparameters.loss,
            optimizer=Optimizer(learning_rate=hyperparameters.learning_rate),
            metrics=hyperparameters.metrics,
        )

        train_generator = cls.__generator(train_data_raw, labels_raw, batch_size, validation_split, is_validation=False)
        test_generator = cls.__generator(train_data_raw, labels_raw, batch_size, validation_split, is_validation=True)
        steps_per_epoch = math.ceil(float(train_data_raw.shape[0]) * (1 - validation_split) / batch_size)
        validation_steps = math.ceil(float(train_data_raw.shape[0]) * validation_split / batch_size)

        hist = network.__model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            validation_data=test_generator,
            validation_steps=validation_steps,
            epochs=hyperparameters.epochs,
            shuffle=False,
            initial_epoch=initial_epoch,
        )

        return hist.history["val_loss"], hist.history["val_acc"] if int(tf.__version__.split(".")[0]) < 2 else hist.history["val_accuracy"]

    @staticmethod
    def reset() -> None:
        K.clear_session()

    @staticmethod
    def __lstm(input_shape: Tuple, hyperparameters: Hyperparameters, is_bidirectional: bool = False) -> Model:
        inputs = Input(shape=input_shape)
        hidden = BatchNormalization()(inputs)

        for nodes in hyperparameters.front_hidden_size:
            hidden = Bidirectional(LSTM(nodes))(hidden) if is_bidirectional else LSTM(nodes)(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = Activation("relu")(hidden)
            hidden = Dropout(hyperparameters.dropout)(hidden)

        for nodes in hyperparameters.back_hidden_size:
            hidden = Dense(nodes)(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = Activation("relu")(hidden)
            hidden = Dropout(hyperparameters.dropout)(hidden)

        hidden = Dense(1)(hidden)
        outputs = Activation("sigmoid")(hidden)

        return Model(inputs, outputs)

    @staticmethod
    def __conv1d(input_shape: Tuple, hyperparameters: Hyperparameters) -> Model:
        inputs = Input(shape=input_shape)
        hidden = BatchNormalization()(inputs)

        for nodes in hyperparameters.front_hidden_size:
            hidden = Conv1D(filters=nodes, kernel_size=2, activation="relu", input_shape=hidden.shape[1:])(hidden)
            hidden = MaxPooling1D(pool_size=1)(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = Dropout(hyperparameters.dropout)(hidden)

        for nodes in hyperparameters.back_hidden_size:
            hidden = Dense(nodes)(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = Activation("relu")(hidden)
            hidden = Dropout(hyperparameters.dropout)(hidden)

        hidden = Dense(1)(hidden)
        outputs = Activation("sigmoid")(hidden)

        return Model(inputs, outputs)

    @staticmethod
    def __set_keras_backend(backend: str):
        # Changing backend is no longer supported by tf.keras in TF2
        if K.backend() != backend:
            os.environ["KERAS_BACKEND"] = backend
            importlib.reload(K)
            assert K.backend() == backend, "Unable to set backend to {}".format(backend)

        if backend.lower() == "tensorflow":
            # Set the number of inter/intra threads to the number of physical cores (experiment shows this is the best)
            physical_core_num = psutil.cpu_count(logical=False)
            tf.config.threading.set_inter_op_parallelism_threads(physical_core_num)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.set_soft_device_placement(True)
            physical_devices = tf.config.experimental.list_physical_devices("GPU")
            try:
                for gpu in physical_devices:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                raise ValueError("Invalid device or cannot modify virtual devices once initialised")
            K.clear_session()
        elif backend.lower() == "theano" or backend.lower() == "cntk":
            #  Backends other than tensorflow require separate installations before being used.
            if (int(tf.__version__.split(".")[0]) >= 2):
                raise ValueError("Multi-backend is not supported")
        else:
            raise ValueError("Unknown backend: {}".format(backend))

    @staticmethod
    def __generator(train_data_raw: np.ndarray, labels_raw: np.ndarray, batch_size: int, validation_split: float, is_validation: bool) -> Generator:
        while True:
            total_size = train_data_raw.shape[0]
            for i in range(0, total_size, batch_size):
                real_batch_size = total_size - i - 1 if total_size - i - 1 < batch_size else batch_size
                train_range_right = i + int(real_batch_size * (1 - validation_split))
                if is_validation:
                    batched_train_data = train_data_raw[train_range_right:i + real_batch_size]
                    batched_labels = labels_raw[train_range_right:i + real_batch_size]
                else:
                    batched_train_data = train_data_raw[i:train_range_right]
                    batched_labels = labels_raw[i:train_range_right]

                np_batched_train_data = np.array(batched_train_data)
                np_batched_labels = np.array(batched_labels)

                rand = np.random.permutation(np.arange(len(np_batched_labels)))
                np_batched_random_train_data = np_batched_train_data[rand]
                np_batched_random_labels = np_batched_labels[rand]

                np_batched_random_train_data = np.array(
                    [np.rot90(m=val, k=1, axes=(0, 1)) for val in np_batched_random_train_data]
                )
                np_batched_random_train_data = np_batched_random_train_data - np.mean(np_batched_random_train_data,
                                                                                      axis=0)

                yield np_batched_random_train_data, np_batched_random_labels
