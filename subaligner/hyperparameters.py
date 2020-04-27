import sys
import tensorflow as tf


class Hyperparameters(object):
    """ The configuration for hyper parameters used for training
    """

    def __init__(self):
        self.__learning_rate = 0.001
        self.__hidden_size = {
            "front_layers": [64],
            "back_layers": [32, 16]
        }
        self.__dropout = 0.2
        self.__epochs = 100
        self.__optimizer = tf.keras.optimizers.Adam(lr=0.001)
        self.__loss = "binary_crossentropy"
        self.__metrics = ["accuracy"]
        self.__batch_size = 32
        self.__validation_split = 0.25
        self.__monitor = "val_loss"
        self.__es_mode = "min"
        self.__es_min_delta = 0.00001
        self.__es_patience = sys.maxsize

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self.__learning_rate = value

    @property
    def front_hidden_size(self):
        return self.__hidden_size["front_layers"]

    @front_hidden_size.setter
    def front_hidden_size(self, value):
        self.__hidden_size["front_layers"] = value

    @property
    def back_hidden_size(self):
        return self.__hidden_size["back_layers"]

    @back_hidden_size.setter
    def back_hidden_size(self, value):
        self.__hidden_size["back_layers"] = value

    @property
    def dropout(self):
        return self.__dropout

    @dropout.setter
    def dropout(self, value):
        self.__dropout = value

    @property
    def epochs(self):
        return self.__epochs

    @epochs.setter
    def epochs(self, value):
        self.__epochs = value

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, value):
        if value.lower() == "adam":
            self.__optimizer = tf.keras.optimizers.Adam(learning_rate=self.__learning_rate)
        elif value.lower == "adagrad":
            self.__optimizer = tf.keras.optimizers.Adagrad(learning_rate=self.__learning_rate)
        elif value.lower == "rms":
            self.__optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.__learning_rate)
        elif value.lower == "sgd":
            self.__optimizer = tf.keras.optimizers.SGD(learning_rate=self.__learning_rate)
        else:
            raise ValueError("Optimizer {} is not supported".format(value))

    @property
    def loss(self):
        return self.__loss

    @property
    def metrics(self):
        return self.__metrics

    @metrics.setter
    def metrics(self, value):
        self.__metrics = value

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.__batch_size = value

    @property
    def validation_split(self):
        return self.__validation_split

    @validation_split.setter
    def validation_split(self, value):
        self.__validation_split = value

    @property
    def monitor(self):
        return self.__monitor

    @monitor.setter
    def monitor(self, value):
        self.__monitor = value

    @property
    def es_mode(self):
        return self.__es_mode

    @es_mode.setter
    def es_mode(self, value):
        self.__es_mode = value

    @property
    def es_min_delta(self):
        return self.__es_min_delta

    @es_min_delta.setter
    def es_min_delta(self, value):
        self.__es_min_delta = value

    @property
    def es_patience(self):
        return self.__es_patience

    @es_patience.setter
    def es_patience(self, value):
        self.__es_patience = value
