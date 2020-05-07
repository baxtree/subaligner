import sys
import json


class Hyperparameters(object):
    """ The configuration on hyper parameters used for training
    """

    def __init__(self):
        """Hyper parameters initialiser setting default values"""

        self.__learning_rate = 0.001
        self.__hidden_size = {
            "front_layers": [64],
            "back_layers": [32, 16]
        }
        self.__dropout = 0.2
        self.__epochs = 100
        self.__optimizer = "Adam"
        self.__loss = "binary_crossentropy"
        self.__metrics = ["accuracy"]
        self.__batch_size = 32
        self.__validation_split = 0.25
        self.__monitor = "val_loss"
        self.__es_mode = "min"
        self.__es_min_delta = 0.00001
        self.__es_patience = sys.maxsize
        self.__network_type = "lstm"

    def __eq__(self, other):
        """Comparator for Hyperparameters objects"""

        if isinstance(other, Hyperparameters):
            return all([
                self.__learning_rate == other.learning_rate,
                self.__hidden_size["front_layers"] == other.front_hidden_size,
                self.__hidden_size["back_layers"] == other.back_hidden_size,
                self.__dropout == other.dropout,
                self.__epochs == other.epochs,
                self.__optimizer == other.optimizer,
                self.__loss == other.loss,
                self.__metrics == other.metrics,
                self.__batch_size == other.batch_size,
                self.__validation_split == other.validation_split,
                self.__monitor == other.monitor,
                self.__es_mode == other.es_mode,
                self.__es_min_delta == other.es_min_delta,
                self.__es_patience == other.es_patience,
                self.__network_type == other.network_type
            ])
        return False

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
            self.__optimizer = "Adam"
        elif value.lower() == "adagrad":
            self.__optimizer = "Adagrad"
        elif value.lower() == "rms":
            self.__optimizer = "RMSprop"
        elif value.lower() == "sgd":
            self.__optimizer = "SGD"
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

    @property
    def network_type(self):
        return self.__network_type

    @network_type.setter
    def network_type(self, value):
        self.__network_type = value

    def to_json(self):
        """Serialise hyper parameters into JSON string

        Returns:
            string -- The serialised hyper parameters in JSON
        """
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_file(self, file_path):
        """Serialise hyper parameters into JSON and save the content to a file

        Arguments:
            file_path {string} -- The path to the file containing saved hyper parameters.
        """
        with open(file_path, "w", encoding="utf8") as file:
            file.write(self.to_json())

    def clone(self):
        """Make a cloned hyper parameters object

        Returns:
            Hyperparameters -- The cloned Hyperparameters object.
        """
        return self.from_json(self.to_json())

    @classmethod
    def from_json(cls, json_str):
        """Deserialise JSON string into a Hyperparameters object

        Arguments:
            json_str {string} -- Hyper parameters in JSON.

        Returns:
            Hyperparameters -- The deserialised Hyperparameters object.
        """
        hp = cls()
        hp.__dict__ = json.loads(json_str)
        return hp

    @classmethod
    def from_file(cls, file_path):
        """Deserialise a file content into a Hyperparameters object

        Arguments:
            file_path {string} -- The path to the file containing hyper parameters.

        Returns:
            Hyperparameters -- The deserialised Hyperparameters object.
        """
        with open(file_path, "r", encoding="utf8") as file:
            return cls.from_json(file.read())
