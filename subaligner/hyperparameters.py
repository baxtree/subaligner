import sys
import json
from typing import Any, List


class Hyperparameters(object):
    """ The configuration on hyperparameters used for training
    """

    OPTIMIZERS = ["adadelta", "adagrad", "adam", "adamax", "ftrl", "nadam", "rmsprop", "sgd"]

    def __init__(self) -> None:
        """Hyperparameters initialiser setting default values"""

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

    def __eq__(self, other: Any) -> bool:
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
    def learning_rate(self) -> float:
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.__learning_rate = value

    @property
    def front_hidden_size(self) -> List[int]:
        return self.__hidden_size["front_layers"]

    @front_hidden_size.setter
    def front_hidden_size(self, value: List[int]) -> None:
        self.__hidden_size["front_layers"] = value

    @property
    def back_hidden_size(self) -> List[int]:
        return self.__hidden_size["back_layers"]

    @back_hidden_size.setter
    def back_hidden_size(self, value: List[int]) -> None:
        self.__hidden_size["back_layers"] = value

    @property
    def dropout(self) -> float:
        return self.__dropout

    @dropout.setter
    def dropout(self, value: float) -> None:
        self.__dropout = value

    @property
    def epochs(self) -> int:
        return self.__epochs

    @epochs.setter
    def epochs(self, value: int) -> None:
        self.__epochs = value

    @property
    def optimizer(self) -> str:
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, value: str) -> None:
        if value not in self.OPTIMIZERS:
            raise ValueError("Optimizer {} is not supported".format(value))

        if value.lower() == "adadelta":
            self.__optimizer = "Adadelta"
        elif value.lower() == "adagrad":
            self.__optimizer = "Adagrad"
        elif value.lower() == "adam":
            self.__optimizer = "Adam"
        elif value.lower() == "adamax":
            self.__optimizer = "Adamax"
        elif value.lower() == "ftrl":
            self.__optimizer = "Ftrl"
        elif value.lower() == "nadam":
            self.__optimizer = "Nadam"
        elif value.lower() == "rmsprop":
            self.__optimizer = "RMSprop"
        elif value.lower() == "sgd":
            self.__optimizer = "SGD"

    @property
    def loss(self) -> str:
        return self.__loss

    @property
    def metrics(self) -> List[str]:
        return self.__metrics

    @metrics.setter
    def metrics(self, value: List[str]):
        self.__metrics = value

    @property
    def batch_size(self) -> int:
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.__batch_size = value

    @property
    def validation_split(self) -> float:
        return self.__validation_split

    @validation_split.setter
    def validation_split(self, value: float) -> None:
        self.__validation_split = value

    @property
    def monitor(self) -> str:
        return self.__monitor

    @monitor.setter
    def monitor(self, value: str) -> None:
        self.__monitor = value

    @property
    def es_mode(self) -> str:
        return self.__es_mode

    @es_mode.setter
    def es_mode(self, value: str) -> None:
        self.__es_mode = value

    @property
    def es_min_delta(self) -> float:
        return self.__es_min_delta

    @es_min_delta.setter
    def es_min_delta(self, value: float) -> None:
        self.__es_min_delta = value

    @property
    def es_patience(self) -> int:
        return self.__es_patience

    @es_patience.setter
    def es_patience(self, value: int) -> None:
        self.__es_patience = value

    @property
    def network_type(self) -> str:
        return self.__network_type

    @network_type.setter
    def network_type(self, value: str) -> None:
        self.__network_type = value

    def to_json(self) -> str:
        """Serialise hyperparameters into JSON string

        Returns:
            string -- The serialised hyperparameters in JSON
        """
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_file(self, file_path: str) -> None:
        """Serialise hyperparameters into JSON and save the content to a file

        Arguments:
            file_path {string} -- The path to the file containing saved hyperparameters.
        """
        with open(file_path, "w", encoding="utf8") as file:
            file.write(self.to_json())

    def clone(self) -> "Hyperparameters":
        """Make a cloned hyperparameters object

        Returns:
            Hyperparameters -- The cloned Hyperparameters object.
        """
        return self.from_json(self.to_json())

    @classmethod
    def from_json(cls, json_str: str) -> "Hyperparameters":
        """Deserialise JSON string into a Hyperparameters object

        Arguments:
            json_str {string} -- Hyperparameters in JSON.

        Returns:
            Hyperparameters -- The deserialised Hyperparameters object.
        """
        hp = cls()
        hp.__dict__ = json.loads(json_str)
        return hp

    @classmethod
    def from_file(cls, file_path: str) -> "Hyperparameters":
        """Deserialise a file content into a Hyperparameters object

        Arguments:
            file_path {string} -- The path to the file containing hyperparameters.

        Returns:
            Hyperparameters -- The deserialised Hyperparameters object.
        """
        with open(file_path, "r", encoding="utf8") as file:
            return cls.from_json(file.read())
