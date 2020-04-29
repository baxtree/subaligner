import unittest
import os
import shutil
import sys
from subaligner.hyperparameters import Hyperparameters as Undertest


class TestHyperparameters(unittest.TestCase):

    def setUp(self):
        self.__hyperparams_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/models/training/config/hyperparameters.json"
        )
        self.__resource_tmp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/tmp"
        )
        if os.path.exists(self.__resource_tmp):
            shutil.rmtree(self.__resource_tmp)
        os.mkdir(self.__resource_tmp)

    def tearDown(self):
        shutil.rmtree(self.__resource_tmp)

    def test_default_params(self):
        unittest = Undertest()

        self.assertEqual(0.001, unittest.learning_rate)
        self.assertEqual([64], unittest.front_hidden_size)
        self.assertEqual([32, 16], unittest.back_hidden_size)
        self.assertEqual(0.2, unittest.dropout)
        self.assertEqual(100, unittest.epochs)
        self.assertEqual("Adam", unittest.optimizer)
        self.assertEqual("binary_crossentropy", unittest.loss)
        self.assertEqual(["accuracy"], unittest.metrics)
        self.assertEqual(32, unittest.batch_size)
        self.assertEqual(0.25, unittest.validation_split)
        self.assertEqual("val_loss", unittest.monitor)
        self.assertEqual("min", unittest.es_mode)
        self.assertEqual(0.00001, unittest.es_min_delta)
        self.assertEqual(sys.maxsize, unittest.es_patience)

    def test_serialisation(self):
        with open(self.__hyperparams_file_path, "r") as file:
            expected = Undertest()
            expected.epochs = 1
            self.assertEqual(expected.to_json(), file.read())

    def test_deserialisation(self):
        with open(self.__hyperparams_file_path, "r") as file:
            hyperparams = Undertest.from_json(file.read())

        self.assertEqual(0.001, hyperparams.learning_rate)
        self.assertEqual([64], hyperparams.front_hidden_size)
        self.assertEqual([32, 16], hyperparams.back_hidden_size)
        self.assertEqual(0.2, hyperparams.dropout)
        self.assertEqual(1, hyperparams.epochs)
        self.assertEqual("Adam", hyperparams.optimizer)
        self.assertEqual("binary_crossentropy", hyperparams.loss)
        self.assertEqual(["accuracy"], hyperparams.metrics)
        self.assertEqual(32, hyperparams.batch_size)
        self.assertEqual(0.25, hyperparams.validation_split)
        self.assertEqual("val_loss", hyperparams.monitor)
        self.assertEqual("min", hyperparams.es_mode)
        self.assertEqual(0.00001, hyperparams.es_min_delta)
        self.assertEqual(sys.maxsize, hyperparams.es_patience)

    def test_saved_to_file(self):
        hp_file_path = "{}/{}".format(self.__resource_tmp, "hyperparameters.json")
        hyperparameters = Undertest()
        hyperparameters.to_file(hp_file_path)
        with open(hp_file_path, "r") as file:
            saved = Undertest.from_json(file.read())
        self.assertEqual(hyperparameters, saved)

    def test_load_from_file(self):
        expected = Undertest()
        expected.epochs = 1
        hyperparams = Undertest.from_file(self.__hyperparams_file_path)
        self.assertEqual(expected, hyperparams)


if __name__ == "__main__":
    unittest.main()
