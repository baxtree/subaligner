import unittest
import os
import shutil
import sys
from subaligner.hyperparameters import Hyperparameters as Undertest


class TestHyperparameters(unittest.TestCase):

    def setUp(self):
        self.hyperparams_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/models/training/config/hyperparameters.json"
        )
        self.resource_tmp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/tmp"
        )
        if os.path.exists(self.resource_tmp):
            shutil.rmtree(self.resource_tmp)
        os.mkdir(self.resource_tmp)

    def tearDown(self):
        shutil.rmtree(self.resource_tmp)

    def test_default_params(self):
        hyperparams = Undertest()

        self.assertEqual(0.001, hyperparams.learning_rate)
        self.assertEqual([64], hyperparams.front_hidden_size)
        self.assertEqual([32, 16], hyperparams.back_hidden_size)
        self.assertEqual(0.2, hyperparams.dropout)
        self.assertEqual(100, hyperparams.epochs)
        self.assertEqual("Adam", hyperparams.optimizer)
        self.assertEqual("binary_crossentropy", hyperparams.loss)
        self.assertEqual(["accuracy"], hyperparams.metrics)
        self.assertEqual(32, hyperparams.batch_size)
        self.assertEqual(0.25, hyperparams.validation_split)
        self.assertEqual("val_loss", hyperparams.monitor)
        self.assertEqual("min", hyperparams.es_mode)
        self.assertEqual(0.00001, hyperparams.es_min_delta)
        self.assertEqual(sys.maxsize, hyperparams.es_patience)
        self.assertEqual("lstm", hyperparams.network_type)

    def test_serialisation(self):
        with open(self.hyperparams_file_path, "r") as file:
            expected = Undertest()
            expected.epochs = 1
            expected.es_patience = 1000000
            self.assertEqual(expected.to_json(), file.read())

    def test_deserialisation(self):
        with open(self.hyperparams_file_path, "r") as file:
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
        self.assertEqual(1000000, hyperparams.es_patience)
        self.assertEqual("lstm", hyperparams.network_type)

    def test_saved_to_file(self):
        hp_file_path = "{}/{}".format(self.resource_tmp, "hyperparameters.json")
        hyperparameters = Undertest()
        hyperparameters.to_file(hp_file_path)
        with open(hp_file_path, "r") as file:
            saved = Undertest.from_json(file.read())
        self.assertEqual(hyperparameters, saved)

    def test_load_from_file(self):
        expected = Undertest()
        expected.epochs = 1
        expected.es_patience = 1000000
        hyperparams = Undertest.from_file(self.hyperparams_file_path)
        self.assertEqual(expected, hyperparams)

    def test_optimizer_setter(self):
        hyperparams = Undertest()
        hyperparams.optimizer = "adadelta"
        hyperparams.optimizer = "adagrad"
        hyperparams.optimizer = "adam"
        hyperparams.optimizer = "adamax"
        hyperparams.optimizer = "ftrl"
        hyperparams.optimizer = "nadam"
        hyperparams.optimizer = "rmsprop"
        hyperparams.optimizer = "sgd"
        try:
            hyperparams.optimizer = "unknown"
            self.fail("Should have errored out")
        except ValueError as e:
            self.assertEqual("Optimizer unknown is not supported", str(e))

    def test_clone(self):
        hyperparams = Undertest()
        clone = hyperparams.clone()
        self.assertFalse(hyperparams is clone)
        self.assertTrue(hyperparams == clone)


if __name__ == "__main__":
    unittest.main()
