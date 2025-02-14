import unittest
import os
import pickle
import shutil
import h5py
import numpy as np

from mock import patch
from parameterized import parameterized
from keras.models import Model
from subaligner.hyperparameters import Hyperparameters
from subaligner.exception import TerminalException
from subaligner.network import Network as Undertest


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.hyperparameters = Hyperparameters()
        self.hyperparameters.epochs = 1
        self.model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/models/training/model"
        )
        self.weights_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/models/training/weights"
        )
        self.train_data = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/train_data"
        )
        self.labels = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/labels"
        )
        self.training_dump = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/training_dump.hdf5"
        )
        self.resource_tmp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/tmp"
        )
        if os.path.exists(self.resource_tmp):
            shutil.rmtree(self.resource_tmp)
        os.mkdir(self.resource_tmp)

    def tearDown(self):
        for file in os.listdir(self.model_dir):
            os.remove(os.path.join(self.model_dir, file)) if not file.endswith(
                ".hdf5"
            ) else None
        shutil.rmtree(self.resource_tmp)
        Undertest.reset()

    def test_suppressed_init(self):
        self.assertRaises(
            AssertionError, Undertest, "guess", (2, 20), "lstm", None, None
        )

    def test_get_from_model(self):
        model_filepath = "{}/{}".format(self.model_dir, "model.hdf5")
        network = Undertest.get_from_model(model_filepath, self.hyperparameters)
        self.assertEqual((2, 20), network.input_shape)
        self.assertEqual("unknown", network.n_type)

    def test_save_model_and_weights(self):
        model_filepath = "{}/{}".format(self.model_dir, "model.hdf5")
        weights_filepath = "{}/{}".format(self.weights_dir, "weights.hdf5")
        Undertest.save_model_and_weights(
            model_filepath,
            weights_filepath,
            "{}/{}".format(self.resource_tmp, "model_combined.hdf5"),
        )
        self.assertEqual((2, 20), Undertest.get_from_model(model_filepath, self.hyperparameters).input_shape)

    def test_input_shape(self):
        network = Undertest.get_network((2, 20), self.hyperparameters)
        self.assertEqual((2, 20), network.input_shape)

    @parameterized.expand([
        ["lstm"],
        ["bi_lstm"],
        ["conv_1d"],
    ])
    def test_create_network(self, network_type):
        self.hyperparameters.network_type = network_type
        network = Undertest.get_network((2, 20), self.hyperparameters)
        self.assertEqual(network_type, network.n_type)

    def test_summary(self):
        network = Undertest.get_network((2, 20), self.hyperparameters)
        self.assertTrue(network.summary is None)

    def test_layers(self):
        network = Undertest.get_network((2, 20), self.hyperparameters)
        self.assertEqual(16, len(network.layers))

    def test_get_predictions(self):
        network = Undertest.get_from_model("{}/model.hdf5".format(self.model_dir), self.hyperparameters)
        with open(self.train_data, "rb") as file:
            train_data = pickle.load(file)
        self.assertEqual((11431, 1), network.get_predictions(train_data, "{}/weights.hdf5".format(self.weights_dir)).shape)

    def test_load_model_and_weights(self):
        network_old = Undertest.get_network((2, 20), self.hyperparameters)
        weights_old = network_old._Network__model.get_weights()
        network_new = Undertest.load_model_and_weights("{}/model.hdf5".format(self.model_dir), "{}/weights.hdf5".format(self.weights_dir),
                                                       self.hyperparameters)
        weights_new = network_new._Network__model.get_weights()
        self.assertFalse(np.array_equal(weights_old, weights_new))

    @parameterized.expand([
        ["lstm"],
        ["bi_lstm"],
    ])
    def test_fit_lstm_and_get_history(self, network_type):
        self.hyperparameters.network_type = network_type
        network = Undertest.get_network((2, 20), self.hyperparameters)
        model_filepath = "{}/model.hdf5".format(self.resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.resource_tmp)
        with open(self.train_data, "rb") as file:
            train_data = pickle.load(file)
        with open(self.labels, "rb") as file:
            labels = pickle.load(file)
        val_loss, val_acc = network.fit_and_get_history(
            train_data,
            labels,
            model_filepath,
            weights_filepath,
            self.resource_tmp,
            "training.log",
            False,
        )
        self.assertEqual(list, type(val_loss))
        self.assertEqual(list, type(val_acc))

    def test_resume_and_get_history(self):
        self.hyperparameters.epochs = 2
        network = Undertest.get_network((2, 20), self.hyperparameters)
        model_filepath = "{}/model.hdf5".format(self.resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.resource_tmp)
        with open(self.train_data, "rb") as file:
            train_data = pickle.load(file)
        with open(self.labels, "rb") as file:
            labels = pickle.load(file)
        _, _ = network.fit_and_get_history(
            train_data,
            labels,
            model_filepath,
            weights_filepath,
            self.resource_tmp,
            "{}/training.log".format(self.resource_tmp),
            False,
        )
        self.hyperparameters.epochs = 3
        val_loss, val_acc = network.fit_and_get_history(
            train_data,
            labels,
            model_filepath,
            weights_filepath,
            self.resource_tmp,
            "{}/training.log".format(self.resource_tmp),
            True,
        )
        self.assertEqual(list, type(val_loss))
        self.assertEqual(list, type(val_acc))

    def test_exception_on_creating_an_unknown_network(self):
        self.hyperparameters.network_type = "unknown"
        try:
            Undertest.get_network((2, 20), self.hyperparameters)
        except Exception as e:
            self.assertTrue(isinstance(e, ValueError))
            self.assertTrue("Unknown network type" in str(e))
        else:
            self.fail("Should have thrown exception")

    def test_exception_on_resume_with_no_extra_epochs(self):
        self.hyperparameters.epochs = 2
        network = Undertest.get_network((2, 20), self.hyperparameters)
        model_filepath = "{}/model.hdf5".format(self.resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.resource_tmp)
        with open(self.train_data, "rb") as file:
            train_data = pickle.load(file)
        with open(self.labels, "rb") as file:
            labels = pickle.load(file)
        _, _ = network.fit_and_get_history(
            train_data,
            labels,
            model_filepath,
            weights_filepath,
            self.resource_tmp,
            "{}/training.log".format(self.resource_tmp),
            False,
        )
        try:
            network.fit_and_get_history(
                train_data,
                labels,
                model_filepath,
                weights_filepath,
                self.resource_tmp,
                "{}/training.log".format(self.resource_tmp),
                True,
            )
        except Exception as e:
            self.assertTrue(isinstance(e, AssertionError))
            self.assertEqual("The existing model has been trained for 2 epochs. Make sure the total epochs are larger than 2", str(e))
        else:
            self.fail("Should have thrown exception")

    def test_exception_on_resume_with_no_previous_training_log(self):
        self.hyperparameters.epochs = 2
        network = Undertest.get_network((2, 20), self.hyperparameters)
        model_filepath = "{}/model.hdf5".format(self.resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.resource_tmp)
        with open(self.train_data, "rb") as file:
            train_data = pickle.load(file)
        with open(self.labels, "rb") as file:
            labels = pickle.load(file)
        _, _ = network.fit_and_get_history(
            train_data,
            labels,
            model_filepath,
            weights_filepath,
            self.resource_tmp,
            "{}/training_1.log".format(self.resource_tmp),
            False,
        )
        try:
            network.fit_and_get_history(
                train_data,
                labels,
                model_filepath,
                weights_filepath,
                self.resource_tmp,
                "{}/training_2.log".format(self.resource_tmp),
                True,
            )
        except Exception as e:
            self.assertTrue(isinstance(e, AssertionError))
            self.assertTrue("does not exist and is required by training resumption" in str(e))
        else:
            self.fail("Should have thrown exception")

    def test_fit_with_generator(self):
        self.hyperparameters.epochs = 3
        network = Undertest.get_network((2, 20), self.hyperparameters)
        model_filepath = "{}/model.hdf5".format(self.resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.resource_tmp)
        with h5py.File(self.training_dump, "r") as hf:
            val_loss, val_acc = network.fit_with_generator(
                hf["train_data"],
                hf["labels"],
                model_filepath,
                weights_filepath,
                self.resource_tmp,
                "training.log",
                False,
            )
            self.assertEqual(list, type(val_loss))
            self.assertEqual(list, type(val_acc))
            self.assertTrue(len(val_loss) == self.hyperparameters.epochs)
            self.assertTrue(len(val_acc) == self.hyperparameters.epochs)

    def test_early_stop_with_patience(self):
        self.hyperparameters.epochs = 3
        self.hyperparameters.es_patience = 0
        network = Undertest.get_network((2, 20), self.hyperparameters)
        model_filepath = "{}/model.hdf5".format(self.resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.resource_tmp)
        with h5py.File(self.training_dump, "r") as hf:
            val_loss, val_acc = network.fit_with_generator(
                hf["train_data"],
                hf["labels"],
                model_filepath,
                weights_filepath,
                self.resource_tmp,
                "training.log",
                False,
            )
            self.assertEqual(list, type(val_loss))
            self.assertEqual(list, type(val_acc))
            self.assertTrue(len(val_loss) < self.hyperparameters.epochs)
            self.assertTrue(len(val_acc) < self.hyperparameters.epochs)

    def test_simple_fit(self):
        with open(self.train_data, "rb") as file:
            train_data = pickle.load(file)
        with open(self.labels, "rb") as file:
            labels = pickle.load(file)
        val_loss, val_acc = Undertest.simple_fit(
            (2, 20),
            train_data,
            labels,
            self.hyperparameters,
        )
        self.assertEqual(list, type(val_loss))
        self.assertEqual(list, type(val_acc))

    def test_simple_fit_with_generator(self):
        self.hyperparameters.epochs = 3
        with h5py.File(self.training_dump, "r") as hf:
            val_loss, val_acc = Undertest.simple_fit_with_generator(
                (2, 20),
                hf["train_data"],
                hf["labels"],
                self.hyperparameters,
            )
            self.assertEqual(list, type(val_loss))
            self.assertEqual(list, type(val_acc))
            self.assertTrue(len(val_loss) == self.hyperparameters.epochs)
            self.assertTrue(len(val_acc) == self.hyperparameters.epochs)

    @patch("keras.models.Model.fit", side_effect=KeyboardInterrupt)
    def test_throw_exception_on_fit_and_get_history(self, mock_fit):
        try:
            network = Undertest.get_network((2, 20), self.hyperparameters)
            model_filepath = "{}/model.hdf5".format(self.resource_tmp)
            weights_filepath = "{}/weights.hdf5".format(self.resource_tmp)
            with open(self.train_data, "rb") as file:
                train_data = pickle.load(file)
            with open(self.labels, "rb") as file:
                labels = pickle.load(file)
            network.fit_and_get_history(
                train_data,
                labels,
                model_filepath,
                weights_filepath,
                self.resource_tmp,
                "training.log",
                False,
            )
        except Exception as e:
            self.assertTrue(mock_fit.called)
            self.assertTrue(isinstance(e, TerminalException))
            self.assertTrue("interrupted" in str(e))
        else:
            self.fail("Should have thrown exception")

    @patch("keras.models.Model.fit", side_effect=KeyboardInterrupt)
    def test_throw_exception_on_fit_with_generator(self, mock_fit):
        self.hyperparameters.epochs = 3
        network = Undertest.get_network((2, 20), self.hyperparameters)
        model_filepath = "{}/model.hdf5".format(self.resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.resource_tmp)
        with h5py.File(self.training_dump, "r") as hf:
            try:
                network.fit_with_generator(
                    hf["train_data"],
                    hf["labels"],
                    model_filepath,
                    weights_filepath,
                    self.resource_tmp,
                    "training.log",
                    False,
                )
            except Exception as e:
                self.assertTrue(mock_fit.called)
                self.assertTrue(isinstance(e, TerminalException))
                self.assertTrue("interrupted" in str(e))
            else:
                self.fail("Should have thrown exception")


if __name__ == "__main__":
    unittest.main()
