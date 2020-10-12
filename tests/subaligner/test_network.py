import unittest
import os
import pickle
import shutil
import h5py
import numpy as np

from subaligner.hyperparameters import Hyperparameters
from subaligner.network import Network as Undertest


class NetworkTests(unittest.TestCase):
    def setUp(self):
        self.__hyperparameters = Hyperparameters()
        self.__hyperparameters.epochs = 1
        self.__model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/models/training/model"
        )
        self.__weights_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/models/training/weights"
        )
        self.__train_data = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/train_data"
        )
        self.__labels = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/labels"
        )
        self.__training_dump = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/training_dump.hdf5"
        )
        self.__resource_tmp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/tmp"
        )
        if os.path.exists(self.__resource_tmp):
            shutil.rmtree(self.__resource_tmp)
        os.mkdir(self.__resource_tmp)

    def tearDown(self):
        for file in os.listdir(self.__model_dir):
            os.remove(os.path.join(self.__model_dir, file)) if not file.endswith(
                ".hdf5"
            ) else None
        shutil.rmtree(self.__resource_tmp)
        Undertest.reset()

    def test_suppressed_init(self):
        self.assertRaises(
            AssertionError, Undertest, "guess", (2, 20), "lstm", None, None
        )

    def test_get_from_model(self):
        model_filepath = "{}/{}".format(self.__model_dir, "model.hdf5")
        network = Undertest.get_from_model(model_filepath, self.__hyperparameters)
        self.assertEqual((2, 20), network.input_shape)
        self.assertEqual("unknown", network.n_type)

    def test_save_model_and_weights(self):
        model_filepath = "{}/{}".format(self.__model_dir, "model.hdf5")
        weights_filepath = "{}/{}".format(self.__weights_dir, "weights.hdf5")
        Undertest.save_model_and_weights(
            model_filepath,
            weights_filepath,
            "{}/{}".format(self.__resource_tmp, "model_combined.hdf5"),
        )
        self.assertEqual((2, 20), Undertest.get_from_model(model_filepath, self.__hyperparameters).input_shape)

    def test_input_shape(self):
        network = Undertest.get_network((2, 20), self.__hyperparameters)
        self.assertEqual((2, 20), network.input_shape)

    def test_create_lstm_network(self):
        network = Undertest.get_network((2, 20), self.__hyperparameters)
        self.assertEqual("lstm", network.n_type)

    def test_create_bi_lstm_network(self):
        self.__hyperparameters.network_type = "bi_lstm"
        network = Undertest.get_network((2, 20), self.__hyperparameters)
        self.assertEqual("bi_lstm", network.n_type)

    def test_create_conv_1d_network(self):
        self.__hyperparameters.network_type = "conv_1d"
        network = Undertest.get_network((2, 20), self.__hyperparameters)
        self.assertEqual("conv_1d", network.n_type)

    def test_summary(self):
        network = Undertest.get_network((2, 20), self.__hyperparameters)
        self.assertTrue(network.summary is None)  # Why this is None

    def test_get_predictions(self):
        network = Undertest.get_from_model("{}/model.hdf5".format(self.__model_dir), self.__hyperparameters)
        with open(self.__train_data, "rb") as file:
            train_data = pickle.load(file)
        self.assertEqual((11431, 1), network.get_predictions(train_data, "{}/weights.hdf5".format(self.__weights_dir)).shape)

    def test_load_model_and_weights(self):
        network_old = Undertest.get_network((2, 20), self.__hyperparameters)
        weights_old = network_old._Network__model.get_weights()
        network_new = Undertest.load_model_and_weights("{}/model.hdf5".format(self.__model_dir), "{}/weights.hdf5".format(self.__weights_dir),
                                                       self.__hyperparameters)
        weights_new = network_new._Network__model.get_weights()
        self.assertFalse(np.array_equal(weights_old, weights_new))

    def test_fit_lstm_and_get_history(self):
        network = Undertest.get_network((2, 20), self.__hyperparameters)
        model_filepath = "{}/model.hdf5".format(self.__resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.__resource_tmp)
        with open(self.__train_data, "rb") as file:
            train_data = pickle.load(file)
        with open(self.__labels, "rb") as file:
            labels = pickle.load(file)
        val_loss, val_acc = network.fit_and_get_history(
            train_data,
            labels,
            model_filepath,
            weights_filepath,
            self.__resource_tmp,
            "training.log",
            False,
        )
        self.assertEqual(list, type(val_loss))
        self.assertEqual(list, type(val_acc))

    def test_fit_bi_lstm_and_get_history(self):
        self.__hyperparameters.network_type = "bi_lstm"
        network = Undertest.get_network((2, 20), self.__hyperparameters)
        model_filepath = "{}/model.hdf5".format(self.__resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.__resource_tmp)
        with open(self.__train_data, "rb") as file:
            train_data = pickle.load(file)
        with open(self.__labels, "rb") as file:
            labels = pickle.load(file)
        val_loss, val_acc = network.fit_and_get_history(
            train_data,
            labels,
            model_filepath,
            weights_filepath,
            self.__resource_tmp,
            "training.log",
            False,
        )
        self.assertEqual(list, type(val_loss))
        self.assertEqual(list, type(val_acc))

    def test_resume_and_get_history(self):
        self.__hyperparameters.epochs = 2
        network = Undertest.get_network((2, 20), self.__hyperparameters)
        model_filepath = "{}/model.hdf5".format(self.__resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.__resource_tmp)
        with open(self.__train_data, "rb") as file:
            train_data = pickle.load(file)
        with open(self.__labels, "rb") as file:
            labels = pickle.load(file)
        _, _ = network.fit_and_get_history(
            train_data,
            labels,
            model_filepath,
            weights_filepath,
            self.__resource_tmp,
            "{}/training.log".format(self.__resource_tmp),
            False,
        )
        self.__hyperparameters.epochs = 3
        val_loss, val_acc = network.fit_and_get_history(
            train_data,
            labels,
            model_filepath,
            weights_filepath,
            self.__resource_tmp,
            "{}/training.log".format(self.__resource_tmp),
            True,
        )
        self.assertEqual(list, type(val_loss))
        self.assertEqual(list, type(val_acc))

    def test_exception_on_resume_with_no_extra_epochs(self):
        self.__hyperparameters.epochs = 2
        network = Undertest.get_network((2, 20), self.__hyperparameters)
        model_filepath = "{}/model.hdf5".format(self.__resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.__resource_tmp)
        with open(self.__train_data, "rb") as file:
            train_data = pickle.load(file)
        with open(self.__labels, "rb") as file:
            labels = pickle.load(file)
        _, _ = network.fit_and_get_history(
            train_data,
            labels,
            model_filepath,
            weights_filepath,
            self.__resource_tmp,
            "{}/training.log".format(self.__resource_tmp),
            False,
        )
        try:
            network.fit_and_get_history(
                train_data,
                labels,
                model_filepath,
                weights_filepath,
                self.__resource_tmp,
                "{}/training.log".format(self.__resource_tmp),
                True,
            )
        except Exception as e:
            self.assertTrue(isinstance(e, AssertionError))
            self.assertEqual("The existing model has been trained for 2 epochs. Make sure the total epochs are larger than 2", str(e))
        else:
            self.fail("Should have thrown exception")

    def test_exception_on_resume_with_no_previous_training_log(self):
        self.__hyperparameters.epochs = 2
        network = Undertest.get_network((2, 20), self.__hyperparameters)
        model_filepath = "{}/model.hdf5".format(self.__resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.__resource_tmp)
        with open(self.__train_data, "rb") as file:
            train_data = pickle.load(file)
        with open(self.__labels, "rb") as file:
            labels = pickle.load(file)
        _, _ = network.fit_and_get_history(
            train_data,
            labels,
            model_filepath,
            weights_filepath,
            self.__resource_tmp,
            "{}/training_1.log".format(self.__resource_tmp),
            False,
        )
        try:
            network.fit_and_get_history(
                train_data,
                labels,
                model_filepath,
                weights_filepath,
                self.__resource_tmp,
                "{}/training_2.log".format(self.__resource_tmp),
                True,
            )
        except Exception as e:
            self.assertTrue(isinstance(e, AssertionError))
            self.assertTrue("does not exist and is required by training resumption" in str(e))
        else:
            self.fail("Should have thrown exception")

    def test_fit_with_generator(self):
        self.__hyperparameters.epochs = 3
        network = Undertest.get_network((2, 20), self.__hyperparameters)
        model_filepath = "{}/model.hdf5".format(self.__resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.__resource_tmp)
        with h5py.File(self.__training_dump, "r") as hf:
            val_loss, val_acc = network.fit_with_generator(
                hf["train_data"],
                hf["labels"],
                model_filepath,
                weights_filepath,
                self.__resource_tmp,
                "training.log",
                False,
            )
            self.assertEqual(list, type(val_loss))
            self.assertEqual(list, type(val_acc))
            self.assertTrue(len(val_loss) == self.__hyperparameters.epochs)
            self.assertTrue(len(val_acc) == self.__hyperparameters.epochs)

    def test_early_stop_with_patience(self):
        self.__hyperparameters.epochs = 3
        self.__hyperparameters.es_patience = 0
        network = Undertest.get_network((2, 20), self.__hyperparameters)
        model_filepath = "{}/model.hdf5".format(self.__resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.__resource_tmp)
        with h5py.File(self.__training_dump, "r") as hf:
            val_loss, val_acc = network.fit_with_generator(
                hf["train_data"],
                hf["labels"],
                model_filepath,
                weights_filepath,
                self.__resource_tmp,
                "training.log",
                False,
            )
            self.assertEqual(list, type(val_loss))
            self.assertEqual(list, type(val_acc))
            self.assertTrue(len(val_loss) < self.__hyperparameters.epochs)
            self.assertTrue(len(val_acc) < self.__hyperparameters.epochs)

    def test_simple_fit(self):
        with open(self.__train_data, "rb") as file:
            train_data = pickle.load(file)
        with open(self.__labels, "rb") as file:
            labels = pickle.load(file)
        val_loss, val_acc = Undertest.simple_fit(
            (2, 20),
            train_data,
            labels,
            self.__hyperparameters,
        )
        self.assertEqual(list, type(val_loss))
        self.assertEqual(list, type(val_acc))

    def test_simple_fit_with_generator(self):
        self.__hyperparameters.epochs = 3
        with h5py.File(self.__training_dump, "r") as hf:
            val_loss, val_acc = Undertest.simple_fit_with_generator(
                (2, 20),
                hf["train_data"],
                hf["labels"],
                self.__hyperparameters,
            )
            self.assertEqual(list, type(val_loss))
            self.assertEqual(list, type(val_acc))
            self.assertTrue(len(val_loss) == self.__hyperparameters.epochs)
            self.assertTrue(len(val_acc) == self.__hyperparameters.epochs)


if __name__ == "__main__":
    unittest.main()
