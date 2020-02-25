import unittest
import os
import pickle
import shutil
import h5py
import numpy as np

from subaligner.network import Network as Undertest


class NetworkTests(unittest.TestCase):
    def setUp(self):
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
        Undertest.clear_session()

    def test_suppressed_init(self):
        self.assertRaises(
            AssertionError, Undertest, "guess", (2, 20), "lstm", None, None, None, None
        )

    def test_get_from_model(self):
        model_filepath = "{}/{}".format(self.__model_dir, "model.hdf5")
        network = Undertest.get_from_model(model_filepath)
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
        self.assertEqual((2, 20), Undertest.get_from_model(model_filepath).input_shape)

    def test_input_shape(self):
        network = Undertest.get_lstm((2, 20), [12], [56, 28])
        self.assertEqual((2, 20), network.input_shape)

    def test_create_lstm_network(self):
        network = Undertest.get_lstm((2, 20), [12], [56, 28])
        self.assertEqual("lstm", network.n_type)

    def test_create_bi_lstm_network(self):
        network = Undertest.get_bi_lstm((2, 20), [12], [56, 28])
        self.assertEqual("bi_lstm", network.n_type)

    def test_summary(self):
        network = Undertest.get_lstm((2, 20), [12], [56, 28])
        self.assertTrue(network.summary is None)  # Why this is None

    def test_get_predictions(self):
        network = Undertest.get_from_model("{}/model.hdf5".format(self.__model_dir))
        with open(self.__train_data, "rb") as file:
            train_data = pickle.load(file)
        self.assertEqual((11431, 1), network.get_predictions(train_data, "{}/weights.hdf5".format(self.__weights_dir)).shape)

    def test_load_model_and_weights(self):
        network_old = Undertest.get_lstm((2, 20), [12], [56, 28])
        weights_old = network_old._Network__model.get_weights()
        network_new = Undertest.load_model_and_weights("{}/model.hdf5".format(self.__model_dir), "{}/weights.hdf5".format(self.__weights_dir))
        weights_new = network_new._Network__model.get_weights()
        self.assertFalse(np.array_equal(weights_old, weights_new))

    def test_fit_lstm_and_get_history(self):
        network = Undertest.get_lstm((2, 20), [12], [56, 28])
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
            1,
            "training.log",
            False,
        )
        self.assertEqual(list, type(val_loss))
        self.assertEqual(list, type(val_acc))

    def test_fit_bi_lstm_and_get_history(self):
        network = Undertest.get_bi_lstm((2, 20), [12], [56, 28])
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
            1,
            "training.log",
            False,
        )
        self.assertEqual(list, type(val_loss))
        self.assertEqual(list, type(val_acc))

    def test_resume_and_get_history(self):
        network = Undertest.get_lstm((2, 20), [12], [56, 28])
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
            2,
            "{}/training.log".format(self.__resource_tmp),
            False,
        )
        val_loss, val_acc = network.fit_and_get_history(
            train_data,
            labels,
            model_filepath,
            weights_filepath,
            self.__resource_tmp,
            3,
            "{}/training.log".format(self.__resource_tmp),
            True,
        )
        self.assertEqual(list, type(val_loss))
        self.assertEqual(list, type(val_acc))

    def test_exception_on_resume_with_no_extra_epochs(self):
        network = Undertest.get_lstm((2, 20), [12], [56, 28])
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
            2,
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
                2,
                "{}/training.log".format(self.__resource_tmp),
                True,
            )
        except Exception as e:
            self.assertTrue(isinstance(e, AssertionError))
            self.assertEquals("Existing model has been trained for 2 epochs", str(e))
        else:
            self.fail("Should have thrown exception")

    def test_exception_on_resume_with_no_previous_training_log(self):
        network = Undertest.get_lstm((2, 20), [12], [56, 28])
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
            2,
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
                2,
                "{}/training_2.log".format(self.__resource_tmp),
                True,
            )
        except Exception as e:
            self.assertTrue(isinstance(e, AssertionError))
            self.assertTrue("does not exist and is required by training resumption" in str(e))
        else:
            self.fail("Should have thrown exception")

    def test_fit_with_generator(self):
        model_filepath = "{}/model.hdf5".format(self.__resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.__resource_tmp)
        network = Undertest.get_lstm((2, 20), [12], [56, 28])
        with open(self.__train_data, "rb") as file:
            train_data = pickle.load(file)
        with open(self.__labels, "rb") as file:
            labels = pickle.load(file)

        def generator():
            yield train_data, labels

        val_loss, val_acc = network.fit_with_generator(
            generator(),
            1,
            model_filepath,
            weights_filepath,
            self.__resource_tmp,
            1,
            "training.log",
            False,
        )
        self.assertEqual(list, type(val_loss))
        self.assertEqual(list, type(val_acc))

    def test_fit_with_generator(self):
        network = Undertest.get_lstm((2, 20), [12], [56, 28])
        model_filepath = "{}/model.hdf5".format(self.__resource_tmp)
        weights_filepath = "{}/weights.hdf5".format(self.__resource_tmp)
        with h5py.File(self.__training_dump, "r") as hf:
            val_loss, val_acc = network.fit_with_generator(
                hf["train_data"],
                hf["labels"],
                model_filepath,
                weights_filepath,
                self.__resource_tmp,
                2,
                "training.log",
                False,
            )
            self.assertEqual(list, type(val_loss))
            self.assertEqual(list, type(val_acc))


if __name__ == "__main__":
    unittest.main()
