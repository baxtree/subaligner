import unittest
import sys
from subaligner.hyperparameters import Hyperparameters as Undertest


class TestHyperparameters(unittest.TestCase):

    def test(self):
        unittest = Undertest()

        self.assertEqual(0.001, unittest.learning_rate)
        self.assertEqual([64], unittest.front_hidden_size)
        self.assertEqual([32, 16], unittest.back_hidden_size)
        self.assertEqual(0.2, unittest.dropout)
        self.assertEqual(100, unittest.epochs)
        self.assertIsNotNone(unittest.optimizer)
        self.assertEqual("binary_crossentropy", unittest.loss)
        self.assertEqual(["accuracy"], unittest.metrics)
        self.assertEqual(32, unittest.batch_size)
        self.assertEqual(0.25, unittest.validation_split)
        self.assertEqual("val_loss", unittest.monitor)
        self.assertEqual("min", unittest.es_mode)
        self.assertEqual(0.00001, unittest.es_min_delta)
        self.assertEqual(sys.maxsize, unittest.es_patience)


if __name__ == "__main__":
    unittest.main()
