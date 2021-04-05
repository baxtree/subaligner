import os
import unittest

from subaligner.hparam_tuner import HyperParameterTuner as Undertest
from mock import patch


class TestHparamTuner(unittest.TestCase):

    def setUp(self):
        self.video_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.mp4"
        )
        self.srt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.srt"
        )
        self.training_dump_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource"
        )

    def test_tune_hyper_paramerters(self):
        unittest = Undertest([self.video_file_path, self.video_file_path],
                             [self.srt_file_path, self.srt_file_path],
                             self.training_dump_dir, num_of_trials=3, tuning_epochs=2, n_mfcc=20)
        before = unittest.hyperparameters
        self.assertTrue(before == unittest.hyperparameters)

        unittest.tune_hyperparameters()

        self.assertFalse(before == unittest.hyperparameters)

    @patch("subaligner.trainer.Trainer.pre_train", return_value=([], []))
    def test_throw_exception_on_blank_losses(self, mock_pre_train):
        unittest = Undertest([self.video_file_path, self.video_file_path],
                             [self.srt_file_path, self.srt_file_path],
                             self.training_dump_dir, n_mfcc=20)
        self.assertRaises(ValueError, unittest.tune_hyperparameters)
        self.assertTrue(mock_pre_train.called)


if __name__ == "__main__":
    unittest.main()
