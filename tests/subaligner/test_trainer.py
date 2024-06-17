import unittest
import os
import shutil
from subaligner.embedder import FeatureEmbedder
from subaligner.hyperparameters import Hyperparameters
from subaligner.trainer import Trainer as Undertest
from subaligner.exception import TerminalException
from mock import patch


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.hyperparameters = Hyperparameters()
        self.hyperparameters.epochs = 1
        self.video_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.mp4"
        )
        self.audio_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.wav"
        )
        self.srt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.srt"
        )
        self.ttml_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.xml"
        )
        self.vtt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.vtt"
        )
        self.ass_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ass"
        )
        self.ssa_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ssa"
        )
        self.microdvd_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.sub"
        )
        self.mpl2_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.mpl2.txt"
        )
        self.tmp_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.tmp"
        )
        self.sami_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.smi"
        )
        self.stl_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.stl"
        )
        self.training_dump_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource"
        )
        self.resource_tmp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/tmp"
        )
        if os.path.exists(self.resource_tmp):
            shutil.rmtree(self.resource_tmp)
        os.mkdir(self.resource_tmp)
        self.model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/models/training/model"
        )
        self.training_log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/training.log"
        )

    def tearDown(self):
        shutil.rmtree(self.resource_tmp)

    def test_train_with_mixed_audio_and_video(self):
        Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).train(
            [self.video_file_path, self.audio_file_path],
            [self.srt_file_path, self.srt_file_path],
            model_dir=self.resource_tmp,
            weights_dir=self.resource_tmp,
            config_dir=self.resource_tmp,
            logs_dir=self.resource_tmp,
            training_dump_dir=self.resource_tmp,
            hyperparameters=self.hyperparameters,
        )
        output_files = os.listdir(self.resource_tmp)
        outputs = [file for file in output_files if file.endswith(".hdf5")]
        self.assertEqual(
            4, len(outputs)
        )  # one model file, one weights file and one combined file and one training dump
        hyperparams_files = [file for file in output_files if file.endswith(".json")]
        self.assertEqual(1, len(hyperparams_files))

    def test_train_with_mixed_subtitle_formats(self):
        Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).train(
            [self.video_file_path, self.video_file_path, self.video_file_path, self.video_file_path, self.video_file_path,
             self.video_file_path, self.video_file_path, self.video_file_path, self.video_file_path, self.video_file_path],
            [self.srt_file_path, self.ttml_file_path, self.vtt_file_path, self.ass_file_path, self.ssa_file_path,
             self.microdvd_file_path, self.mpl2_file_path, self.tmp_file_path, self.sami_file_path, self.stl_file_path],
            model_dir=self.resource_tmp,
            weights_dir=self.resource_tmp,
            config_dir=self.resource_tmp,
            logs_dir=self.resource_tmp,
            training_dump_dir=self.resource_tmp,
            hyperparameters=self.hyperparameters,
        )
        output_files = os.listdir(self.resource_tmp)
        model_files = [file for file in output_files if file.endswith(".hdf5")]
        self.assertEqual(
            4, len(model_files)
        )  # one model file, one weights file, one combined file and one training dump
        hyperparams_files = [file for file in output_files if file.endswith(".json")]
        self.assertEqual(1, len(hyperparams_files))

    def test_train_with_data_dump(self):
        Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).train(
            None,
            None,
            model_dir=self.resource_tmp,
            weights_dir=self.resource_tmp,
            config_dir=self.resource_tmp,
            logs_dir=self.resource_tmp,
            training_dump_dir=self.training_dump_dir,
            hyperparameters=self.hyperparameters,
        )
        output_files = os.listdir(self.resource_tmp)
        model_files = [file for file in output_files if file.endswith(".hdf5")]

        self.assertEqual(
            3, len(model_files)
        )  # one model file, one weights file and one combined file
        hyperparams_files = [file for file in output_files if file.endswith(".json")]
        self.assertEqual(1, len(hyperparams_files))

    def test_resume_training(self):
        underTest = Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05))
        underTest.train(
            [self.video_file_path, self.video_file_path],
            [self.srt_file_path, self.srt_file_path],
            model_dir=self.resource_tmp,
            weights_dir=self.resource_tmp,
            config_dir=self.resource_tmp,
            logs_dir=self.resource_tmp,
            training_dump_dir=self.resource_tmp,
            hyperparameters=self.hyperparameters,
        )

        # increase the maximum epoch
        hyperparams_file = "{}/hyperparameters.json".format(self.resource_tmp)
        hyperparameters = Hyperparameters.from_file(hyperparams_file)
        hyperparameters.epochs = 2
        hyperparameters.to_file(hyperparams_file)

        underTest.train(
            None,
            None,
            model_dir=self.resource_tmp,
            weights_dir=self.resource_tmp,
            config_dir=self.resource_tmp,
            logs_dir=self.resource_tmp,
            training_dump_dir=self.resource_tmp,
            hyperparameters=hyperparameters,
            resume=True,
        )
        output_files = os.listdir(self.resource_tmp)
        outputs = [file for file in output_files if file.endswith(".hdf5")]

        self.assertEqual(
            4, len(outputs)
        )  # one model file,,one model file, one combined file and one training dump file
        hyperparams_files = [file for file in output_files if file.endswith(".json")]
        self.assertEqual(1, len(hyperparams_files))

    def test_pre_train(self):
        val_loss, val_acc = Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).pre_train(
            [self.video_file_path, self.video_file_path],
            [self.srt_file_path, self.vtt_file_path],
            training_dump_dir=self.resource_tmp,
            hyperparameters=self.hyperparameters,
        )
        output_files = os.listdir(self.resource_tmp)
        outputs = [file for file in output_files if file.endswith(".hdf5")]
        self.assertEqual(1, len(outputs))  # one training dump file
        self.assertEqual(self.hyperparameters.epochs, len(val_loss))
        self.assertEqual(self.hyperparameters.epochs, len(val_acc))

    def test_pre_train_with_training_dump(self):
        val_loss, val_acc = Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).pre_train(
            None,
            None,
            training_dump_dir=self.training_dump_dir,
            hyperparameters=self.hyperparameters,
        )
        output_files = os.listdir(self.resource_tmp)
        outputs = [file for file in output_files if file.endswith(".hdf5")]
        self.assertEqual(0, len(outputs))
        self.assertEqual(self.hyperparameters.epochs, len(val_loss))
        self.assertEqual(self.hyperparameters.epochs, len(val_acc))

    def test_no_exception_caused_by_bad_media(self):
        not_a_video = self.srt_file_path
        Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).train(
            [self.video_file_path, not_a_video],
            [self.srt_file_path, self.srt_file_path],
            model_dir=self.resource_tmp,
            weights_dir=self.resource_tmp,
            config_dir=self.resource_tmp,
            logs_dir=self.resource_tmp,
            training_dump_dir=self.resource_tmp,
            hyperparameters=self.hyperparameters,
        )
        output_files = os.listdir(self.resource_tmp)
        outputs = [file for file in output_files if file.endswith(".hdf5")]
        self.assertEqual(
            4, len(outputs)
        )  # one model file, one weights file and one combined file and one training dump
        hyperparams_files = [file for file in output_files if file.endswith(".json")]
        self.assertEqual(1, len(hyperparams_files))

    def test_no_exception_caused_by_timeout(self):
        Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05), feature_embedding_timeout=0.01).train(
            [self.video_file_path],
            [self.srt_file_path],
            model_dir=self.resource_tmp,
            weights_dir=self.resource_tmp,
            config_dir=self.resource_tmp,
            logs_dir=self.resource_tmp,
            training_dump_dir=self.resource_tmp,
            hyperparameters=self.hyperparameters,
        )
        output_files = os.listdir(self.resource_tmp)
        outputs = [file for file in output_files if file.endswith(".hdf5")]
        self.assertEqual(
            4, len(outputs)
        )  # one model file, one weights file and one combined file and one training dump
        hyperparams_files = [file for file in output_files if file.endswith(".json")]
        self.assertEqual(1, len(hyperparams_files))

    def test_get_done_epochs(self):
        assert Undertest.get_done_epochs(self.training_log_path) == 1
        assert Undertest.get_done_epochs("not_exist_training.log") == 0

    @patch("concurrent.futures.wait", side_effect=KeyboardInterrupt)
    def test_throw_exception_on_training_interrupted(self, mock_wait):
        try:
            Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).train(
                [self.video_file_path, self.video_file_path],
                [self.srt_file_path, self.srt_file_path],
                model_dir=self.resource_tmp,
                weights_dir=self.resource_tmp,
                config_dir=self.resource_tmp,
                logs_dir=self.resource_tmp,
                training_dump_dir=self.resource_tmp,
                hyperparameters=self.hyperparameters,
            )
        except Exception as e:
            self.assertTrue(mock_wait.called)
            self.assertTrue(isinstance(e, TerminalException))
            self.assertTrue("interrupted" in str(e))
        else:
            self.fail("Should have thrown exception")


if __name__ == "__main__":
    unittest.main()
