import unittest
import os
import shutil
from subaligner.embedder import FeatureEmbedder
from subaligner.hyperparameters import Hyperparameters
from subaligner.trainer import Trainer as Undertest


class TrainerTests(unittest.TestCase):
    def setUp(self):
        self.__hyperparameters = Hyperparameters()
        self.__hyperparameters.epochs = 1
        self.__video_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.mp4"
        )
        self.__audio_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.wav"
        )
        self.__srt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.srt"
        )
        self.__ttml_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.xml"
        )
        self.__vtt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.vtt"
        )
        self.__training_dump_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource"
        )
        self.__resource_tmp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/tmp"
        )
        if os.path.exists(self.__resource_tmp):
            shutil.rmtree(self.__resource_tmp)
        os.mkdir(self.__resource_tmp)
        self.__model_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/models/training/model"
        )

    def tearDown(self):
        shutil.rmtree(self.__resource_tmp)

    def test_train(self):
        Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).train(
            [self.__video_file_path, self.__video_file_path],
            [self.__srt_file_path, self.__srt_file_path],
            model_dir=self.__resource_tmp,
            weights_dir=self.__resource_tmp,
            config_dir=self.__resource_tmp,
            logs_dir=self.__resource_tmp,
            training_dump_dir=self.__resource_tmp,
            hyperparameters=self.__hyperparameters,
        )
        output_files = os.listdir(self.__resource_tmp)
        model_files = [file for file in output_files if file.endswith(".hdf5")]
        self.assertEqual(
            4, len(model_files)
        )  # one model file, one weights file, one combined file and one training dump
        hyperparams_files = [file for file in output_files if file.endswith(".json")]
        self.assertEqual(1, len(hyperparams_files))

    def test_train_with_ttml(self):
        Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).train(
            [self.__video_file_path, self.__video_file_path],
            [self.__ttml_file_path, self.__ttml_file_path],
            model_dir=self.__resource_tmp,
            weights_dir=self.__resource_tmp,
            config_dir=self.__resource_tmp,
            logs_dir=self.__resource_tmp,
            training_dump_dir=self.__resource_tmp,
            hyperparameters=self.__hyperparameters,
        )
        output_files = os.listdir(self.__resource_tmp)
        model_files = [file for file in output_files if file.endswith(".hdf5")]
        self.assertEqual(
            4, len(model_files)
        )  # one model file, one weights file, one combined file and one training dump
        hyperparams_files = [file for file in output_files if file.endswith(".json")]
        self.assertEqual(1, len(hyperparams_files))

    def test_train_with_vtt(self):
        Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).train(
            [self.__video_file_path, self.__video_file_path],
            [self.__vtt_file_path, self.__vtt_file_path],
            model_dir=self.__resource_tmp,
            weights_dir=self.__resource_tmp,
            config_dir=self.__resource_tmp,
            logs_dir=self.__resource_tmp,
            training_dump_dir=self.__resource_tmp,
            hyperparameters=self.__hyperparameters,
        )
        output_files = os.listdir(self.__resource_tmp)
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
            model_dir=self.__resource_tmp,
            weights_dir=self.__resource_tmp,
            config_dir=self.__resource_tmp,
            logs_dir=self.__resource_tmp,
            training_dump_dir=self.__training_dump_dir,
            hyperparameters=self.__hyperparameters,
        )
        output_files = os.listdir(self.__resource_tmp)
        model_files = [file for file in output_files if file.endswith(".hdf5")]

        self.assertEqual(
            3, len(model_files)
        )  # one model file, one weights file and one combined file
        hyperparams_files = [file for file in output_files if file.endswith(".json")]
        self.assertEqual(1, len(hyperparams_files))

    def test_resume_training(self):
        underTest = Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05))
        underTest.train(
            [self.__video_file_path, self.__video_file_path],
            [self.__srt_file_path, self.__srt_file_path],
            model_dir=self.__resource_tmp,
            weights_dir=self.__resource_tmp,
            config_dir=self.__resource_tmp,
            logs_dir=self.__resource_tmp,
            training_dump_dir=self.__resource_tmp,
            hyperparameters=self.__hyperparameters,
        )

        # increase the maximum epoch
        hyperparams_file = "{}/hyperparameters.json".format(self.__resource_tmp)
        hyperparameters = Hyperparameters.from_file(hyperparams_file)
        hyperparameters.epochs = 2
        hyperparameters.to_file(hyperparams_file)

        underTest.train(
            None,
            None,
            model_dir=self.__resource_tmp,
            weights_dir=self.__resource_tmp,
            config_dir=self.__resource_tmp,
            logs_dir=self.__resource_tmp,
            training_dump_dir=self.__resource_tmp,
            hyperparameters=None,
            resume=True,
        )
        output_files = os.listdir(self.__resource_tmp)
        outputs = [file for file in output_files if file.endswith(".hdf5")]

        self.assertEqual(
            4, len(outputs)
        )  # one model file,,one model file, one combined file and one training dump file
        hyperparams_files = [file for file in output_files if file.endswith(".json")]
        self.assertEqual(1, len(hyperparams_files))

    def test_pre_train(self):
        val_loss, val_acc = Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).pre_train(
            [self.__video_file_path, self.__video_file_path],
            [self.__srt_file_path, self.__vtt_file_path],
            training_dump_dir=self.__resource_tmp,
            hyperparameters=self.__hyperparameters,
        )
        output_files = os.listdir(self.__resource_tmp)
        outputs = [file for file in output_files if file.endswith(".hdf5")]
        self.assertEqual(1, len(outputs))  # one training dump file
        self.assertEqual(self.__hyperparameters.epochs, len(val_loss))
        self.assertEqual(self.__hyperparameters.epochs, len(val_acc))

    def test_pre_train_with_training_dump(self):
        val_loss, val_acc = Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).pre_train(
            None,
            None,
            training_dump_dir=self.__training_dump_dir,
            hyperparameters=self.__hyperparameters,
        )
        output_files = os.listdir(self.__resource_tmp)
        outputs = [file for file in output_files if file.endswith(".hdf5")]
        self.assertEqual(0, len(outputs))
        self.assertEqual(self.__hyperparameters.epochs, len(val_loss))
        self.assertEqual(self.__hyperparameters.epochs, len(val_acc))

    def test_train_with_mixed_audio_and_video(self):
        Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).train(
            [self.__video_file_path, self.__audio_file_path],
            [self.__srt_file_path, self.__srt_file_path],
            model_dir=self.__resource_tmp,
            weights_dir=self.__resource_tmp,
            config_dir=self.__resource_tmp,
            logs_dir=self.__resource_tmp,
            training_dump_dir=self.__resource_tmp,
            hyperparameters=self.__hyperparameters,
        )
        output_files = os.listdir(self.__resource_tmp)
        outputs = [file for file in output_files if file.endswith(".hdf5")]
        self.assertEqual(
            4, len(outputs)
        )  # one model file, one weights file and one combined file and one training dump
        hyperparams_files = [file for file in output_files if file.endswith(".json")]
        self.assertEqual(1, len(hyperparams_files))

    def test_train_with_mixed_subtitle_formats(self):
        Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).train(
            [self.__video_file_path, self.__video_file_path],
            [self.__srt_file_path, self.__vtt_file_path],
            model_dir=self.__resource_tmp,
            weights_dir=self.__resource_tmp,
            config_dir=self.__resource_tmp,
            logs_dir=self.__resource_tmp,
            training_dump_dir=self.__resource_tmp,
            hyperparameters=self.__hyperparameters,
        )
        output_files = os.listdir(self.__resource_tmp)
        outputs = [file for file in output_files if file.endswith(".hdf5")]
        self.assertEqual(
            4, len(outputs)
        )  # one model file, one weights file and one combined file and one training dump
        hyperparams_files = [file for file in output_files if file.endswith(".json")]
        self.assertEqual(1, len(hyperparams_files))

    def test_no_exception_caused_by_bad_media(self):
        not_a_video = self.__srt_file_path
        Undertest(FeatureEmbedder(n_mfcc=20, step_sample=0.05)).train(
            [self.__video_file_path, not_a_video],
            [self.__srt_file_path, self.__srt_file_path],
            model_dir=self.__resource_tmp,
            weights_dir=self.__resource_tmp,
            config_dir=self.__resource_tmp,
            logs_dir=self.__resource_tmp,
            training_dump_dir=self.__resource_tmp,
            hyperparameters=self.__hyperparameters,
        )
        output_files = os.listdir(self.__resource_tmp)
        outputs = [file for file in output_files if file.endswith(".hdf5")]
        self.assertEqual(
            4, len(outputs)
        )  # one model file, one weights file and one combined file and one training dump
        hyperparams_files = [file for file in output_files if file.endswith(".json")]
        self.assertEqual(1, len(hyperparams_files))

    def test_convert_to_pb_model(self):
        pass

    def test_create_saved_model(self):
        pass


if __name__ == "__main__":
    unittest.main()
