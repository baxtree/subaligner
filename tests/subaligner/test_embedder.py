import unittest
import pysrt
import os
from subaligner.embedder import FeatureEmbedder as Undertest
from subaligner.exception import TerminalException


class FeatureEmbedderTests(unittest.TestCase):
    def setUp(self):
        self.__subtitle_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.srt"
        )
        self.__audio_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.wav"
        )

    def test_get_len_mfcc(self):
        self.assertEqual(2.34375, Undertest(n_mfcc=20, step_sample=0.05).get_len_mfcc())

    def test_get_step_mfcc(self):
        self.assertEqual(1.5625, Undertest(n_mfcc=20, step_sample=0.05).get_step_mfcc())

    def test_time_to_sec(self):
        subs = pysrt.open(self.__subtitle_file_path, encoding="utf-8")
        self.assertEqual(12.44, Undertest.time_to_sec(subs[0].start))
        self.assertEqual(14.955, Undertest.time_to_sec(subs[0].end))

    def test_time_to_pos(self):
        subs = pysrt.open(self.__subtitle_file_path, encoding="utf-8")
        self.assertEqual(
            248, Undertest(n_mfcc=20, step_sample=0.05).time_to_pos(subs[0].start)
        )
        self.assertEqual(
            299, Undertest(n_mfcc=20, step_sample=0.05).time_to_pos(subs[0].end)
        )

    def test_sec_to_pos(self):
        self.assertEqual(275, Undertest(n_mfcc=20, step_sample=0.05).sec_to_pos(13.75))
        self.assertEqual(322, Undertest(n_mfcc=20, step_sample=0.05).sec_to_pos(16.15))

    def test_pos_to_sec(self):
        self.assertEqual(13.75, Undertest(n_mfcc=20, step_sample=0.05).pos_to_sec(275))
        self.assertEqual(16.1, Undertest(n_mfcc=20, step_sample=0.05).pos_to_sec(322))

    def test_pos_to_time_str(self):
        self.assertEqual(
            "01:23:20,150",
            Undertest(n_mfcc=20, step_sample=0.05).pos_to_time_str(100003),
        )

    def test_extract_data_and_label_from_audio(self):
        train_data, labels = Undertest(
            n_mfcc=20, step_sample=0.05
        ).extract_data_and_label_from_audio(
            self.__audio_file_path, self.__subtitle_file_path
        )
        self.assertEqual(len(train_data), len(labels))
        for i in labels:
            if i != 1 and i != 0:
                self.fail("label must be either one or zero")

    def test_throw_terminal_exception(self):
        try:
            Undertest(n_mfcc=20, step_sample=0.05).extract_data_and_label_from_audio(
                self.__audio_file_path, None
            )
        except Exception as e:
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")


if __name__ == "__main__":
    unittest.main()
