import unittest
import os
import sys

from concurrent.futures._base import Future
from mock import patch
from subaligner.exception import TerminalException
from subaligner.predictor import Predictor as Undertest


class PredictorTests(unittest.TestCase):
    def setUp(self):
        Undertest._Predictor__MAX_SHIFT_IN_SECS = sys.maxsize
        self.__weights_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/models/training/weights"
        )
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
        self.__long_subtitle_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_too_long.srt"
        )
        self.__audio_file_paths = []

    def tearDown(self):
        for file in self.__audio_file_paths:
            os.remove(file) if os.path.isfile(file) else None

    def test_predict_single_pass_on_video(self):
        subs, audio_file_path, voice_probabilities, frame_rate = Undertest(n_mfcc=20).predict_single_pass(
            self.__video_file_path, self.__srt_file_path, self.__weights_dir
        )
        self.__audio_file_paths.append(audio_file_path)

        self.assertGreater(len(subs), 0)
        self.assertIsNotNone(audio_file_path)
        self.assertGreater(len(voice_probabilities), 0)
        self.assertEqual(24.0, frame_rate)

    def test_predict_single_pass_on_audio(self):
        subs, audio_file_path, voice_probabilities, frame_rate = Undertest(n_mfcc=20).predict_single_pass(
            self.__audio_file_path, self.__srt_file_path, self.__weights_dir
        )
        self.__audio_file_paths.append(audio_file_path)

        self.assertGreater(len(subs), 0)
        self.assertIsNotNone(audio_file_path)
        self.assertGreater(len(voice_probabilities), 0)
        self.assertIsNone(frame_rate)

    def test_predict_single_pass_with_fps(self):
        subs, audio_file_path, voice_probabilities, frame_rate = Undertest(n_mfcc=20, step_sample=0.02).predict_single_pass(
            self.__video_file_path, self.__srt_file_path, self.__weights_dir
        )
        self.__audio_file_paths.append(audio_file_path)

        self.assertGreater(len(subs), 0)
        self.assertIsNotNone(audio_file_path)
        self.assertGreater(len(voice_probabilities), 0)
        self.assertEqual(24.0, frame_rate)

    def test_predict_single_pass_with_ttml(self):
        subs, audio_file_path, voice_probabilities, frame_rate = Undertest(n_mfcc=20).predict_single_pass(
            self.__video_file_path, self.__ttml_file_path, self.__weights_dir
        )
        self.__audio_file_paths.append(audio_file_path)

        self.assertGreater(len(subs), 0)
        self.assertIsNotNone(audio_file_path)
        self.assertGreater(len(voice_probabilities), 0)
        self.assertEqual(24.0, frame_rate)

    def test_predict_single_pass_with_vtt(self):
        subs, audio_file_path, voice_probabilities, frame_rate = Undertest(n_mfcc=20).predict_single_pass(
            self.__video_file_path, self.__vtt_file_path, self.__weights_dir
        )
        self.__audio_file_paths.append(audio_file_path)

        self.assertGreater(len(subs), 0)
        self.assertIsNotNone(audio_file_path)
        self.assertGreater(len(voice_probabilities), 0)
        self.assertEqual(24.0, frame_rate)

    def test_predict_on_subtitle_longer_than_audio_within_threshold(self):
        subs, audio_file_path, _, _ = Undertest(n_mfcc=20).predict_single_pass(
            self.__video_file_path, self.__long_subtitle_file_path, self.__weights_dir
        )
        self.__audio_file_paths.append(audio_file_path)

        self.assertGreater(len(subs), 0)
        self.assertIsNotNone(audio_file_path)

    def test_predict_on_subtitle_longer_than_audio_above_threshold(self):
        subs, audio_file_path, _, _ = Undertest(n_mfcc=20).predict_single_pass(
            self.__video_file_path, self.__long_subtitle_file_path, self.__weights_dir
        )
        self.__audio_file_paths.append(audio_file_path)

        self.assertGreater(len(subs), 0)
        self.assertIsNotNone(audio_file_path)

    def test_predict_dual_pass_on_video(self):
        undertest_obj = Undertest(n_mfcc=20)
        new_subs, subs, voice_probabilities, frame_rate = undertest_obj.predict_dual_pass(
            self.__video_file_path, self.__srt_file_path, self.__weights_dir
        )

        self.assertGreater(len(new_subs), 0)
        self.assertEqual(len(new_subs), len(subs))
        self.assertGreater(len(voice_probabilities), 0)
        self.assertEqual(24.0, frame_rate)

    def test_predict_dual_pass_on_audio(self):
        undertest_obj = Undertest(n_mfcc=20)
        new_subs, subs, voice_probabilities, frame_rate = undertest_obj.predict_dual_pass(
            self.__audio_file_path, self.__srt_file_path, self.__weights_dir
        )

        self.assertGreater(len(new_subs), 0)
        self.assertEqual(len(new_subs), len(subs))
        self.assertGreater(len(voice_probabilities), 0)
        self.assertIsNone(frame_rate)

    def test_predict_dual_pass_with_stretching(self):
        undertest_obj = Undertest(n_mfcc=20)

        new_subs, subs, voice_probabilities, frame_rate = undertest_obj.predict_dual_pass(
            self.__video_file_path, self.__srt_file_path, self.__weights_dir, stretch=True
        )
        stretched = False
        for index, sub in enumerate(new_subs):
            if (sub.duration != subs[index].duration):
                stretched = True
                break

        self.assertGreater(len(new_subs), 0)
        self.assertEqual(len(new_subs), len(subs))
        self.assertTrue(stretched)
        self.assertGreater(len(voice_probabilities), 0)
        self.assertEqual(24.0, frame_rate)

    def test_get_log_loss(self):
        undertest_obj = Undertest(n_mfcc=20)
        subs, audio_file_path, voice_probabilities, frame_rate = undertest_obj.predict_single_pass(
            self.__video_file_path, self.__srt_file_path, self.__weights_dir
        )
        log_loss = undertest_obj.get_log_loss(voice_probabilities, subs)
        self.assertGreater(log_loss, 0)
        self.assertEqual(24.0, frame_rate)

    def test_get_min_log_loss_and_index(self):
        undertest_obj = Undertest(n_mfcc=20)
        subs, audio_file_path, voice_probabilities, frame_rate = undertest_obj.predict_single_pass(
            self.__video_file_path, self.__srt_file_path, self.__weights_dir
        )
        min_log_loss, min_log_loss_pos = undertest_obj.get_min_log_loss_and_index(
            voice_probabilities, subs
        )
        self.assertGreater(min_log_loss, 0)
        self.assertGreaterEqual(min_log_loss_pos, 0)
        self.assertEqual(24.0, frame_rate)

    def test_housekeeping_on_exceptions(self):
        pass

    def test_throw_terminal_exception_on_unmatched_audio_subtitle_durations(self):
        pass

    def test_throw_terminal_exception_on_missing_video(self):
        try:
            subs, audio_file_path, _, _ = Undertest(n_mfcc=20).predict_single_pass(None, self.__srt_file_path, self.__weights_dir)
        except Exception as e:
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")

    @patch("subaligner.media_helper.MediaHelper.extract_audio_from_start_to_end", side_effect=Exception("exception"))
    def test_not_throw_exception_on_segment_alignment_failure(self, mock_time_to_sec):
        undertest_obj = Undertest(n_mfcc=20)
        new_subs, subs, voice_probabilities, frame_rate = undertest_obj.predict_dual_pass(
            self.__video_file_path, self.__srt_file_path, self.__weights_dir
        )
        self.assertGreater(len(new_subs), 0)
        self.assertEqual(len(new_subs), len(subs))
        self.assertGreater(len(voice_probabilities), 0)
        self.assertTrue(mock_time_to_sec.called)
        self.assertEqual(24.0, frame_rate)

    @patch("subaligner.media_helper.MediaHelper.extract_audio_from_start_to_end", side_effect=Exception("exception"))
    def test_throw_exception_on_segment_alignment_failure_when_flag_on(self, mock_time_to_sec):
        try:
            undertest_obj = Undertest(n_mfcc=20)
            undertest_obj.predict_dual_pass(self.__video_file_path, self.__srt_file_path, self.__weights_dir, exit_segfail=True)
        except Exception as e:
            self.assertTrue(mock_time_to_sec.called)
            self.assertTrue(isinstance(e, TerminalException))
            self.assertTrue("At least one of the segments failed on alignment. Exiting..." in str(e))
        else:
            self.fail("Should have thrown exception")

    @patch("concurrent.futures._base.Future.result", side_effect=KeyboardInterrupt)
    def test_throw_exception_on_predict_interrupted(self, mock_result):
        try:
            undertest_obj = Undertest(n_mfcc=20)
            undertest_obj.predict_dual_pass(
                self.__video_file_path, self.__srt_file_path, self.__weights_dir
            )
        except Exception as e:
            self.assertTrue(mock_result.called)
            self.assertTrue(isinstance(e, TerminalException))
            self.assertTrue("interrupted" in str(e))
        else:
            self.fail("Should have thrown exception")

    def test_throw_terminal_exception_on_missing_subtitle(self):
        try:
            subs, audio_file_path, _, _ = Undertest(n_mfcc=20).predict_single_pass(self.__video_file_path, None, self.__weights_dir)
            self.fail("Should not have reached here")
        except Exception as e:
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")

    def test_throw_terminal_exception_on_timeout(self):
        backup = Undertest._Predictor__SEGMENT_PREDICTION_TIMEOUT
        Undertest._Predictor__SEGMENT_PREDICTION_TIMEOUT = 0.05
        try:
            undertest_obj = Undertest(n_mfcc=20)
            undertest_obj.predict_dual_pass(self.__video_file_path, self.__srt_file_path, self.__weights_dir)
        except Exception as e:
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")
        finally:
            Undertest._Predictor__SEGMENT_PREDICTION_TIMEOUT = backup


if __name__ == "__main__":
    unittest.main()
