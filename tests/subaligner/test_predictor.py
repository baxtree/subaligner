import unittest
import os
import sys

from mock import patch
from parameterized import parameterized
from subaligner.exception import TerminalException
from subaligner.logger import Logger
from subaligner.media_helper import MediaHelper
from subaligner.predictor import Predictor as Undertest


class PredictorTests(unittest.TestCase):
    def setUp(self):
        Undertest._Predictor__MAX_SHIFT_IN_SECS = sys.maxsize
        self.weights_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/models/training/weights"
        )
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
            os.path.dirname(os.path.abspath(__file__)), "resource/test_mpl2.txt"
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
        self.sbv_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.sbv"
        )
        self.ytt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ytt"
        )
        self.long_subtitle_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_too_long.srt"
        )
        self.plain_text_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_plain.txt"
        )
        self.audio_file_paths = []

    def tearDown(self):
        for file in self.audio_file_paths:
            os.remove(file) if os.path.isfile(file) else None

    def test_predict_single_pass_with_fps(self):
        subs, audio_file_path, voice_probabilities, frame_rate = Undertest(n_mfcc=20, step_sample=0.02).predict_single_pass(
            self.video_file_path, self.srt_file_path, self.weights_dir
        )
        self.audio_file_paths.append(audio_file_path)

        self.assertGreater(len(subs), 0)
        self.assertIsNotNone(audio_file_path)
        self.assertGreater(len(voice_probabilities), 0)
        self.assertEqual(24.0, frame_rate)

    @parameterized.expand([
        ["video_file_path", "srt_file_path"],
        ["video_file_path", "ttml_file_path"],
        ["video_file_path", "vtt_file_path"],
        ["video_file_path", "ass_file_path"],
        ["video_file_path", "ssa_file_path"],
        ["video_file_path", "microdvd_file_path"],
        ["video_file_path", "mpl2_file_path"],
        ["video_file_path", "tmp_file_path"],
        ["video_file_path", "sami_file_path"],
        ["video_file_path", "stl_file_path"],
        ["video_file_path", "sbv_file_path"],
        ["video_file_path", "ytt_file_path"],
    ])
    def test_predict_single_pass_on_video(self, media_file_path, subtitle_file_path):
        subs, audio_file_path, voice_probabilities, frame_rate = Undertest(n_mfcc=20).predict_single_pass(
            getattr(self, media_file_path), getattr(self, subtitle_file_path), self.weights_dir
        )
        self.audio_file_paths.append(audio_file_path)

        self.assertGreater(len(subs), 0)
        self.assertIsNotNone(audio_file_path)
        self.assertGreater(len(voice_probabilities), 0)
        self.assertEqual(24.0, frame_rate)

    @parameterized.expand([
        ["audio_file_path", "srt_file_path"],
        ["audio_file_path", "ttml_file_path"],
        ["audio_file_path", "vtt_file_path"],
        ["audio_file_path", "ass_file_path"],
        ["audio_file_path", "ssa_file_path"],
        ["audio_file_path", "microdvd_file_path"],
        ["audio_file_path", "mpl2_file_path"],
        ["audio_file_path", "tmp_file_path"],
        ["audio_file_path", "sami_file_path"],
        ["audio_file_path", "stl_file_path"],
        ["audio_file_path", "sbv_file_path"],
        ["audio_file_path", "ytt_file_path"],
    ])
    def test_predict_single_pass_on_audio(self, media_file_path, subtitle_file_path):
        subs, audio_file_path, voice_probabilities, frame_rate = Undertest(n_mfcc=20).predict_single_pass(
            getattr(self, media_file_path), getattr(self, subtitle_file_path), self.weights_dir
        )
        self.audio_file_paths.append(audio_file_path)

        self.assertGreater(len(subs), 0)
        self.assertIsNotNone(audio_file_path)
        self.assertGreater(len(voice_probabilities), 0)
        self.assertIsNone(frame_rate)

    def test_predict_on_subtitle_longer_than_audio_within_threshold(self):
        subs, audio_file_path, _, _ = Undertest(n_mfcc=20).predict_single_pass(
            self.video_file_path, self.long_subtitle_file_path, self.weights_dir
        )
        self.audio_file_paths.append(audio_file_path)

        self.assertGreater(len(subs), 0)
        self.assertIsNotNone(audio_file_path)

    def test_predict_on_subtitle_longer_than_audio_above_threshold(self):
        subs, audio_file_path, _, _ = Undertest(n_mfcc=20).predict_single_pass(
            self.video_file_path, self.long_subtitle_file_path, self.weights_dir
        )
        self.audio_file_paths.append(audio_file_path)

        self.assertGreater(len(subs), 0)
        self.assertIsNotNone(audio_file_path)

    @parameterized.expand([
        ["video_file_path", "srt_file_path"],
        ["video_file_path", "ttml_file_path"],
        ["video_file_path", "vtt_file_path"],
        ["video_file_path", "ass_file_path"],
        ["video_file_path", "ssa_file_path"],
        ["video_file_path", "microdvd_file_path"],
        ["video_file_path", "mpl2_file_path"],
        ["video_file_path", "tmp_file_path"],
        ["video_file_path", "sami_file_path"],
        ["video_file_path", "stl_file_path"],
        ["video_file_path", "sbv_file_path"],
        ["video_file_path", "ytt_file_path"],
    ])
    def test_predict_dual_pass_on_video(self, media_file_path, subtitle_file_path):
        undertest_obj = Undertest(n_mfcc=20)
        new_subs, subs, voice_probabilities, frame_rate = undertest_obj.predict_dual_pass(
            getattr(self, media_file_path), getattr(self, subtitle_file_path), self.weights_dir
        )

        self.assertGreater(len(new_subs), 0)
        self.assertEqual(len(new_subs), len(subs))
        self.assertGreater(len(voice_probabilities), 0)
        self.assertEqual(24.0, frame_rate)

    @parameterized.expand([
        ["audio_file_path", "srt_file_path"],
        ["audio_file_path", "ttml_file_path"],
        ["audio_file_path", "vtt_file_path"],
        ["audio_file_path", "ass_file_path"],
        ["audio_file_path", "ssa_file_path"],
        ["audio_file_path", "microdvd_file_path"],
        ["audio_file_path", "mpl2_file_path"],
        ["audio_file_path", "tmp_file_path"],
        ["audio_file_path", "sami_file_path"],
        ["audio_file_path", "stl_file_path"],
        ["audio_file_path", "sbv_file_path"],
        ["audio_file_path", "ytt_file_path"],
    ])
    def test_predict_dual_pass_on_video(self, media_file_path, subtitle_file_path):
        undertest_obj = Undertest(n_mfcc=20)
        new_subs, subs, voice_probabilities, frame_rate = undertest_obj.predict_dual_pass(
            getattr(self, media_file_path), getattr(self, subtitle_file_path), self.weights_dir
        )

        self.assertGreater(len(new_subs), 0)
        self.assertEqual(len(new_subs), len(subs))
        self.assertGreater(len(voice_probabilities), 0)
        self.assertIsNone(frame_rate)

    def test_predict_dual_pass_without_stretching_logs(self):
        quiet = Logger.QUIET
        Logger.QUIET = True
        undertest_obj = Undertest(n_mfcc=20)
        new_subs, subs, voice_probabilities, frame_rate = undertest_obj.predict_dual_pass(
            self.audio_file_path, self.srt_file_path, self.weights_dir
        )

        self.assertGreater(len(new_subs), 0)
        self.assertEqual(len(new_subs), len(subs))
        self.assertGreater(len(voice_probabilities), 0)
        self.assertIsNone(frame_rate)
        Logger.QUIET = quiet

    def test_predict_dual_pass_with_stretching(self):
        undertest_obj = Undertest(n_mfcc=20)

        new_subs, subs, voice_probabilities, frame_rate = undertest_obj.predict_dual_pass(
            self.video_file_path, self.srt_file_path, self.weights_dir, stretch=True
        )
        stretched = False
        for index, sub in enumerate(new_subs):
            if sub.duration != subs[index].duration:
                stretched = True
                break

        self.assertGreater(len(new_subs), 0)
        self.assertEqual(len(new_subs), len(subs))
        self.assertTrue(stretched)
        self.assertGreater(len(voice_probabilities), 0)
        self.assertEqual(24.0, frame_rate)

    def test_predict_dual_pass_with_specified_language(self):
        undertest_obj = Undertest(n_mfcc=20)

        new_subs, subs, voice_probabilities, frame_rate = undertest_obj.predict_dual_pass(
            self.video_file_path, self.srt_file_path, self.weights_dir, stretch=True, stretch_in_lang="zho"
        )
        stretched = False
        for index, sub in enumerate(new_subs):
            if sub.duration != subs[index].duration:
                stretched = True
                break

        self.assertGreater(len(new_subs), 0)
        self.assertEqual(len(new_subs), len(subs))
        self.assertTrue(stretched)
        self.assertGreater(len(voice_probabilities), 0)
        self.assertEqual(24.0, frame_rate)

    def test_predict_plain_text(self):
        subs, audio_file_path, voice_probabilities, frame_rate = Undertest(n_mfcc=20, step_sample=0.02).predict_plain_text(
            self.video_file_path, self.plain_text_file_path
        )

        self.assertGreater(len(subs), 0)
        self.assertIsNone(audio_file_path)
        self.assertIsNone(voice_probabilities)
        self.assertEqual(24.0, frame_rate)

    def test_get_log_loss(self):
        undertest_obj = Undertest(n_mfcc=20)
        subs, audio_file_path, voice_probabilities, frame_rate = undertest_obj.predict_single_pass(
            self.video_file_path, self.srt_file_path, self.weights_dir
        )
        log_loss = undertest_obj.get_log_loss(voice_probabilities, subs)
        self.assertGreater(log_loss, 0)
        self.assertEqual(24.0, frame_rate)

    def test_get_log_loss_on_speech_shorter_than_subtitle(self):
        undertest_obj = Undertest(n_mfcc=20)
        shorter_audio_file_path, _ = MediaHelper().extract_audio_from_start_to_end(self.audio_file_path, "00:00:00,000", "00:00:32,797")
        self.audio_file_paths.append(shorter_audio_file_path)
        subs, audio_file_path, voice_probabilities, frame_rate = undertest_obj.predict_single_pass(
            shorter_audio_file_path, self.srt_file_path, self.weights_dir
        )
        log_loss = undertest_obj.get_log_loss(voice_probabilities, subs)
        self.assertGreater(log_loss, 0)
        self.assertIsNone(frame_rate)

    def test_get_min_log_loss_and_index(self):
        undertest_obj = Undertest(n_mfcc=20)
        subs, audio_file_path, voice_probabilities, frame_rate = undertest_obj.predict_single_pass(
            self.video_file_path, self.srt_file_path, self.weights_dir
        )
        min_log_loss, min_log_loss_pos = undertest_obj.get_min_log_loss_and_index(
            voice_probabilities, subs
        )
        self.assertGreater(min_log_loss, 0)
        self.assertGreaterEqual(min_log_loss_pos, 0)
        self.assertEqual(24.0, frame_rate)

    def test_throw_terminal_exception_on_missing_video(self):
        try:
            subs, audio_file_path, _, _ = Undertest(n_mfcc=20).predict_single_pass(None, self.srt_file_path, self.weights_dir)
        except Exception as e:
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")

    @unittest.skip("Mocking does not work for spawned processes")
    @patch("subaligner.media_helper.MediaHelper.extract_audio_from_start_to_end", side_effect=Exception("exception"))
    def test_not_throw_exception_on_segment_alignment_failure(self, mock_time_to_sec):
        undertest_obj = Undertest(n_mfcc=20)
        new_subs, subs, voice_probabilities, frame_rate = undertest_obj.predict_dual_pass(
            self.video_file_path, self.srt_file_path, self.weights_dir
        )
        self.assertGreater(len(new_subs), 0)
        self.assertEqual(len(new_subs), len(subs))
        self.assertGreater(len(voice_probabilities), 0)
        self.assertTrue(mock_time_to_sec.called)
        self.assertEqual(24.0, frame_rate)

    @unittest.skip("Mocking does not work for spawned processes")
    @patch("subaligner.media_helper.MediaHelper.extract_audio_from_start_to_end", side_effect=Exception("exception"))
    def test_throw_exception_on_segment_alignment_failure_when_flag_on(self, mock_time_to_sec):
        try:
            undertest_obj = Undertest(n_mfcc=20)
            undertest_obj.predict_dual_pass(self.video_file_path, self.srt_file_path, self.weights_dir, exit_segfail=True)
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
                self.video_file_path, self.srt_file_path, self.weights_dir
            )
        except Exception as e:
            self.assertTrue(mock_result.called)
            self.assertTrue(isinstance(e, TerminalException))
            self.assertTrue("interrupted" in str(e))
        else:
            self.fail("Should have thrown exception")

    def test_throw_terminal_exception_on_missing_subtitle(self):
        try:
            subs, audio_file_path, _, _ = Undertest(n_mfcc=20).predict_single_pass(self.video_file_path, None, self.weights_dir)
            self.fail("Should not have reached here")
        except Exception as e:
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")

    def test_throw_terminal_exception_on_timeout(self):
        try:
            undertest_obj = Undertest(segment_alignment_timeout=0, n_mfcc=20)
            undertest_obj.predict_dual_pass(self.video_file_path, self.srt_file_path, self.weights_dir)
            self.fail("Should not have reached here")
        except Exception as e:
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")


if __name__ == "__main__":
    unittest.main()
