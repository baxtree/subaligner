import unittest
import os
import pysrt
import subprocess

from subaligner.exception import TerminalException
from subaligner.media_helper import MediaHelper as Undertest
from mock import patch


class MediaHelperTests(unittest.TestCase):
    def setUp(self):
        self.__video_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.mp4"
        )
        self.__subtitle_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.srt"
        )
        self.__audio_file_path = None
        self.__segment_paths = []

    def tearDown(self):
        if self.__audio_file_path is not None:
            os.remove(self.__audio_file_path) if os.path.isfile(
                self.__audio_file_path
            ) else None

        if self.__segment_paths is not None:
            for segment_path in self.__segment_paths:
                os.remove(segment_path) if os.path.isfile(segment_path) else None

    def test_extract_audio_wav(self):
        self.__audio_file_path = Undertest.extract_audio(
            self.__video_file_path, True, 16000
        )
        self.assertTrue(os.path.isfile(self.__audio_file_path))

    def test_extract_audio_aac(self):
        self.__audio_file_path = Undertest.extract_audio(self.__video_file_path)
        self.assertTrue(os.path.isfile(self.__audio_file_path))

    def test_extract_audio_wav_from_start(self):
        self.__audio_file_path = Undertest.extract_audio(
            self.__video_file_path, True, 16000
        )
        segment_path, duration = Undertest.extract_audio_from_start_to_end(
            self.__audio_file_path, "00:00:13,750"
        )
        self.assertTrue(os.path.isfile(segment_path))
        self.__segment_paths.append(segment_path)
        self.assertIsNone(duration)

    def test_get_duration_in_seconds(self):
        duration = Undertest.get_duration_in_seconds(
            start="02:10:12,222", end="03:12:24,328"
        )
        self.assertEqual(3732.106, duration)

    def test_get_duration_in_seconds_without_start(self):
        duration = Undertest.get_duration_in_seconds(start=None, end="01:01:01,100")
        self.assertEqual(3661.100, duration)

    def test_extract_audio_wav_from_start_to_end(self):
        self.__audio_file_path = Undertest.extract_audio(
            self.__video_file_path, True, 16000
        )
        segment_path, duration = Undertest.extract_audio_from_start_to_end(
            self.__audio_file_path, "00:00:13,750", "00:00:16,150"
        )
        self.assertTrue(os.path.isfile(segment_path))
        self.__segment_paths.append(segment_path)
        self.assertEqual(2.4, duration)

    def test_get_audio_segment_starts_and_ends(self):
        subs = pysrt.open(self.__subtitle_file_path, encoding="utf-8")
        segment_starts, segment_ends, new_subs = Undertest.get_audio_segment_starts_and_ends(
            subs
        )
        self.assertEqual(len(segment_starts), len(segment_ends))
        self.assertEqual(len(segment_starts), len(new_subs))
        for sub in new_subs:
            self.assertIsInstance(sub, pysrt.SubRipFile)

    def test_throw_terminal_exception_on_bad_video(self):
        try:
            Undertest.extract_audio("bad_video_file_path", True, 16000)
        except Exception as e:
            self.assertTrue(isinstance(e, TerminalException))
            self.assertFalse(os.path.exists("bad_video_file_path.mp4.wav"))
        else:
            self.fail("Should have thrown exception")

    @patch("subprocess.Popen.communicate", return_value=1)
    def test_throw_exception_on_extract_audio_with_error_code(self, mock_communicate):
        try:
            Undertest.extract_audio(self.__video_file_path)
        except Exception as e:
            self.assertTrue(mock_communicate.called)
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")

    @patch("subprocess.Popen.communicate", side_effect=subprocess.TimeoutExpired(None, None))
    def test_throw_exception_on_extract_audio_timeout(self, mock_communicate):
        try:
            Undertest.extract_audio(self.__video_file_path)
        except Exception as e:
            self.assertTrue(mock_communicate.called)
            self.assertTrue(isinstance(e, subprocess.TimeoutExpired))
        else:
            self.fail("Should have thrown exception")

    @patch("subprocess.Popen.communicate", side_effect=Exception())
    def test_throw_exception_on_vtt2srt_exception(self, mock_communicate):
        try:
            Undertest.extract_audio(self.__video_file_path)
        except Exception as e:
            self.assertTrue(mock_communicate.called)
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")

    @patch("subprocess.Popen.communicate", return_value=1)
    def test_throw_exception_on_extract_partial_audio_with_error_code(self, mock_communicate):
        try:
            self.__audio_file_path = Undertest.extract_audio(
                self.__video_file_path, True, 16000
            )
            Undertest.extract_audio_from_start_to_end(
                self.__audio_file_path, "00:00:13,750", "00:00:16,150"
            )
        except Exception as e:
            self.assertTrue(mock_communicate.called)
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")

    @patch("subprocess.Popen.communicate", side_effect=subprocess.TimeoutExpired(None, None))
    def test_throw_exception_on_extract_partial_audio_timeout(self, mock_communicate):
        try:
            self.__audio_file_path = Undertest.extract_audio(
                self.__video_file_path, True, 16000
            )
            Undertest.extract_audio_from_start_to_end(
                self.__audio_file_path, "00:00:13,750", "00:00:16,150"
            )
        except Exception as e:
            self.assertTrue(mock_communicate.called)
            self.assertTrue(isinstance(e, subprocess.TimeoutExpired))
        else:
            self.fail("Should have thrown exception")

    @patch("subprocess.Popen.communicate", side_effect=Exception())
    def test_throw_exception_on_extract_partial_audio_exception(self, mock_communicate):
        try:
            self.__audio_file_path = Undertest.extract_audio(
                self.__video_file_path, True, 16000
            )
            Undertest.extract_audio_from_start_to_end(
                self.__audio_file_path, "00:00:13,750", "00:00:16,150"
            )
        except Exception as e:
            self.assertTrue(mock_communicate.called)
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")


if __name__ == "__main__":
    unittest.main()
