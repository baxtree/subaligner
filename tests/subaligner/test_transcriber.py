import os
import unittest
from subaligner.transcriber import Transcriber as Undertest
from subaligner.exception import TranscriptionException


class TranscriberTest(unittest.TestCase):

    def setUp(self) -> None:
        self.video_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource", "test.mp4")
        self.undertest = Undertest(recipe="whisper", flavour="tiny")

    def test_transcribe(self):
        subtitle, frame_rate = self.undertest.transcribe(self.video_file_path, "eng")
        assert len(subtitle.subs) > 0
        assert frame_rate == 24

    def test_throw_exception_on_unknown_recipe(self):
        try:
            Undertest(recipe="unknown")
        except Exception as e:
            self.assertTrue(isinstance(e, NotImplementedError))
            self.assertEqual(str(e), "Unknown recipe: unknown")
        else:
            self.fail("Should have thrown exception")

    def test_throw_exception_on_unknown_flavour(self):
        try:
            Undertest(recipe="whisper", flavour="unknown")
        except Exception as e:
            self.assertTrue(isinstance(e, NotImplementedError))
            self.assertEqual(str(e), "Unknown whisper flavour: unknown")
        else:
            self.fail("Should have thrown exception")

    def test_throw_exception_on_unsupported_language(self):
        try:
            self.undertest.transcribe(self.video_file_path, "abc")
        except Exception as e:
            self.assertTrue(isinstance(e, TranscriptionException))
            self.assertEqual(str(e), '"abc" is not supported by whisper (tiny)')
        else:
            self.fail("Should have thrown exception")
