import os
import unittest
import shutil
import filecmp
import subprocess
from subaligner.utils import Utils as Undertest
from subaligner.exception import TerminalException
from mock import patch


class UtilsTests(unittest.TestCase):
    def setUp(self):
        self.__mocked_srt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_utils.srt"
        )
        self.__mocked_ttml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_utils.xml"
        )
        self.__real_srt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.srt"
        )
        self.__real_vtt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.vtt"
        )
        self.__resource_tmp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/tmp"
        )
        if os.path.exists(self.__resource_tmp):
            shutil.rmtree(self.__resource_tmp)
        os.mkdir(self.__resource_tmp)

    def tearDown(self):
        shutil.rmtree(self.__resource_tmp)

    @patch("pycaption.CaptionConverter.write", side_effect=lambda writer: "output")
    @patch("pycaption.CaptionConverter.read")
    def test_srt2ttml(self, mock_read, mock_write):
        Undertest.srt2ttml(self.__mocked_srt_path)

        self.assertTrue(mock_read.called)
        self.assertTrue(mock_write.called)

    @patch("pycaption.CaptionConverter.write", side_effect=lambda writer: "output")
    @patch("pycaption.CaptionConverter.read")
    def test_ttml2srt(self, mock_read, mock_write):
        Undertest.ttml2srt(self.__mocked_ttml_path)

        self.assertTrue(mock_read.called)
        self.assertTrue(mock_write.called)

    def test_srt2vtt(self):
        output_file_path = os.path.join(self.__resource_tmp, "converted.vtt")

        Undertest.srt2vtt(self.__real_srt_path, output_file_path)

        self.assertTrue(filecmp.cmp(self.__real_vtt_path, output_file_path))


    def test_vtt2srt(self):
        output_file_path = os.path.join(self.__resource_tmp, "converted.srt")

        Undertest.vtt2srt(self.__real_vtt_path, output_file_path)

        self.assertTrue(filecmp.cmp(self.__real_srt_path, output_file_path))

    @patch("subprocess.Popen.communicate", return_value=1)
    def test_throw_exception_on_srt2vtt_with_error_code(self, mock_communicate):
        try:
            Undertest.srt2vtt(self.__real_srt_path, "output")
        except Exception as e:
            self.assertTrue(mock_communicate.called)
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")


    @patch("subprocess.Popen.communicate", side_effect=subprocess.TimeoutExpired(None, None))
    def test_throw_exception_on_srt2vtt_timeout(self, mock_communicate):
        try:
            Undertest.srt2vtt(self.__real_srt_path, "output")
        except Exception as e:
            self.assertTrue(mock_communicate.called)
            self.assertTrue(isinstance(e, subprocess.TimeoutExpired))
        else:
            self.fail("Should have thrown exception")

    @patch("subprocess.Popen.communicate", side_effect=Exception())
    def test_throw_exception_on_srt2vtt_exception(self, mock_communicate):
        try:
            Undertest.srt2vtt(self.__real_srt_path, "output")
        except Exception as e:
            self.assertTrue(mock_communicate.called)
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")

    @patch("subprocess.Popen.communicate", return_value=1)
    def test_throw_exception_on_vtt2srt_with_error_code(self, mock_communicate):
        try:
            Undertest.vtt2srt(self.__real_vtt_path, "output")
        except Exception as e:
            self.assertTrue(mock_communicate.called)
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")

    @patch("subprocess.Popen.communicate", side_effect=subprocess.TimeoutExpired(None, None))
    def test_throw_exception_on_vtt2srt_timeout(self, mock_communicate):
        try:
            Undertest.vtt2srt(self.__real_vtt_path, "output")
        except Exception as e:
            self.assertTrue(mock_communicate.called)
            self.assertTrue(isinstance(e, subprocess.TimeoutExpired))
        else:
            self.fail("Should have thrown exception")

    @patch("subprocess.Popen.communicate", side_effect=Exception())
    def test_throw_exception_on_vtt2srt_exception(self, mock_communicate):
        try:
            Undertest.vtt2srt(self.__real_vtt_path, "output")
        except Exception as e:
            self.assertTrue(mock_communicate.called)
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")

if __name__ == "__main__":
    unittest.main()
