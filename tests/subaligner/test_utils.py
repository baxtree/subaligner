import os
import unittest
import shutil
import filecmp
import subprocess
import requests
import shutil
from subaligner.utils import Utils as Undertest
from subaligner.exception import TerminalException
from mock import patch
from unittest.mock import ANY


class UtilsTests(unittest.TestCase):
    def setUp(self):
        self.__mocked_srt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_utils.srt"
        )
        self.__mocked_ttml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_utils.xml"
        )
        self.__mocked_sami_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_utils.smi"
        )
        self.__real_srt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.srt"
        )
        self.__real_vtt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.vtt"
        )
        self.__real_ass_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ass"
        )
        self.__real_ssa_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ssa"
        )
        self.__real_mircodvd_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.sub"
        )
        self.__real_mpl2_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.mpl2.txt"
        )
        self.__real_tmp_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.tmp"
        )
        self.__real_stl_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.stl"
        )
        self.__ts_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ts"
        )
        self.__mp4_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.mp4"
        )
        self.__mkv_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.mkv"
        )
        self.__with_newlines_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/with_newlines.txt"
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
        output_file_path = os.path.join(self.__resource_tmp, "converted.vtt.srt")

        Undertest.vtt2srt(self.__real_vtt_path, output_file_path)

        self.assertTrue(filecmp.cmp(self.__real_srt_path, output_file_path))

    def test_srt2ass(self):
        output_file_path = os.path.join(self.__resource_tmp, "converted.ass")

        Undertest.srt2ass(self.__real_srt_path, output_file_path)

        self.assertTrue(filecmp.cmp(self.__real_ass_path, output_file_path))

    def test_ass2srt(self):
        output_file_path = os.path.join(self.__resource_tmp, "converted.ass.srt")

        Undertest.ass2srt(self.__real_ass_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_srt2ssa(self):
        output_file_path = os.path.join(self.__resource_tmp, "converted.ssa")

        Undertest.srt2ssa(self.__real_srt_path, output_file_path)

        self.assertTrue(filecmp.cmp(self.__real_ssa_path, output_file_path))

    def test_ssa2srt(self):
        output_file_path = os.path.join(self.__resource_tmp, "converted.ssa.srt")

        Undertest.ssa2srt(self.__real_ssa_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_srt2microdvd(self):
        output_file_path = os.path.join(self.__resource_tmp, "converted.sub")

        Undertest.srt2microdvd(self.__real_srt_path, output_file_path)

        self.assertTrue(filecmp.cmp(self.__real_mircodvd_path, output_file_path))

    def test_microdvd2srt(self):
        output_file_path = os.path.join(self.__resource_tmp, "converted.sub.srt")

        Undertest.microdvd2srt(self.__real_mircodvd_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_srt2mpl2(self):
        output_file_path = os.path.join(self.__resource_tmp, "converted.mpl2.txt")

        Undertest.srt2mpl2(self.__real_srt_path, output_file_path)

        self.assertTrue(filecmp.cmp(self.__real_mpl2_path, output_file_path))

    def test_mpl22srt(self):
        output_file_path = os.path.join(self.__resource_tmp, "converted.mpl2.txt.srt")

        Undertest.mpl22srt(self.__real_mpl2_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_srt2tmp(self):
        output_file_path = os.path.join(self.__resource_tmp, "converted.tmp")

        Undertest.srt2tmp(self.__real_srt_path, output_file_path)

        self.assertTrue(filecmp.cmp(self.__real_tmp_path, output_file_path))

    def test_tmp2srt(self):
        output_file_path = os.path.join(self.__resource_tmp, "converted.tmp.srt")

        Undertest.mpl22srt(self.__real_mpl2_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    @patch("pycaption.CaptionConverter.write", side_effect=lambda writer: "output")
    @patch("pycaption.CaptionConverter.read")
    def test_srt2sami(self, mock_read, mock_write):
        Undertest.srt2sami(self.__mocked_srt_path)

        self.assertTrue(mock_read.called)
        self.assertTrue(mock_write.called)

    @patch("pycaption.CaptionConverter.write", side_effect=lambda writer: "output")
    @patch("pycaption.CaptionConverter.read")
    def test_sami2srt(self, mock_read, mock_write):
        Undertest.sami2srt(self.__mocked_sami_path)

        self.assertTrue(mock_read.called)
        self.assertTrue(mock_write.called)

    def test_stl2srt(self):
        output_file_path = os.path.join(self.__resource_tmp, "converted.stl.srt")

        Undertest.stl2srt(self.__real_stl_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    @unittest.skip("transport stream sample does not contain any teletext")
    def test_extract_teletext_as_srt(self):
        output_file_path = os.path.join(self.__resource_tmp, "extracted.teletext.srt")

        Undertest.extract_teletext_as_subtitle(self.__ts_file_path, 888, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_extract_matroska_subtitle(self):
        output_file_path = os.path.join(self.__resource_tmp, "extracted.matroska.srt")

        Undertest.extract_matroska_subtitle(self.__mkv_file_path, 0, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_remove_trailing_newlines(self):
        output_file_path = os.path.join(self.__resource_tmp, "stripped.txt")

        Undertest.remove_trailing_newlines(self.__with_newlines_path, "utf-8", output_file_path)

        with open(output_file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            self.assertEqual(3, len(lines))
            self.assertEqual("\n", lines[0])
            self.assertEqual("\n", lines[1])
            self.assertEqual("Contains two leading newlines and three trailing newlines", lines[2])

    @patch("requests.get")
    @patch("builtins.open")
    @patch("shutil.copyfileobj")
    def test_download(self, mocked_copyfileobj, mocked_open, mocked_get):
        Undertest.download_file("remote_file_url", "local_file_path")

        mocked_get.assert_called_once_with("remote_file_url", verify=True, stream=True)
        mocked_open.assert_called_once_with("local_file_path", "wb")
        mocked_copyfileobj.assert_called_once_with(ANY, ANY)

    def test_contains_embedded_subtitle(self):
        self.assertTrue(Undertest.contains_embedded_subtitles(self.__mkv_file_path))
        self.assertFalse(Undertest.contains_embedded_subtitles(self.__mp4_file_path))

    def test_detect_encoding(self):
        self.assertEqual("ASCII", Undertest.detect_encoding(self.__real_srt_path))
        self.assertEqual("UTF-8", Undertest.detect_encoding(self.__mkv_file_path))

    @patch("subprocess.Popen.communicate", return_value=1)
    def test_throw_exception_on_srt2vtt_with_error_code(self, mock_communicate):
        self._assert_exception_on_subproces(lambda: Undertest.srt2vtt(self.__real_srt_path, "output"), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=subprocess.TimeoutExpired("", 1.0))
    def test_throw_exception_on_srt2vtt_timeout(self, mock_communicate):
        self._assert_exception_on_subproces(lambda: Undertest.srt2vtt(self.__real_srt_path, "output"), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=Exception())
    def test_throw_exception_on_srt2vtt_exception(self, mock_communicate):
        self._assert_exception_on_subproces(lambda: Undertest.srt2vtt(self.__real_srt_path, "output"), mock_communicate)

    @patch("subprocess.Popen.communicate", return_value=1)
    def test_throw_exception_on_vtt2srt_with_error_code(self, mock_communicate):
        self._assert_exception_on_subproces(lambda: Undertest.vtt2srt(self.__real_vtt_path, "output"), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=subprocess.TimeoutExpired("", 1.0))
    def test_throw_exception_on_vtt2srt_timeout(self, mock_communicate):
        self._assert_exception_on_subproces(lambda: Undertest.vtt2srt(self.__real_vtt_path, "output"), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=Exception())
    def test_throw_exception_on_vtt2srt_exception(self, mock_communicate):
        self._assert_exception_on_subproces(lambda: Undertest.vtt2srt(self.__real_vtt_path, "output"), mock_communicate)

    @patch("subprocess.Popen.communicate", return_value=1)
    def test_throw_exception_on_extracting_teletext_with_error_code(self, mock_communicate):
        output_file_path = os.path.join(self.__resource_tmp, "extracted.tmp.srt")
        self._assert_exception_on_subproces(
            lambda: Undertest.extract_teletext_as_subtitle(self.__ts_file_path, 888, output_file_path),
            mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=subprocess.TimeoutExpired("", 1.0))
    def test_throw_exception_on_extracting_teletext_timeout(self, mock_communicate):
        output_file_path = os.path.join(self.__resource_tmp, "extracted.tmp.srt")
        self._assert_exception_on_subproces(
            lambda: Undertest.extract_teletext_as_subtitle(self.__ts_file_path, 888, output_file_path),
            mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=Exception())
    def test_throw_exception_on_extracting_teletext_exception(self, mock_communicate):
        output_file_path = os.path.join(self.__resource_tmp, "extracted.tmp.srt")
        self._assert_exception_on_subproces(
            lambda: Undertest.extract_teletext_as_subtitle(self.__ts_file_path, 888, output_file_path),
            mock_communicate)

    @patch("subprocess.Popen.communicate", return_value=1)
    def test_throw_exception_on_extracting_matroska_subtitle_with_error_code(self, mock_communicate):
        output_file_path = os.path.join(self.__resource_tmp, "extracted.tmp.srt")
        self._assert_exception_on_subproces(
            lambda: Undertest.extract_matroska_subtitle(self.__mkv_file_path, 0, output_file_path), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=subprocess.TimeoutExpired("", 1.0))
    def test_throw_exception_on_extracting_matroska_subtitle_timeout(self, mock_communicate):
        output_file_path = os.path.join(self.__resource_tmp, "extracted.tmp.srt")
        self._assert_exception_on_subproces(
            lambda: Undertest.extract_matroska_subtitle(self.__mkv_file_path, 0, output_file_path), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=Exception())
    def test_throw_exception_on_extracting_matroska_subtitle_exception(self, mock_communicate):
        output_file_path = os.path.join(self.__resource_tmp, "extracted.tmp.srt")
        self._assert_exception_on_subproces(
            lambda: Undertest.extract_matroska_subtitle(self.__mkv_file_path, 0, output_file_path), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=subprocess.TimeoutExpired("", 1.0))
    def test_throw_exception_on_detecting_embedded_subtitles(self, mock_communicate):
        self._assert_exception_on_subproces(
            lambda: self.assertTrue(Undertest.contains_embedded_subtitles(self.__mkv_file_path)), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=Exception())
    def test_throw_exception_on_detecting_embedded_subtitles_exception(self, mock_communicate):
        self._assert_exception_on_subproces(
            lambda: self.assertTrue(Undertest.contains_embedded_subtitles(self.__mkv_file_path)), mock_communicate)

    def _assert_exception_on_subproces(self, trigger, mock_communicate):
        try:
            trigger()
        except Exception as e:
            self.assertTrue(mock_communicate.called)
            self.assertTrue(isinstance(e, TerminalException))
        else:
            self.fail("Should have thrown exception")


if __name__ == "__main__":
    unittest.main()
