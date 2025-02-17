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
        self.mocked_srt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_utils.srt"
        )
        self.mocked_ttml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_utils.xml"
        )
        self.mocked_sami_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_utils.smi"
        )
        self.mocked_scc_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_utils.scc"
        )
        self.real_srt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.srt"
        )
        self.real_vtt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.vtt"
        )
        self.real_ass_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ass"
        )
        self.real_ssa_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ssa"
        )
        self.real_mircodvd_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.sub"
        )
        self.real_mpl2_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_mpl2.txt"
        )
        self.real_tmp_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.tmp"
        )
        self.real_stl_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.stl"
        )
        self.real_sbv_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.sbv"
        )
        self.real_ytt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ytt"
        )
        self.real_json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.json"
        )
        self.mp4_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.mp4"
        )
        self.mkv_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.mkv"
        )
        self.with_newlines_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/with_newlines.txt"
        )
        self.resource_tmp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/tmp"
        )
        if os.path.exists(self.resource_tmp):
            shutil.rmtree(self.resource_tmp)
        os.mkdir(self.resource_tmp)

    def tearDown(self):
        shutil.rmtree(self.resource_tmp)

    def test_srt2vtt(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.vtt")

        Undertest.srt2vtt(self.real_srt_path, output_file_path)

        self.assertTrue(filecmp.cmp(self.real_vtt_path, output_file_path))

    def test_vtt2srt(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.vtt.srt")

        Undertest.vtt2srt(self.real_vtt_path, output_file_path)

        self.assertTrue(filecmp.cmp(self.real_srt_path, output_file_path))

    def test_srt2ass(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.ass")

        Undertest.srt2ass(self.real_srt_path, output_file_path)

        self.assertTrue(filecmp.cmp(self.real_ass_path, output_file_path))

    def test_ass2srt(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.ass.srt")

        Undertest.ass2srt(self.real_ass_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_srt2ssa(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.ssa")

        Undertest.srt2ssa(self.real_srt_path, output_file_path)

        self.assertTrue(filecmp.cmp(self.real_ssa_path, output_file_path))

    def test_ssa2srt(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.ssa.srt")

        Undertest.ssa2srt(self.real_ssa_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_srt2microdvd(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.sub")

        Undertest.srt2microdvd(self.real_srt_path, output_file_path, 25.0)

        self.assertTrue(filecmp.cmp(self.real_mircodvd_path, output_file_path))

    def test_microdvd2srt(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.sub.srt")

        Undertest.microdvd2srt(self.real_mircodvd_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_srt2mpl2(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.mpl2.txt")

        Undertest.srt2mpl2(self.real_srt_path, output_file_path)

        self.assertTrue(filecmp.cmp(self.real_mpl2_path, output_file_path))

    def test_mpl22srt(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.mpl2.txt.srt")

        Undertest.mpl22srt(self.real_mpl2_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_srt2tmp(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.tmp")

        Undertest.srt2tmp(self.real_srt_path, output_file_path)

        self.assertTrue(filecmp.cmp(self.real_tmp_path, output_file_path))

    def test_tmp2srt(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.tmp.srt")

        Undertest.mpl22srt(self.real_mpl2_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_stl2srt(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.stl.srt")

        Undertest.stl2srt(self.real_stl_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    @patch("pycaption.CaptionConverter.write", side_effect=lambda writer: "output")
    @patch("pycaption.CaptionConverter.read")
    def test_srt2ttml(self, mock_read, mock_write):
        Undertest.srt2ttml(self.mocked_srt_path)

        self.assertTrue(mock_read.called)
        self.assertTrue(mock_write.called)

    @patch("pycaption.CaptionConverter.write", side_effect=lambda writer: "output")
    @patch("pycaption.CaptionConverter.read")
    def test_ttml2srt(self, mock_read, mock_write):
        Undertest.ttml2srt(self.mocked_ttml_path)

        self.assertTrue(mock_read.called)
        self.assertTrue(mock_write.called)

    @patch("pycaption.CaptionConverter.write", side_effect=lambda writer: "output")
    @patch("pycaption.CaptionConverter.read")
    def test_srt2sami(self, mock_read, mock_write):
        Undertest.srt2sami(self.mocked_srt_path)

        self.assertTrue(mock_read.called)
        self.assertTrue(mock_write.called)

    @patch("pycaption.CaptionConverter.write", side_effect=lambda writer: "output")
    @patch("pycaption.CaptionConverter.read")
    def test_sami2srt(self, mock_read, mock_write):
        Undertest.sami2srt(self.mocked_sami_path)

        self.assertTrue(mock_read.called)
        self.assertTrue(mock_write.called)

    @patch("pycaption.CaptionConverter.write", side_effect=lambda writer: "output")
    @patch("pycaption.CaptionConverter.read")
    def test_srt2scc(self, mock_read, mock_write):
        Undertest.srt2scc(self.mocked_srt_path)

        self.assertTrue(mock_read.called)
        self.assertTrue(mock_write.called)

    @patch("pycaption.CaptionConverter.write", side_effect=lambda writer: "output")
    @patch("pycaption.CaptionConverter.read")
    def test_scc2srt(self, mock_read, mock_write):
        Undertest.scc2srt(self.mocked_scc_path)

        self.assertTrue(mock_read.called)
        self.assertTrue(mock_write.called)

    def test_srt2sbv(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.sbv")

        Undertest.srt2sbv(self.real_srt_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_sbv2srt(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.srt")

        Undertest.sbv2srt(self.real_sbv_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_srt2ytt(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.ytt")

        Undertest.srt2ytt(self.real_srt_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_ytt2srt(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.srt")

        Undertest.ytt2srt(self.real_ytt_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_srt2json(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.json")

        Undertest.srt2json(self.real_srt_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_json2srt(self):
        output_file_path = os.path.join(self.resource_tmp, "converted.srt")

        Undertest.json2srt(self.real_json_path, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    @patch("subaligner.utils.Utils._run_command")
    def test_extract_teletext_as_srt(self, mocked_run_command):
        Undertest.extract_teletext_as_subtitle("ts_file_path", 888, "srt_file_path")

        mocked_run_command.assert_called_once_with("ffmpeg -y -fix_sub_duration -txt_page 888 -txt_format text -i {} {}".format("\"ts_file_path\"", "\"srt_file_path\""), ANY, ANY, ANY, ANY)

    def test_extract_matroska_subtitle(self):
        output_file_path = os.path.join(self.resource_tmp, "extracted.matroska.srt")

        Undertest.extract_matroska_subtitle(self.mkv_file_path, 0, output_file_path)

        self.assertTrue(os.path.isfile(output_file_path))

    def test_remove_trailing_newlines(self):
        output_file_path = os.path.join(self.resource_tmp, "stripped.txt")

        Undertest.remove_trailing_newlines(self.with_newlines_path, "utf-8", output_file_path)

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
        self.assertTrue(Undertest.contains_embedded_subtitles(self.mkv_file_path))
        self.assertFalse(Undertest.contains_embedded_subtitles(self.mp4_file_path))

    def test_detect_encoding(self):
        self.assertEqual("ascii", Undertest.detect_encoding(self.real_srt_path))
        self.assertEqual("utf-8", Undertest.detect_encoding(self.mkv_file_path))

    def test_get_file_root_and_extension(self):
        root, extension = Undertest.get_file_root_and_extension("/path/to/root.ext1.ext2")
        self.assertEqual("/path/to/root", root)
        self.assertEqual("ext1.ext2", extension)

    def test_get_stretch_language_codes(self):
        self.assertEqual(87, len(Undertest.get_stretch_language_codes()))

    def test_get_misc_language_codes(self):
        self.assertEqual(200, len(Undertest.get_misc_language_codes()))

    def test_get_language_table(self):
        self.assertEqual(200, len(Undertest.get_language_table()))

    def test_get_iso_639_alpha_2(self):
        self.assertEqual("en", Undertest.get_iso_639_alpha_2("eng"))
        self.assertEqual("ada", Undertest.get_iso_639_alpha_2("ada"))
        self.assertEqual("xyz", Undertest.get_iso_639_alpha_2("xyz"))

    def test_format_timestamp(self):
        test_cases = [
            (0, "00:00:00,000"),
            (100, "00:01:40,000"),
            (100.1, "00:01:40,100"),
        ]
        for seconds, time_code in test_cases:
            self.assertEqual(time_code, Undertest.format_timestamp(seconds))

    def test_double_quoted(self):
        self.assertEqual("\"file'path\"", Undertest.double_quoted("file'path"))
        self.assertEqual("\"file\\\"path\"", Undertest.double_quoted("file\"path"))

    @patch("subprocess.Popen.communicate", return_value=1)
    def test_throw_exception_on_srt2vtt_with_error_code(self, mock_communicate):
        self._assert_exception_on_subproces(lambda: Undertest.srt2vtt(self.real_srt_path, "output"), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=subprocess.TimeoutExpired("", 1.0))
    def test_throw_exception_on_srt2vtt_timeout(self, mock_communicate):
        self._assert_exception_on_subproces(lambda: Undertest.srt2vtt(self.real_srt_path, "output"), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=Exception())
    def test_throw_exception_on_srt2vtt_exception(self, mock_communicate):
        self._assert_exception_on_subproces(lambda: Undertest.srt2vtt(self.real_srt_path, "output"), mock_communicate)

    @patch("subprocess.Popen.communicate", return_value=1)
    def test_throw_exception_on_vtt2srt_with_error_code(self, mock_communicate):
        self._assert_exception_on_subproces(lambda: Undertest.vtt2srt(self.real_vtt_path, "output"), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=subprocess.TimeoutExpired("", 1.0))
    def test_throw_exception_on_vtt2srt_timeout(self, mock_communicate):
        self._assert_exception_on_subproces(lambda: Undertest.vtt2srt(self.real_vtt_path, "output"), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=Exception())
    def test_throw_exception_on_vtt2srt_exception(self, mock_communicate):
        self._assert_exception_on_subproces(lambda: Undertest.vtt2srt(self.real_vtt_path, "output"), mock_communicate)

    @patch("subprocess.Popen.communicate", return_value=1)
    def test_throw_exception_on_extracting_teletext_with_error_code(self, mock_communicate):
        output_file_path = os.path.join(self.resource_tmp, "extracted.tmp.srt")
        self._assert_exception_on_subproces(
            lambda: Undertest.extract_teletext_as_subtitle("ts_file_path", 888, output_file_path),
            mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=subprocess.TimeoutExpired("", 1.0))
    def test_throw_exception_on_extracting_teletext_timeout(self, mock_communicate):
        output_file_path = os.path.join(self.resource_tmp, "extracted.tmp.srt")
        self._assert_exception_on_subproces(
            lambda: Undertest.extract_teletext_as_subtitle("ts_file_path", 888, output_file_path),
            mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=Exception())
    def test_throw_exception_on_extracting_teletext_exception(self, mock_communicate):
        output_file_path = os.path.join(self.resource_tmp, "extracted.tmp.srt")
        self._assert_exception_on_subproces(
            lambda: Undertest.extract_teletext_as_subtitle("ts_file_path", 888, output_file_path),
            mock_communicate)

    @patch("subprocess.Popen.communicate", return_value=1)
    def test_throw_exception_on_extracting_matroska_subtitle_with_error_code(self, mock_communicate):
        output_file_path = os.path.join(self.resource_tmp, "extracted.tmp.srt")
        self._assert_exception_on_subproces(
            lambda: Undertest.extract_matroska_subtitle(self.mkv_file_path, 0, output_file_path), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=subprocess.TimeoutExpired("", 1.0))
    def test_throw_exception_on_extracting_matroska_subtitle_timeout(self, mock_communicate):
        output_file_path = os.path.join(self.resource_tmp, "extracted.tmp.srt")
        self._assert_exception_on_subproces(
            lambda: Undertest.extract_matroska_subtitle(self.mkv_file_path, 0, output_file_path), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=Exception())
    def test_throw_exception_on_extracting_matroska_subtitle_exception(self, mock_communicate):
        output_file_path = os.path.join(self.resource_tmp, "extracted.tmp.srt")
        self._assert_exception_on_subproces(
            lambda: Undertest.extract_matroska_subtitle(self.mkv_file_path, 0, output_file_path), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=subprocess.TimeoutExpired("", 1.0))
    def test_throw_exception_on_detecting_embedded_subtitles(self, mock_communicate):
        self._assert_exception_on_subproces(
            lambda: self.assertTrue(Undertest.contains_embedded_subtitles(self.mkv_file_path)), mock_communicate)

    @patch("subprocess.Popen.communicate", side_effect=Exception())
    def test_throw_exception_on_detecting_embedded_subtitles_exception(self, mock_communicate):
        self._assert_exception_on_subproces(
            lambda: self.assertTrue(Undertest.contains_embedded_subtitles(self.mkv_file_path)), mock_communicate)

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
