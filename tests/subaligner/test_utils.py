import os
import unittest
from subaligner.utils import Utils as Undertest
from mock import patch


class UtilsTests(unittest.TestCase):
    def setUp(self):
        self.__subtitle_srt_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_utils.srt"
        )
        self.__subtitle_ttml_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_utils.xml"
        )

    @patch("pycaption.CaptionConverter.write", side_effect=lambda writer: "output")
    @patch("pycaption.CaptionConverter.read")
    def test_srt2ttml(self, mock_read, mock_write):
        Undertest.srt2ttml(self.__subtitle_srt_path)

        assert mock_read.called
        assert mock_write.called

    @patch("pycaption.CaptionConverter.write", side_effect=lambda writer: "output")
    @patch("pycaption.CaptionConverter.read")
    def test_ttml2srt(self, mock_read, mock_write):
        Undertest.ttml2srt(self.__subtitle_ttml_path)

        assert mock_read.called
        assert mock_write.called


if __name__ == "__main__":
    unittest.main()
