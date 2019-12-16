from pycaption import (
    CaptionConverter,
    SRTReader,
    DFXPWriter,
    DFXPReader,
    SRTWriter,
)


class Utils(object):
    """Utility functions
    """

    @staticmethod
    def srt2ttml(srt_file_path):
        """Convert SubRip subtitles to TTML subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
        """

        converter = CaptionConverter()
        with open(srt_file_path, "r") as file:
            # converter.read(file.read().decode("iso-8859-1"), SRTReader())
            converter.read(file.read(), SRTReader())
        ttml_file_path = srt_file_path.replace(".srt", ".xml")
        with open(ttml_file_path, "wb") as file:
            file.write(converter.write(DFXPWriter()).encode("utf-8"))

    @staticmethod
    def ttml2srt(ttml_file_path):
        """Convert TTML subtitles to SubRip subtitles.

        Arguments:
            ttml_file_path {string} -- The path to the TTML file.
        """

        converter = CaptionConverter()
        with open(ttml_file_path, "r") as file:
            # converter.read(file.read().decode("utf-8"), DFXPReader())
            converter.read(file.read(), DFXPReader())
        srt_file_path = ttml_file_path.replace(".xml", ".srt")
        with open(srt_file_path, "wb") as file:
            file.write(converter.write(SRTWriter()).encode("utf-8"))

    @staticmethod
    def suppress_lib_logs():
        import os
        import logging
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"
        logging.getLogger('tensorflow').disabled = True
