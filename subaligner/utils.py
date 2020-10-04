import os
import subprocess
from pycaption import (
    CaptionConverter,
    SRTReader,
    DFXPWriter,
    DFXPReader,
    SRTWriter,
)
import pysubs2
from .exception import TerminalException


class Utils(object):
    """Utility functions
    """

    @staticmethod
    def srt2ttml(srt_file_path, ttml_file_path=None):
        """Convert SubRip subtitles to TTML subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            ttml_file_path {string} -- The path to the TTML file.
        """

        converter = CaptionConverter()
        with open(srt_file_path, "r") as file:
            converter.read(file.read(), SRTReader())
        if ttml_file_path is None:
            ttml_file_path = srt_file_path.replace(".srt", ".xml")
        with open(ttml_file_path, "wb") as file:
            file.write(converter.write(DFXPWriter()).encode("utf-8"))

    @staticmethod
    def ttml2srt(ttml_file_path, srt_file_path=None):
        """Convert TTML subtitles to SubRip subtitles.

        Arguments:
            ttml_file_path {string} -- The path to the TTML file.
            srt_file_path {string} -- The path to the SubRip file.
        """

        converter = CaptionConverter()
        with open(ttml_file_path, "r") as file:
            converter.read(file.read(), DFXPReader())
        if srt_file_path is None:
            srt_file_path = ttml_file_path.replace(".xml", ".srt")
        with open(srt_file_path, "wb") as file:
            file.write(converter.write(SRTWriter()).encode("utf-8"))

    @staticmethod
    def srt2vtt(srt_file_path, vtt_file_path=None, timeout_secs=30):
        """Convert SubRip subtitles to WebVTT subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            vtt_file_path {string} -- The path to the WebVTT file.
        """
        if vtt_file_path is None:
            vtt_file_path = srt_file_path.replace(".srt", ".vtt")
        command = "ffmpeg -y -i {0} -f webvtt {1}".format(srt_file_path, vtt_file_path)
        with subprocess.Popen(
            command.split(),
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            universal_newlines=True,
            bufsize=1,
        ) as process:
            try:
                _, std_err = process.communicate(timeout=timeout_secs)
                if process.returncode != 0:
                    raise TerminalException(
                        "Cannot convert SubRip to WebVTT: {} with error {}".format(
                            srt_file_path, std_err
                        )
                    )
            except subprocess.TimeoutExpired as te:
                process.kill()
                raise TerminalException(
                    "Timeout on converting SubRip to WebVTT: {}".format(
                        srt_file_path
                    )
                ) from te
            except Exception as e:
                process.kill()
                if isinstance(e, TerminalException):
                    raise e
                else:
                    raise TerminalException(
                        "Cannot convert SubRip to WebVTT: {}".format(
                            srt_file_path
                        )
                    ) from e
            finally:
                os.system("stty sane")

    @staticmethod
    def vtt2srt(vtt_file_path, srt_file_path=None, timeout_secs=30):
        """Convert WebVTT subtitles to SubRip subtitles.

        Arguments:
            vtt_file_path {string} -- The path to the WebVTT file.
            srt_file_path {string} -- The path to the SubRip file.
        """
        if srt_file_path is None:
            srt_file_path = vtt_file_path.replace(".vtt", ".srt")
        command = "ffmpeg -y -i {0} -f srt {1}".format(vtt_file_path, srt_file_path)
        with subprocess.Popen(
            command.split(),
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            universal_newlines=True,
            bufsize=1,
        ) as process:
            try:
                _, std_err = process.communicate(timeout=timeout_secs)
                if process.returncode != 0:
                    raise TerminalException(
                        "Cannot convert WebVTT to SubRip: {} with error {}".format(
                            vtt_file_path, std_err
                        )
                    )
            except subprocess.TimeoutExpired as te:
                process.kill()
                process.wait()
                raise TerminalException(
                    "Timeout on converting WebVTT to SubRip: {}".format(
                        vtt_file_path
                    )
                ) from te
            except Exception as e:
                process.kill()
                process.wait()
                if isinstance(e, TerminalException):
                    raise e
                else:
                    raise TerminalException(
                        "Cannot convert WebVTT to SubRip: {}".format(
                            vtt_file_path
                        )
                    ) from e
            finally:
                os.system("stty sane")

    @staticmethod
    def srt2ass(srt_file_path, ass_file_path=None):
        """Convert SubRip subtitles to Advanced SubStation Alpha v4.0+ subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            ass_file_path {string} -- The path to the ASS file.
        """

        Utils.__convert_subtitle(srt_file_path, "srt", ass_file_path, "ass", "ass")

    @staticmethod
    def ass2srt(ass_file_path, srt_file_path=None):
        """Convert Advanced SubStation Alpha v4.0+ subtitles to SubRip subtitles.

        Arguments:
            ass_file_path {string} -- The path to the ASS file.
            srt_file_path {string} -- The path to the SubRip file.
        """

        Utils.__convert_subtitle(ass_file_path, "ass", srt_file_path, "srt", "srt")

    @staticmethod
    def srt2ssa(srt_file_path, ssa_file_path=None):
        """Convert SubRip subtitles to SubStation Alpha v4.0 subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            ssa_file_path {string} -- The path to the SSA file.
        """

        Utils.__convert_subtitle(srt_file_path, "srt", ssa_file_path, "ssa", "ssa")

    @staticmethod
    def ssa2srt(ssa_file_path, srt_file_path=None):
        """Convert SubStation Alpha v4.0 subtitles to SubRip subtitles.

        Arguments:
            ssa_file_path {string} -- The path to the SSA file.
            srt_file_path {string} -- The path to the SubRip file.
        """

        Utils.__convert_subtitle(ssa_file_path, "ssa", srt_file_path, "srt", "srt")

    @staticmethod
    def srt2microdvd(srt_file_path, microdvd_file_path=None, frame_rate=25.0):
        """Convert SubRip subtitles to MicroDVD subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            microdvd_file_path {string} -- The path to the MicroDVD file.
            frame_rate {float} -- The frame rate for frame-based MicroDVD.
        """

        Utils.__convert_subtitle(srt_file_path, "srt", microdvd_file_path, "sub", "microdvd", frame_rate=frame_rate)

    @staticmethod
    def microdvd2srt(microdvd_file_path, srt_file_path=None):
        """Convert MicroDVD subtitles to SubRip subtitles.

        Arguments:
            microdvd_file_path {string} -- The path to the MPL2 file.
            srt_file_path {string} -- The path to the SubRip file.
        """

        Utils.__convert_subtitle(microdvd_file_path, "sub", srt_file_path, "srt", "srt")

    @staticmethod
    def srt2mpl2(srt_file_path, mpl2_file_path=None):
        """Convert SubRip subtitles to MPL2 subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            mpl2_file_path {string} -- The path to the MPL2 file.
        """

        Utils.__convert_subtitle(srt_file_path, "srt", mpl2_file_path, "txt", "mpl2")

    @staticmethod
    def mpl22srt(mpl2_file_path, srt_file_path=None):
        """Convert MPL2 subtitles to SubRip subtitles.

        Arguments:
            mpl2_file_path {string} -- The path to the MPL2 file.
            srt_file_path {string} -- The path to the SubRip file.
        """

        Utils.__convert_subtitle(mpl2_file_path, "txt", srt_file_path, "srt", "srt")

    @staticmethod
    def srt2tmp(srt_file_path, tmp_file_path=None):
        """Convert SubRip subtitles to TMP subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            tmp_file_path {string} -- The path to the TMP file.
        """

        Utils.__convert_subtitle(srt_file_path, "srt", tmp_file_path, "tmp", "tmp")

    @staticmethod
    def tmp2srt(tmp_file_path, srt_file_path=None):
        """Convert TMP subtitles to SubRip subtitles.

        Arguments:
            mpl2_file_path {string} -- The path to the TMP file.
            tmp_file_path {string} -- The path to the SubRip file.
        """

        Utils.__convert_subtitle(tmp_file_path, "tmp", srt_file_path, "srt", "srt")

    @staticmethod
    def suppress_lib_logs():
        import os
        import logging
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "2"
        logging.getLogger("tensorflow").disabled = True

    @staticmethod
    def __convert_subtitle(source_file_path, source_ext, target_file_path, target_ext, format, frame_rate=None):
        subs = pysubs2.load(source_file_path, encoding="utf-8")
        if target_file_path is None:
            target_file_path = source_file_path.replace(".%s" % source_ext, ".%s" % target_ext)
        if frame_rate is None:
            subs.save(target_file_path, encoding="utf-8", format_=format)
        else:
            subs.save(target_file_path, encoding="utf-8", format_=format, fps=frame_rate)
