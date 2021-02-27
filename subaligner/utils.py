import os
import subprocess
import pysubs2
import requests
import shutil
import cchardet

from pycaption import (
    CaptionConverter,
    SRTWriter,
    SRTReader,
    DFXPWriter,
    DFXPReader,
    SAMIWriter,
    SAMIReader,
)
from typing import Optional, TextIO, BinaryIO, Union, Callable, Any, Tuple
from .exception import TerminalException
from subaligner.lib.to_srt import STL, SRT


class Utils(object):
    """Utility functions
    """

    FFMPEG_BIN = os.getenv("FFMPEG_PATH") or os.getenv("ffmpeg_path") or "ffmpeg"

    @staticmethod
    def srt2ttml(srt_file_path: str, ttml_file_path: Optional[str] = None) -> None:
        """Convert SubRip subtitles to TTML subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            ttml_file_path {string} -- The path to the TTML file.
        """

        file: Union[TextIO, BinaryIO]
        converter = CaptionConverter()
        encoding = Utils.detect_encoding(srt_file_path)
        with open(srt_file_path, "r", encoding=encoding) as file:
            converter.read(file.read(), SRTReader())
        if ttml_file_path is None:
            ttml_file_path = srt_file_path.replace(".srt", ".xml")
        with open(ttml_file_path, "wb") as file:
            file.write(converter.write(DFXPWriter()).encode(encoding))

    @staticmethod
    def ttml2srt(ttml_file_path: str, srt_file_path: Optional[str] = None) -> None:
        """Convert TTML subtitles to SubRip subtitles.

        Arguments:
            ttml_file_path {string} -- The path to the TTML file.
            srt_file_path {string} -- The path to the SubRip file.
        """

        file: Union[TextIO, BinaryIO]
        converter = CaptionConverter()
        encoding = Utils.detect_encoding(ttml_file_path)
        with open(ttml_file_path, "r", encoding=encoding) as file:
            converter.read(file.read(), DFXPReader())
        if srt_file_path is None:
            srt_file_path = ttml_file_path.replace(".xml", ".srt")
        with open(srt_file_path, "wb") as file:
            file.write(converter.write(SRTWriter()).encode(encoding))

    @staticmethod
    def srt2vtt(srt_file_path: str, vtt_file_path: Optional[str] = None, timeout_secs: int = 30) -> None:
        """Convert SubRip subtitles to WebVTT subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            vtt_file_path {string} -- The path to the WebVTT file.
            timeout_secs {int} -- The timeout in seconds on conversion {default: 30}.
        """

        _vtt_file_path = srt_file_path.replace(".srt", ".vtt") if vtt_file_path is None else vtt_file_path
        encoding = Utils.detect_encoding(srt_file_path)
        command = "{0} -y -sub_charenc {1} -i {2} -f webvtt {3}".format(Utils.FFMPEG_BIN, encoding, srt_file_path, _vtt_file_path)
        timeout_msg = "Timeout on converting SubRip to WebVTT: {}".format(srt_file_path)
        error_msg = "Cannot convert SubRip to WebVTT: {}".format(srt_file_path)

        def _callback(returncode: int, std_err: str) -> None:
            if returncode != 0:
                raise TerminalException(
                    "Cannot convert SubRip to WebVTT: {} with error {}".format(
                        srt_file_path, std_err
                    )
                )
            Utils.remove_trailing_newlines(_vtt_file_path, encoding)

        Utils.__run_command(command, timeout_secs, timeout_msg, error_msg, _callback)

    @staticmethod
    def vtt2srt(vtt_file_path: str, srt_file_path: Optional[str] = None, timeout_secs: int = 30) -> None:
        """Convert WebVTT subtitles to SubRip subtitles.

        Arguments:
            vtt_file_path {string} -- The path to the WebVTT file.
            srt_file_path {string} -- The path to the SubRip file.
            timeout_secs {int} -- The timeout in seconds on conversion {default: 30}.
        """

        _srt_file_path = vtt_file_path.replace(".vtt", ".srt") if srt_file_path is None else srt_file_path
        encoding = Utils.detect_encoding(vtt_file_path)
        command = "{0} -y -sub_charenc {1} -i {2} -f srt {3}".format(Utils.FFMPEG_BIN, encoding, vtt_file_path, _srt_file_path)
        timeout_msg = "Timeout on converting WebVTT to SubRip: {}".format(vtt_file_path)
        error_msg = "Cannot convert WebVTT to SubRip: {}".format(vtt_file_path)

        def _callback(returncode: int, std_err: str) -> None:
            if returncode != 0:
                raise TerminalException(
                    "Cannot convert WebVTT to SubRip: {} with error {}".format(
                        vtt_file_path, std_err
                    )
                )
            Utils.remove_trailing_newlines(_srt_file_path, encoding)

        Utils.__run_command(command, timeout_secs, timeout_msg, error_msg, _callback)

    @staticmethod
    def srt2ass(srt_file_path: str, ass_file_path: Optional[str] = None) -> None:
        """Convert SubRip subtitles to Advanced SubStation Alpha v4.0+ subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            ass_file_path {string} -- The path to the ASS file.
        """

        new_ass_file_path, encoding = Utils.__convert_subtitle(srt_file_path, "srt", ass_file_path, "ass", "ass")
        Utils.remove_trailing_newlines(new_ass_file_path, encoding)

    @staticmethod
    def ass2srt(ass_file_path: str, srt_file_path: Optional[str] = None) -> None:
        """Convert Advanced SubStation Alpha v4.0+ subtitles to SubRip subtitles.

        Arguments:
            ass_file_path {string} -- The path to the ASS file.
            srt_file_path {string} -- The path to the SubRip file.
        """

        new_srt_file_path, encoding = Utils.__convert_subtitle(ass_file_path, "ass", srt_file_path, "srt", "srt")
        Utils.remove_trailing_newlines(new_srt_file_path, encoding)

    @staticmethod
    def srt2ssa(srt_file_path: str, ssa_file_path: Optional[str] = None) -> None:
        """Convert SubRip subtitles to SubStation Alpha v4.0 subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            ssa_file_path {string} -- The path to the SSA file.
        """

        new_ssa_file_path, encoding = Utils.__convert_subtitle(srt_file_path, "srt", ssa_file_path, "ssa", "ssa")
        Utils.remove_trailing_newlines(new_ssa_file_path, encoding)

    @staticmethod
    def ssa2srt(ssa_file_path: str, srt_file_path: Optional[str] = None) -> None:
        """Convert SubStation Alpha v4.0 subtitles to SubRip subtitles.

        Arguments:
            ssa_file_path {string} -- The path to the SSA file.
            srt_file_path {string} -- The path to the SubRip file.
        """

        new_srt_file_path, encoding = Utils.__convert_subtitle(ssa_file_path, "ssa", srt_file_path, "srt", "srt")
        Utils.remove_trailing_newlines(new_srt_file_path, encoding)

    @staticmethod
    def srt2microdvd(srt_file_path: str, microdvd_file_path: Optional[str] = None, frame_rate: Optional[float] = 25.0):
        """Convert SubRip subtitles to MicroDVD subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            microdvd_file_path {string} -- The path to the MicroDVD file.
            frame_rate {float} -- The frame rate for frame-based MicroDVD.
        """

        new_microdvd_file_path, encoding = Utils.__convert_subtitle(srt_file_path, "srt", microdvd_file_path, "sub", "microdvd", frame_rate=frame_rate)
        Utils.remove_trailing_newlines(new_microdvd_file_path, encoding)

    @staticmethod
    def microdvd2srt(microdvd_file_path: str, srt_file_path: Optional[str] = None) -> None:
        """Convert MicroDVD subtitles to SubRip subtitles.

        Arguments:
            microdvd_file_path {string} -- The path to the MPL2 file.
            srt_file_path {string} -- The path to the SubRip file.
        """

        new_srt_file_path, encoding = Utils.__convert_subtitle(microdvd_file_path, "sub", srt_file_path, "srt", "srt")
        Utils.remove_trailing_newlines(new_srt_file_path, encoding)

    @staticmethod
    def srt2mpl2(srt_file_path: str, mpl2_file_path: Optional[str] = None) -> None:
        """Convert SubRip subtitles to MPL2 subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            mpl2_file_path {string} -- The path to the MPL2 file.
        """

        new_mpl2_file_path, encoding = Utils.__convert_subtitle(srt_file_path, "srt", mpl2_file_path, "txt", "mpl2")
        Utils.remove_trailing_newlines(new_mpl2_file_path, encoding)

    @staticmethod
    def mpl22srt(mpl2_file_path: str, srt_file_path: Optional[str] = None) -> None:
        """Convert MPL2 subtitles to SubRip subtitles.

        Arguments:
            mpl2_file_path {string} -- The path to the MPL2 file.
            srt_file_path {string} -- The path to the SubRip file.
        """

        new_srt_file_path, encoding = Utils.__convert_subtitle(mpl2_file_path, "txt", srt_file_path, "srt", "srt")
        Utils.remove_trailing_newlines(new_srt_file_path, encoding)

    @staticmethod
    def srt2tmp(srt_file_path: str, tmp_file_path: Optional[str] = None) -> None:
        """Convert SubRip subtitles to TMP subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            tmp_file_path {string} -- The path to the TMP file.
        """

        new_tmp_file_path, encoding = Utils.__convert_subtitle(srt_file_path, "srt", tmp_file_path, "tmp", "tmp")
        Utils.remove_trailing_newlines(new_tmp_file_path, encoding)

    @staticmethod
    def tmp2srt(tmp_file_path: str, srt_file_path: Optional[str] = None) -> None:
        """Convert TMP subtitles to SubRip subtitles.

        Arguments:
            mpl2_file_path {string} -- The path to the TMP file.
            tmp_file_path {string} -- The path to the SubRip file.
        """

        new_srt_file_path, encoding = Utils.__convert_subtitle(tmp_file_path, "tmp", srt_file_path, "srt", "srt")
        Utils.remove_trailing_newlines(new_srt_file_path, encoding)

    @staticmethod
    def srt2sami(srt_file_path: str, sami_file_path: Optional[str] = None) -> None:
        """Convert SubRip subtitles to SAMI subtitles.

        Arguments:
            srt_file_path {string} -- The path to the SubRip file.
            sami_file_path {string} -- The path to the SAMI file.
        """

        file: Union[TextIO, BinaryIO]
        converter = CaptionConverter()
        encoding = Utils.detect_encoding(srt_file_path)
        with open(srt_file_path, "r", encoding=encoding) as file:
            converter.read(file.read(), SRTReader())
        if sami_file_path is None:
            sami_file_path = srt_file_path.replace(".srt", ".smi")
        with open(sami_file_path, "wb") as file:
            file.write(converter.write(SAMIWriter()).encode(encoding))

    @staticmethod
    def sami2srt(sami_file_path: str, srt_file_path: Optional[str] = None) -> None:
        """Convert SAMI subtitles to SubRip subtitles.

        Arguments:
            sami_file_path {string} -- The path to the SAMI file.
            srt_file_path {string} -- The path to the SubRip file.
        """

        file: Union[TextIO, BinaryIO]
        converter = CaptionConverter()
        encoding = Utils.detect_encoding(sami_file_path)
        with open(sami_file_path, "r", encoding=encoding) as file:
            converter.read(file.read(), SAMIReader())
        if srt_file_path is None:
            srt_file_path = sami_file_path.replace(".smi", ".srt")
        with open(srt_file_path, "wb") as file:
            file.write(converter.write(SRTWriter()).encode(encoding))
        Utils.remove_trailing_newlines(srt_file_path, encoding)

    @staticmethod
    def stl2srt(stl_file_path: str, srt_file_path: Optional[str] = None) -> None:
        """Convert EBU-STL subtitles to SubRip subtitles.

        Arguments:
            stl_file_path {string} -- The path to the EBU-STL file.
            srt_file_path {string} -- The path to the SubRip file.
        """

        encoding = Utils.detect_encoding(stl_file_path)
        stl = STL(stl_file_path, True)
        if srt_file_path is None:
            srt_file_path = stl_file_path.replace(".stl", ".srt")
        srt = SRT(srt_file_path)
        for sub in stl:
            (tci, tco, txt) = sub
            srt.write(tci, tco, txt)
        srt.file.close()
        stl.file.close()
        Utils.remove_trailing_newlines(srt_file_path, encoding)

    @staticmethod
    def extract_teletext_as_subtitle(ts_file_path: str, page_num: int, output_file_path: str, timeout_secs: int = 30) -> None:
        """Extract DVB Teletext from MPEG transport stream files and convert them into the output format.

        Arguments:
            ts_file_path {string} -- The path to the Transport Stream file.
            page_num {int} -- The page number for the Teletext
            output_file_path {string} -- The path to the output file.
            timeout_secs {int} -- The timeout in seconds on extraction {default: 30}.
        """

        command = "{0} -y -fix_sub_duration -txt_page {1} -txt_format text -i {2} {3}".format(Utils.FFMPEG_BIN, page_num, ts_file_path, output_file_path)
        timeout_msg = "Timeout on extracting Teletext from transport stream: {} on page: {}".format(ts_file_path, page_num)
        error_msg = "Cannot extract Teletext from transport stream: {} on page: {}".format(ts_file_path, page_num)

        def _callback(returncode: int, std_err: str) -> None:
            if returncode != 0:
                raise TerminalException(
                    "Cannot extract Teletext from transport stream: {} on page: {} with error {}".format(
                        ts_file_path, page_num, std_err
                    )
                )
            Utils.remove_trailing_newlines(output_file_path, None)

        Utils.__run_command(command, timeout_secs, timeout_msg, error_msg, _callback)

    @staticmethod
    def extract_matroska_subtitle(mkv_file_path: str, stream_index: int, output_file_path: str, timeout_secs: int = 30) -> None:
        """Extract subtitles from Matroska files and convert them into the output format.

        Arguments:
            mkv_file_path {string} -- The path to the Matroska file.
            stream_index {int} -- The index of the subtitle stream
            output_file_path {string} -- The path to the output file.
            timeout_secs {int} -- The timeout in seconds on extraction {default: 30}.
        """

        command = "{0} -y -i {1} -map 0:s:{2} {3}".format(Utils.FFMPEG_BIN, mkv_file_path, stream_index, output_file_path)
        timeout_msg = "Timeout on extracting the subtitle from file: {} with stream index: {}".format(mkv_file_path, stream_index)
        error_msg = "Cannot extract the subtitle from file: {} with stream index: {}".format(mkv_file_path, stream_index)

        def _callback(returncode: int, std_err: str) -> None:
            if returncode != 0:
                raise TerminalException(
                    "Cannot extract the subtitle from file: {} with stream index: {} with error {}".format(
                        mkv_file_path, stream_index, std_err
                    )
                )
            Utils.remove_trailing_newlines(output_file_path, None)
        Utils.__run_command(command, timeout_secs, timeout_msg, error_msg, _callback)

    @staticmethod
    def suppress_lib_logs() -> None:
        import os
        import logging
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
        logging.getLogger("tensorflow").disabled = True

    @staticmethod
    def remove_trailing_newlines(source_file_path: str, encoding: Optional[str], target_file_path: Optional[str] = None) -> None:
        with open(source_file_path, "r", encoding=encoding) as file:
            content = file.read()
        if target_file_path is not None:
            with open(target_file_path, "w", encoding=encoding) as file:
                file.write(content.rstrip())
        else:
            with open(source_file_path, "w", encoding=encoding) as file:
                file.write(content.rstrip())

    @staticmethod
    def download_file(remote_file_url: str, local_file_path: str) -> None:
        r = requests.get(remote_file_url, verify=True, stream=True)
        r.raw.decode_content = True
        with open(local_file_path, "wb") as file:
            shutil.copyfileobj(r.raw, file)

    @staticmethod
    def contains_embedded_subtitles(video_file_path: str, timeout_secs: int = 30) -> bool:
        """Detect if the input video contains embedded subtitles.

        Arguments:
            video_file_path {string} -- The path to the video file.
            timeout_secs {int} -- The timeout in seconds on extraction {default: 30}.

        Returns:
            bool -- True if the video contains embedded subtitles or False otherwise.
        """

        command = "{0} -y -i {1} -c copy -map 0:s -f null - -v 0 -hide_banner".format(Utils.FFMPEG_BIN, video_file_path)
        timeout_msg = "Timeout on detecting embedded subtitles from file: {}".format(video_file_path)
        error_msg = "Embedded subtitle detection failed for file: {}".format(video_file_path)

        def _callback(returncode: int, std_err: str) -> bool:
            return returncode == 0
        return Utils.__run_command(command, timeout_secs, timeout_msg, error_msg, _callback)

    @staticmethod
    def detect_encoding(subtitle_file_path: str) -> str:
        """Detect the encoding of the subtitle file.

        Arguments:
            subtitle_file_path {string} -- The path to the subtitle file.

        Returns:
            string -- The string represent the encoding
        """

        with open(subtitle_file_path, "rb") as file:
            # raw = b"".join([file.readline() for _ in range(10)])
            # Sampling with 10 lines did not work well enough for large subtitle files
            # and hence this less memory-efficient solution:
            raw = b"".join(file.readlines())

        detected = cchardet.detect(raw)
        detected = detected or {}
        return detected["encoding"] if "encoding" in detected else None

    @staticmethod
    def __convert_subtitle(source_file_path: str, source_ext: str, target_file_path: Optional[str], target_ext: str, format: str, frame_rate: Optional[float] = None) -> Tuple[str, str]:
        encoding = Utils.detect_encoding(source_file_path)
        subs = pysubs2.load(source_file_path, encoding=encoding)
        new_target_file_path = source_file_path.replace(".%s" % source_ext, ".%s" % target_ext) if target_file_path is None else target_file_path
        if frame_rate is None:
            subs.save(new_target_file_path, encoding=encoding, format_=format)
        else:
            subs.save(new_target_file_path, encoding=encoding, format_=format, fps=frame_rate)
        return new_target_file_path, encoding

    @staticmethod
    def __run_command(command: str, timeout_secs: int, timeout_msg: str, error_msg: str, callback: Callable[[int, str], Any]) -> Any:
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
                return callback(process.returncode, std_err)
            except subprocess.TimeoutExpired as te:
                process.kill()
                raise TerminalException(timeout_msg) from te
            except Exception as e:
                process.kill()
                if isinstance(e, TerminalException):
                    raise e
                else:
                    raise TerminalException(error_msg) from e
            finally:
                os.system("stty sane")
