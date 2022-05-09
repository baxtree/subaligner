import subprocess
import os
import threading
import traceback
import tempfile
import shutil
import atexit
import signal
import shlex

from typing import Optional, Tuple, List
from copy import deepcopy
from pysrt import SubRipFile, SubRipItem
from decimal import Decimal
from .embedder import FeatureEmbedder
from .exception import TerminalException
from .exception import NoFrameRateException
from .logger import Logger

TEMP_DIR_PATH = tempfile.mkdtemp()


def clear_temp(*_):
    if os.path.isdir(TEMP_DIR_PATH):
        shutil.rmtree(TEMP_DIR_PATH)


class MediaHelper(object):
    """ Utility for processing media assets including audio, video and
    subtitle files.
    """

    FFMPEG_BIN = os.getenv("FFMPEG_PATH") or os.getenv("ffmpeg_path") or "ffmpeg"

    AUDIO_FILE_EXTENSION = [".wav", ".aac"]

    __MIN_SECS_PER_WORD = 0.414  # 60 secs / 145 wpm
    __MIN_GAP_IN_SECS = (
        1  # minimum gap in seconds between consecutive subtitle during segmentation
    )
    __CMD_TIME_OUT = 180  # time out for subprocess

    atexit.register(clear_temp)
    signal.signal(signal.SIGTERM, clear_temp)

    def __init__(self):
        self.__LOGGER = Logger().get_logger(__name__)

    def extract_audio(self, video_file_path, decompress: bool = False, freq: int = 16000) -> str:
        """Extract audio track from the video file and save it to a WAV file.

        Arguments:
            video_file_path {string} -- The input video file path.
        Keyword Arguments:
            decompress {bool} -- Extract WAV if True otherwise extract AAC (default: {False}).
            freq {int} -- The audio sample frequency (default: {16000}).
        Returns:
            string -- The file path of the extracted audio.
        """

        basename = os.path.basename(video_file_path)

        # Using WAV for training or prediction is faster than using AAC.
        # However the former will result in larger temporary audio files saved on the disk.
        if decompress:
            assert freq is not None, "Frequency is needed for decompression"
            audio_file_path = os.path.join(
                TEMP_DIR_PATH, f"{basename}{self.AUDIO_FILE_EXTENSION[0]}"
            )
        else:
            audio_file_path = os.path.join(
                TEMP_DIR_PATH, f"{basename}{self.AUDIO_FILE_EXTENSION[1]}"
            )

        command = (
            "{0} -y -xerror -i '{1}' -ac 2 -ar {2} -vn '{3}'".format(
                self.FFMPEG_BIN, video_file_path, freq, audio_file_path
            )
            if decompress
            else "{0} -y -xerror -i '{1}' -vn -acodec copy '{2}'".format(
                self.FFMPEG_BIN, video_file_path, audio_file_path
            )
        )
        with subprocess.Popen(
            shlex.split(command),
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            universal_newlines=True,
            bufsize=1,
        ) as process:
            try:
                self.__LOGGER.debug("[{}] Running: {}".format(process.pid, command))
                _, std_err = process.communicate(timeout=self.__CMD_TIME_OUT)
                self.__LOGGER.debug("[{}] {}".format(process.pid, std_err))
                if process.returncode != 0:
                    self.__LOGGER.error("[{}] Cannot extract audio from video: {}\n{}"
                                        .format(process.pid, video_file_path, std_err))
                    raise TerminalException(
                        "Cannot extract audio from video: {}".format(video_file_path)
                    )
                self.__LOGGER.info(
                    "[{}] Extracted audio file: {}".format(process.pid, audio_file_path))
                return audio_file_path
            except subprocess.TimeoutExpired as te:
                self.__LOGGER.error("Timeout on extracting audio from video: {}".format(video_file_path))
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
                raise TerminalException(
                    "Timeout on extracting audio from video: {}".format(video_file_path)
                ) from te
            except Exception as e:
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
                if isinstance(e, TerminalException):
                    raise e
                else:
                    raise TerminalException(
                        "Cannot extract audio from video: {}".format(video_file_path)
                    ) from e
            except KeyboardInterrupt:
                self.__LOGGER.error(
                    "[{}] Extracting audio from video {} interrupted".format(
                        process.pid, video_file_path
                    )
                )
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
                process.send_signal(signal.SIGINT)
                raise TerminalException(
                    "Extracting audio from video {} interrupted".format(video_file_path)
                )
            finally:
                process.kill()
                os.system("stty sane")

    def get_duration_in_seconds(self, start: Optional[str], end: Optional[str]) -> Optional[float]:
        """Get the duration in seconds between a start time and an end time.

        Arguments:
            start {string} -- The start time (e.g., 00:00:00,750).
            end {string} -- The end time (e.g., 00:00:10,230).

        Returns:
            float -- The duration in seconds.
        """

        if start is None:
            start = "00:00:00,000"
        if end is None:
            return None
        start = start.replace(",", ".")
        end = end.replace(",", ".")
        start_h, start_m, start_s = map(Decimal, start.split(":"))
        end_h, end_m, end_s = map(Decimal, end.split(":"))
        return float(
            (end_h * 3600 + end_m * 60 + end_s)
            - (start_h * 3600 + start_m * 60 + start_s)
        )

    def extract_audio_from_start_to_end(self, audio_file_path: str, start: str, end: Optional[str] = None) -> Tuple[str, Optional[float]]:
        """Extract audio based on the start time and the end time and save it to a temporary file.

        Arguments:
            audio_file_path {string} -- The path of the audio file.
            start {string} -- The start time (e.g., 00:00:00,750).

        Keyword Arguments:
            end {string} -- The end time (e.g., 00:00:10,230) (default: {None}).

        Returns:
            tuple -- The file path to the extracted audio and its duration.
        """
        segment_duration = self.get_duration_in_seconds(start, end)
        basename = os.path.basename(audio_file_path)
        filename, extension = os.path.splitext(basename)
        start = start.replace(",", ".")
        if end is not None:
            end = end.replace(",", ".")
        segment_path = os.path.join(TEMP_DIR_PATH, f"{filename}_{str(start)}_{str(end)}{extension}")

        if end is not None:
            duration = self.get_duration_in_seconds(start, end)
            command = "{0} -y -xerror -i '{1}' -ss {2} -t {3} -acodec copy '{4}'".format(
                self.FFMPEG_BIN, audio_file_path, start, duration, segment_path
            )
        else:
            command = "{0} -y -xerror -i '{1}' -ss {2} -acodec copy '{3}'".format(
                self.FFMPEG_BIN, audio_file_path, start, segment_path
            )
        with subprocess.Popen(
            shlex.split(command),
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        ) as process:
            self.__LOGGER.debug("[{}] Running: {}".format(process.pid, command))
            try:
                _, std_err = process.communicate(timeout=self.__CMD_TIME_OUT)
                self.__LOGGER.debug("[{}] {}".format(process.pid, std_err))
                if process.returncode != 0:
                    self.__LOGGER.error("[{}] Cannot clip audio: {} Return Code: {}\n{}"
                                        .format(process.pid, audio_file_path, process.returncode, std_err))
                    raise TerminalException(
                        "Cannot clip audio: {} Return Code: {}".format(audio_file_path, process.returncode)
                    )
                self.__LOGGER.info(
                    "[{}] Extracted audio segment: {}".format(process.pid, segment_path))
                return segment_path, segment_duration
            except subprocess.TimeoutExpired as e:
                self.__LOGGER.error(
                    "[{}] Extracting {} timed out: {}\n{}".format(
                        process.pid, segment_path, str(e), "\n".join(traceback.format_stack())
                    )
                )
                traceback.print_tb(e.__traceback__)
                if os.path.exists(segment_path):
                    os.remove(segment_path)
                raise TerminalException(
                    "Timeout on extracting audio from audio: {} after {} seconds".format(audio_file_path, self.__CMD_TIME_OUT)
                ) from e
            except Exception as e:
                self.__LOGGER.error(
                    "[{}] Extracting {} failed: {}\n{}".format(
                        process.pid, segment_path, str(e), "\n".join(traceback.format_stack())
                    )
                )
                traceback.print_tb(e.__traceback__)
                if os.path.exists(segment_path):
                    os.remove(segment_path)
                if isinstance(e, TerminalException):
                    raise e
                else:
                    raise TerminalException(
                        "Cannot clip audio: {}".format(audio_file_path)
                    ) from e
            except KeyboardInterrupt:
                self.__LOGGER.error(
                    "[{}] Extracting with start and end from {} interrupted".format(
                        process.pid, segment_path
                    )
                )
                if os.path.exists(segment_path):
                    os.remove(segment_path)
                process.send_signal(signal.SIGINT)
                raise TerminalException("Extracting with start and end from {} interrupted".format(segment_path))
            finally:
                process.kill()
                os.system("stty sane")

    def get_audio_segment_starts_and_ends(self, subs: List[SubRipItem]) -> Tuple[List[str], List[str], List[SubRipFile]]:
        """Group subtitle cues into larger segments in terms of silence gaps.

        Arguments:
            subs {list} -- A list of SupRip cues.

        Returns:
            tuple -- A list of start times, a list of end times and a list of grouped SubRip files.
        """

        local_subs = self.__preprocess_subs(subs)

        segment_starts = []
        segment_ends = []
        combined = []
        new_subs = []
        current_start = str(local_subs[0].start)

        for i in range(len(local_subs)):
            if i == len(local_subs) - 1:
                combined.append(local_subs[i])
                segment_starts.append(current_start)
                segment_ends.append(str(local_subs[i].end))
                new_subs.append(SubRipFile(combined))
                del combined[:]
            else:
                # Do not segment when the subtitle is too short
                duration = FeatureEmbedder.time_to_sec(
                    local_subs[i].end
                ) - FeatureEmbedder.time_to_sec(local_subs[i].start)
                if duration < self.__MIN_SECS_PER_WORD:
                    combined.append(local_subs[i])
                    continue
                # Do not segment consecutive subtitles having little or no gap.
                gap = FeatureEmbedder.time_to_sec(
                    local_subs[i + 1].start
                ) - FeatureEmbedder.time_to_sec(local_subs[i].end)
                if (
                    local_subs[i].end == local_subs[i + 1].start
                    or gap < self.__MIN_GAP_IN_SECS
                ):
                    combined.append(local_subs[i])
                    continue
                combined.append(local_subs[i])
                # The start time is set to last cue's end time
                segment_starts.append(current_start)
                # The end time cannot be set to next cue's start time due to possible overlay
                segment_ends.append(str(local_subs[i].end))
                current_start = str(local_subs[i].end)
                new_subs.append(SubRipFile(combined))
                del combined[:]
        return segment_starts, segment_ends, new_subs

    def get_frame_rate(self, file_path: str) -> float:
        """Extract the video frame rate. Will return 25 when input is audio

        Arguments:
            file_path {string} -- The input audiovisual file path.
        Returns:
            float -- The frame rate
        """

        discarded = "NUL:" if os.name == "nt" else "/dev/null"

        with subprocess.Popen(
                shlex.split("{0} -i '{1}' -t 00:00:10 -f null {2}".format(self.FFMPEG_BIN, file_path, discarded)),
                shell=False,
                stderr=subprocess.PIPE,
                close_fds=True,
                universal_newlines=True,
                bufsize=1,
        ) as proc:
            with subprocess.Popen(
                    ['grep', '-Eo', r"[0-9]{1,3}(\.[0-9]{1,3})?\sfps,"],
                    shell=False,
                    stdin=proc.stderr,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    close_fds=True,
                    universal_newlines=True,
                    bufsize=1,
            ) as process:
                try:
                    std_out, std_err = process.communicate(timeout=self.__CMD_TIME_OUT)
                    if process.returncode != 0:
                        self.__LOGGER.warning("[{}] Cannot extract the frame rate from video: {}\n{}".format(process.pid, file_path, std_err))
                        raise NoFrameRateException(
                            "Cannot extract the frame rate from video: {}".format(file_path)
                        )
                    fps = float(std_out.split(" ")[0])
                    # ffmpeg uses two decimal places so be this hack
                    fps = fps if fps != 23.98 else 23.976
                    self.__LOGGER.info("[{}] Extracted frame rate: {} fps".format(process.pid, fps))
                    return fps
                except subprocess.TimeoutExpired as te:
                    raise NoFrameRateException(
                        "Timeout on extracting the frame rate from video: {}".format(file_path)
                    ) from te
                except Exception as e:
                    if isinstance(e, TerminalException):
                        raise e
                    else:
                        raise NoFrameRateException(
                            "Cannot extract the frame rate from video: {}".format(file_path)
                        ) from e
                except KeyboardInterrupt:
                    self.__LOGGER.error(
                        "[{}] Extracting frame rate from video {} interrupted".format(
                            process.pid, file_path
                        )
                    )
                    process.send_signal(signal.SIGINT)
                    proc.send_signal(signal.SIGINT)
                    raise TerminalException("Extracting frame rate from video {} interrupted".format(file_path))
                finally:
                    process.kill()
                    proc.kill()
                    os.system("stty sane")

    def refragment_with_min_duration(self, subs: List[SubRipItem], minimum_segment_duration: float) -> List[SubRipItem]:
        """Re-fragment a list of subtitle cues into new cues each of spans a minimum duration

        Arguments:
            subs {list} -- A list of SupRip cues.
            minimum_segment_duration {float} -- The minimum duration in seconds for each output subtitle cue.
        Returns:
            list -- A list of new SupRip cues after fragmentation.
        """
        new_segment = []
        new_segment_index = 0
        new_segment_duration = 0.0
        new_segment_text = ""
        new_subs = []
        for sub in subs:
            if minimum_segment_duration > new_segment_duration:
                new_segment.append(sub)
                new_segment_duration += self.get_duration_in_seconds(str(sub.start), str(sub.end)) or 0.0
                new_segment_text += "{}\n".format(sub.text)
            else:
                concatenated_item = SubRipItem(new_segment_index, new_segment[0].start, new_segment[-1].end,
                                               new_segment_text, new_segment[0].position)
                new_subs.append(concatenated_item)
                new_segment_index += 1
                new_segment = [sub]
                new_segment_duration = self.get_duration_in_seconds(str(sub.start), str(sub.end)) or 0.0
                new_segment_text = "{}\n".format(sub.text)
        if new_segment:
            concatenated_item = SubRipItem(new_segment_index, new_segment[0].start, new_segment[-1].end,
                                           new_segment_text, new_segment[0].position)
            new_subs.append(concatenated_item)
        return new_subs

    def __preprocess_subs(self, subs: List[SubRipItem]) -> List[SubRipItem]:
        local_subs = deepcopy(subs)

        # Preprocess overlapping subtitles
        for i in range(len(local_subs)):
            if i != 0 and local_subs[i].start < local_subs[i - 1].end:
                self.__LOGGER.warning("Found overlapping subtitle cues and the earlier one's duration will be shortened.")
                local_subs[i - 1].end = local_subs[i].start

        return local_subs
