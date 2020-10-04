import subprocess
import os
import threading
import traceback
import tempfile
import shutil
import atexit
import signal

from copy import deepcopy
from pysrt import SubRipFile
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

    AUDIO_FILE_EXTENSION = [".wav", ".aac"]

    __LOGGER = Logger().get_logger(__name__)
    __MIN_SECS_PER_WORD = 0.414  # 60 secs / 145 wpm
    __MIN_GAP_IN_SECS = (
        1  # minimum gap in seconds between consecutive subtitle during segmentation
    )
    __CMD_TIME_OUT = 180  # time out for subprocess

    atexit.register(clear_temp)
    signal.signal(signal.SIGTERM, clear_temp)
    signal.signal(signal.SIGINT, clear_temp)

    @staticmethod
    def extract_audio(video_file_path, decompress=False, freq=16000):
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
        # However the former will result in large temporary audio files saved on the disk.
        if decompress:
            assert freq is not None, "Frequency is needed for decompression"
            audio_file_path = "{0}/{1}{2}".format(
                TEMP_DIR_PATH, basename, MediaHelper.AUDIO_FILE_EXTENSION[0]
            )
        else:
            audio_file_path = "{0}/{1}{2}".format(
                TEMP_DIR_PATH, basename, MediaHelper.AUDIO_FILE_EXTENSION[1]
            )

        command = (
            "ffmpeg -y -xerror -i {0} -ac 2 -ar {1} -vn {2}".format(
                video_file_path, freq, audio_file_path
            )
            if decompress
            else "ffmpeg -y -xerror -i {0} -vn -acodec copy {1}".format(
                video_file_path, audio_file_path
            )
        )
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
                MediaHelper.__LOGGER.debug("[{}-{}] Running: {}".format(threading.current_thread().name, process.pid, command))
                _, std_err = process.communicate(timeout=MediaHelper.__CMD_TIME_OUT)
                MediaHelper.__LOGGER.debug("[{}-{}] {}".format(threading.current_thread().name, process.pid, std_err))
                if process.returncode != 0:
                    MediaHelper.__LOGGER.error("[{}-{}] Cannot extract audio from video: {}\n{}"
                                               .format(threading.current_thread().name, process.pid, video_file_path, std_err))
                    raise TerminalException(
                        "Cannot extract audio from video: {}".format(video_file_path)
                    )
                MediaHelper.__LOGGER.info(
                    "[{}-{}] Extracted audio file: {}".format(threading.current_thread().name, process.pid,
                                                              audio_file_path))
                return audio_file_path
            except subprocess.TimeoutExpired as te:
                MediaHelper.__LOGGER.error("Timeout on extracting audio from video: {}".format(video_file_path))
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
                MediaHelper.__LOGGER.error(
                    "[{}-{}] Extracting audio from video {} interrupted".format(
                        threading.current_thread().name, process.pid, video_file_path
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

    @staticmethod
    def get_duration_in_seconds(start, end):
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

    @staticmethod
    def extract_audio_from_start_to_end(audio_file_path, start, end=None):
        """Extract audio based on the start time and the end time and save it to a temporary file.

        Arguments:
            audio_file_path {string} -- The path of the audio file.
            start {string} -- The start time (e.g., 00:00:00,750).

        Keyword Arguments:
            end {string} -- The end time (e.g., 00:00:10,230) (default: {None}).

        Returns:
            tuple -- The file path to the extracted audio and its duration.
        """

        segment_duration = MediaHelper.get_duration_in_seconds(start, end)
        basename = os.path.basename(audio_file_path)
        filename, extension = os.path.splitext(basename)
        start = start.replace(",", ".")
        if end is not None:
            end = end.replace(",", ".")
        segment_path = "{0}/{1}_{2}_{3}{4}".format(TEMP_DIR_PATH, filename, str(start), str(end), extension)

        if end is not None:
            command = "ffmpeg -y -xerror -i {0} -ss {1} -to {2} -acodec copy {3}".format(
                audio_file_path, start, end, segment_path
            )
        else:
            command = "ffmpeg -y -xerror -i {0} -ss {1} -acodec copy {2}".format(
                audio_file_path, start, segment_path
            )
        with subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
        ) as process:
            MediaHelper.__LOGGER.debug("[{}-{}] Running: {}".format(threading.current_thread().name, process.pid, command))
            try:
                _, std_err = process.communicate(timeout=MediaHelper.__CMD_TIME_OUT)
                MediaHelper.__LOGGER.debug("[{}-{}] {}".format(threading.current_thread().name, process.pid, std_err))
                if process.returncode != 0:
                    MediaHelper.__LOGGER.error("[{}-{}] Cannot clip audio: {} Return Code: {}\n{}"
                                               .format(threading.current_thread().name, process.pid, audio_file_path, process.returncode, std_err))
                    raise TerminalException(
                        "Cannot clip audio: {} Return Code: {}".format(audio_file_path, process.returncode)
                    )
                MediaHelper.__LOGGER.info(
                    "[{}-{}] Extracted audio segment: {}".format(threading.current_thread().name, process.pid,
                                                                 segment_path))
                return segment_path, segment_duration
            except subprocess.TimeoutExpired as te:
                MediaHelper.__LOGGER.error(
                    "[{}-{}] Extracting {} timed out: {}\n{}".format(
                        threading.current_thread().name, process.pid, segment_path, str(te), "".join(traceback.format_stack())
                    )
                )
                if os.path.exists(segment_path):
                    os.remove(segment_path)
                raise TerminalException(
                    "Timeout on extracting audio from audio: {} after {} seconds".format(audio_file_path, MediaHelper.__CMD_TIME_OUT)
                ) from te
            except Exception as e:
                MediaHelper.__LOGGER.error(
                    "[{}-{}] Extracting {} failed: {}\n{}".format(
                        threading.current_thread().name, process.pid, segment_path, str(e), "".join(traceback.format_stack())
                    )
                )
                if os.path.exists(segment_path):
                    os.remove(segment_path)
                if isinstance(e, TerminalException):
                    raise e
                else:
                    raise TerminalException(
                        "Cannot clip audio: {}".format(audio_file_path)
                    ) from e
            except KeyboardInterrupt:
                MediaHelper.__LOGGER.error(
                    "[{}-{}] Extracting with start and end from {} interrupted".format(
                        threading.current_thread().name, process.pid, segment_path
                    )
                )
                if os.path.exists(segment_path):
                    os.remove(segment_path)
                process.send_signal(signal.SIGINT)
                raise TerminalException("Extracting with start and end from {} interrupted".format(segment_path))
            finally:
                process.kill()
                os.system("stty sane")

    @staticmethod
    def get_audio_segment_starts_and_ends(subs):
        """Group subtitle cues into larger segments in terms of silence gaps.

        Arguments:
            subs {list} -- A list of SupRip cues.

        Returns:
            tuple -- A list of start times, a list of end times and a list of grouped SubRip files.
        """

        local_subs = MediaHelper.__preprocess_subs(subs)

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
                if duration < MediaHelper.__MIN_SECS_PER_WORD:
                    combined.append(local_subs[i])
                    continue
                # Do not segment consecutive subtitles having little or no gap.
                gap = FeatureEmbedder.time_to_sec(
                    local_subs[i + 1].start
                ) - FeatureEmbedder.time_to_sec(local_subs[i].end)
                if (
                    local_subs[i].end == local_subs[i + 1].start
                    or gap < MediaHelper.__MIN_GAP_IN_SECS
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

    @staticmethod
    def get_frame_rate(file_path):
        """Extract the video frame rate. Will return 25 when input is audio

        Arguments:
            file_path {string} -- The input audiovisual file path.
        Returns:
            float -- The frame rate
        """

        with subprocess.Popen(
                "ffmpeg -i {} -t 00:00:10 -f null /dev/null".format(file_path).split(),
                shell=False,
                stderr=subprocess.PIPE,
                close_fds=True,
                universal_newlines=True,
                bufsize=1,
        ) as proc:
            with subprocess.Popen(
                    ['sed', '-n', "s/" + r".*, \(.*\) fp.*" + "/\\1/p"],
                    shell=False,
                    stdin=proc.stderr,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    close_fds=True,
                    universal_newlines=True,
                    bufsize=1,
            ) as process:
                try:
                    std_out, std_err = process.communicate(timeout=MediaHelper.__CMD_TIME_OUT)
                    if process.returncode != 0:
                        MediaHelper.__LOGGER.error("[{}-{}] Cannot extract the frame rate from video: {}\n{}"
                                                   .format(threading.current_thread().name, process.pid, file_path, std_err))
                        raise NoFrameRateException(
                            "Cannot extract the frame rate from video: {}".format(file_path)
                        )
                    fps = float(std_out.split("\n")[0])
                    MediaHelper.__LOGGER.info("[{}-{}] Extracted frame rate: {} fps".format(threading.current_thread().name, process.pid, fps))
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
                    MediaHelper.__LOGGER.error(
                        "[{}-{}] Extracting frame rate from video {} interrupted".format(
                            threading.current_thread().name, process.pid, file_path
                        )
                    )
                    process.send_signal(signal.SIGINT)
                    proc.send_signal(signal.SIGINT)
                    raise TerminalException("Extracting frame rate from video {} interrupted".format(file_path))
                finally:
                    process.kill()
                    proc.kill()
                    os.system("stty sane")

    @staticmethod
    def __preprocess_subs(subs):
        local_subs = deepcopy(subs)

        # Preprocess overlapping subtitles
        for i in range(len(local_subs)):
            if i != 0 and local_subs[i].start < local_subs[i - 1].end:
                MediaHelper.__LOGGER.warning("Found overlapping subtitle cues and the earlier one's duration will be shortened.")
                local_subs[i - 1].end = local_subs[i].start

        return local_subs
