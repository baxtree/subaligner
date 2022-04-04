import os
import datetime
import traceback
import threading
import concurrent.futures
import gc
import math
import logging
import numpy as np
import multiprocessing as mp
from typing import Tuple, List, Optional, Dict, Any, Iterable, Union
from pysrt import SubRipTime, SubRipItem, SubRipFile
from sklearn.metrics import log_loss
from copy import deepcopy
from .network import Network
from .embedder import FeatureEmbedder
from .media_helper import MediaHelper
from .singleton import Singleton
from .subtitle import Subtitle
from .hyperparameters import Hyperparameters
from .exception import TerminalException
from .exception import NoFrameRateException
from .logger import Logger


class Predictor(metaclass=Singleton):
    """ Predictor for working out the time to shift subtitles
    """
    __MAX_SHIFT_IN_SECS = (
        100
    )
    __MAX_CHARS_PER_SEC = (
        50
    )  # Average 0.3 word per sec multiplies average 6 characters per word
    __MAX_HEAD_ROOM = 20000  # Maximum duration without subtitle (10 minutes)

    __SEGMENT_PREDICTION_TIMEOUT = 60  # Maximum waiting time in seconds when predicting each segment

    __THREAD_QUEUE_SIZE = 8
    __THREAD_NUMBER = 4

    def __init__(self, **kwargs) -> None:
        """Feature predictor initialiser.

            Keyword Arguments:
                n_mfcc {int} -- The number of MFCC components (default: {13}).
                frequency {float} -- The sample rate  (default: {16000}).
                hop_len {int} -- The number of samples per frame (default: {512}).
                step_sample {float} -- The space (in seconds) between the begining of each sample (default: 1s / 25 FPS = 0.04s).
                len_sample {float} -- The length in seconds for the input samples (default: {0.075}).
        """

        self.__feature_embedder = FeatureEmbedder(**kwargs)
        self.__LOGGER = Logger().get_logger(__name__)
        self.__media_helper = MediaHelper()

    def predict_single_pass(
            self,
            video_file_path: str,
            subtitle_file_path: str,
            weights_dir: str = os.path.join(os.path.dirname(__file__), "models", "training", "weights"),
    ) -> Tuple[List[SubRipItem], str, Union[np.ndarray, List[float]], Optional[float]]:
        """Predict time to shift with single pass

            Arguments:
                video_file_path {string} -- The input video file path.
                subtitle_file_path {string} -- The path to the subtitle file.
                weights_dir {string} -- The the model weights directory.

            Returns:
                tuple -- The shifted subtitles, the audio file path and the voice probabilities of the original audio.
        """

        weights_file_path = self.__get_weights_path(weights_dir)
        audio_file_path = ""
        frame_rate = None
        try:
            subs, audio_file_path, voice_probabilities = self.__predict(
                video_file_path, subtitle_file_path, weights_file_path
            )
            try:
                frame_rate = self.__media_helper.get_frame_rate(video_file_path)
                self.__feature_embedder.step_sample = 1 / frame_rate
                self.__on_frame_timecodes(subs)
            except NoFrameRateException:
                self.__LOGGER.warning("Cannot detect the frame rate for %s" % video_file_path)
            return subs, audio_file_path, voice_probabilities, frame_rate
        finally:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)

    def predict_dual_pass(
            self,
            video_file_path: str,
            subtitle_file_path: str,
            weights_dir: str = os.path.join(os.path.dirname(__file__), "models", "training", "weights"),
            stretch: bool = False,
            stretch_in_lang: str = "eng",
            exit_segfail: bool = False,
    ) -> Tuple[List[SubRipItem], List[SubRipItem], Union[np.ndarray, List[float]], Optional[float]]:
        """Predict time to shift with single pass

            Arguments:
            video_file_path {string} -- The input video file path.
            subtitle_file_path {string} -- The path to the subtitle file.
            weights_dir {string} -- The the model weights directory.
            stretch {bool} -- True to stretch the subtitle segments (default: {False})
            stretch_in_lang {str} -- The language used for stretching subtitles (default: {"eng"}).
            exit_segfail {bool} -- True to exit on any segment alignment failures (default: {False})

            Returns:
            tuple -- The shifted subtitles, the globally shifted subtitles and the voice probabilities of the original audio.
        """

        weights_file_path = self.__get_weights_path(weights_dir)
        audio_file_path = ""
        frame_rate = None
        try:
            subs, audio_file_path, voice_probabilities = self.__predict(
                video_file_path, subtitle_file_path, weights_file_path
            )
            new_subs = self.__predict_2nd_pass(
                audio_file_path,
                subs,
                weights_file_path=weights_file_path,
                stretch=stretch,
                stretch_in_lang=stretch_in_lang,
                exit_segfail=exit_segfail,
            )
            try:
                frame_rate = self.__media_helper.get_frame_rate(video_file_path)
                self.__feature_embedder.step_sample = 1 / frame_rate
                self.__on_frame_timecodes(new_subs)
            except NoFrameRateException:
                self.__LOGGER.warning("Cannot detect the frame rate for %s" % video_file_path)
            self.__LOGGER.debug("Aligned segments generated")
            return new_subs, subs, voice_probabilities, frame_rate
        finally:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)

    def predict_plain_text(self, video_file_path: str, subtitle_file_path: str, stretch_in_lang: str = "eng") -> Tuple:
        """Predict time to shift with plain texts

            Arguments:
            video_file_path {string} -- The input video file path.
            subtitle_file_path {string} -- The path to the subtitle file.
            stretch_in_lang {str} -- The language used for stretching subtitles (default: {"eng"}).

            Returns:
            tuple -- The shifted subtitles, the audio file path (None) and the voice probabilities of the original audio (None).
        """
        from aeneas.executetask import ExecuteTask
        from aeneas.task import Task
        from aeneas.runtimeconfiguration import RuntimeConfiguration
        from aeneas.logger import Logger as AeneasLogger

        t = datetime.datetime.now()
        audio_file_path = self.__media_helper.extract_audio(
            video_file_path, True, 16000
        )
        self.__LOGGER.debug(
            "[{}] Audio extracted after {}".format(
                os.getpid(), str(datetime.datetime.now() - t)
            )
        )

        root, _ = os.path.splitext(audio_file_path)

        # Initialise a DTW alignment task
        task_config_string = (
            "task_language={}|os_task_file_format=srt|is_text_type=subtitles".format(stretch_in_lang)
        )
        runtime_config_string = "dtw_algorithm=stripe"  # stripe or exact
        task = Task(config_string=task_config_string)

        try:
            task.audio_file_path_absolute = audio_file_path
            task.text_file_path_absolute = subtitle_file_path
            task.sync_map_file_path_absolute = "{}.srt".format(root)

            tee = False if self.__LOGGER.level == getattr(logging, 'DEBUG') else True

            # Execute the task
            ExecuteTask(
                task=task,
                rconf=RuntimeConfiguration(config_string=runtime_config_string),
                logger=AeneasLogger(tee=tee),
            ).execute()

            # Output new subtitle segment to a file
            task.output_sync_map_file()

            # Load the above subtitle segment
            adjusted_subs = Subtitle.load(
                task.sync_map_file_path_absolute
            ).subs

            frame_rate = None
            try:
                frame_rate = self.__media_helper.get_frame_rate(video_file_path)
                self.__feature_embedder.step_sample = 1 / frame_rate
                self.__on_frame_timecodes(adjusted_subs)
            except NoFrameRateException:
                self.__LOGGER.warning("Cannot detect the frame rate for %s" % video_file_path)

            return adjusted_subs, None, None, frame_rate
        except KeyboardInterrupt:
            raise TerminalException("Subtitle stretch interrupted by the user")
        finally:
            # Housekeep intermediate files
            if task.audio_file_path_absolute is not None and os.path.exists(
                    task.audio_file_path_absolute
            ):
                os.remove(task.audio_file_path_absolute)
            if task.sync_map_file_path_absolute is not None and os.path.exists(task.sync_map_file_path_absolute):
                os.remove(task.sync_map_file_path_absolute)

    def get_log_loss(self, voice_probabilities: np.ndarray, subs: List[SubRipItem]) -> float:
        """Returns a single loss value on voice prediction

            Arguments:
                voice_probabilities {list} -- A list of probabilities of audio chunks being speech.
                subs {list} -- A list of subtitle segments.

                Returns:
                    float -- The loss value.
        """

        subtitle_mask = Predictor.__get_subtitle_mask(self, subs)
        if len(subtitle_mask) == 0:
            raise TerminalException("Subtitle is empty")

        # Adjust the voice duration when it is shorter than the subtitle duration
        # so we can have room to shift the subtitle back and forth based on losses.
        head_room = len(voice_probabilities) - len(subtitle_mask)
        if head_room < 0:
            self.__LOGGER.warning("Audio duration is shorter than the subtitle duration")
            local_vp = np.vstack(
                [
                    voice_probabilities,
                    [np.zeros(voice_probabilities.shape[1])] * (-head_room * 5),
                ]
            )
            result = log_loss(
                subtitle_mask, local_vp[: len(subtitle_mask)], labels=[0, 1]
            )
        else:
            result = log_loss(
                subtitle_mask, voice_probabilities[: len(subtitle_mask)], labels=[0, 1]
            )

        self.__LOGGER.debug("Log loss: {}".format(result))
        return result

    def get_min_log_loss_and_index(self, voice_probabilities: np.ndarray, subs: SubRipFile) -> Tuple[float, int]:
        """Returns the minimum loss value and its shift position after going through all possible shifts.
            Arguments:
                voice_probabilities {list} -- A list of probabilities of audio chunks being speech.
                subs {list} -- A list of subtitle segments.
            Returns:
                tuple -- The minimum loss value and its position.
        """

        local_subs = deepcopy(subs)

        local_subs.shift(seconds=-FeatureEmbedder.time_to_sec(subs[0].start))
        subtitle_mask = Predictor.__get_subtitle_mask(self, local_subs)
        if len(subtitle_mask) == 0:
            raise TerminalException("Subtitle is empty")

        # Adjust the voice duration when it is shorter than the subtitle duration
        # so we can have room to shift the subtitle back and forth based on losses.
        head_room = len(voice_probabilities) - len(subtitle_mask)
        self.__LOGGER.debug("head room: {}".format(head_room))
        if head_room < 0:
            local_vp = np.vstack(
                [
                    voice_probabilities,
                    [np.zeros(voice_probabilities.shape[1])] * (-head_room * 5),
                ]
            )
        else:
            local_vp = voice_probabilities
        head_room = len(local_vp) - len(subtitle_mask)
        if head_room > Predictor.__MAX_HEAD_ROOM:
            self.__LOGGER.error("head room: {}".format(head_room))
            raise TerminalException(
                "Maximum head room reached due to the suspicious audio or subtitle duration"
            )

        log_losses = []
        self.__LOGGER.debug(
            "Start calculating {} log loss(es)...".format(head_room)
        )
        for i in np.arange(0, head_room):
            log_losses.append(
                log_loss(
                    subtitle_mask,
                    local_vp[i:i + len(subtitle_mask)],
                    labels=[0, 1],
                )
            )
        if log_losses:
            min_log_loss = min(log_losses)
            min_log_loss_idx = log_losses.index(min_log_loss)
        else:
            min_log_loss = None
            min_log_loss_idx = 0

        del local_vp
        del log_losses
        gc.collect()

        return min_log_loss, min_log_loss_idx

    @staticmethod
    def _predict_in_multiprocesses(
            self,
            batch_idx: List[int],
            segment_starts: List[str],
            segment_ends: List[str],
            weights_file_path: str,
            audio_file_path: str,
            subs: List[SubRipItem],
            subs_copy: List[SubRipItem],
            stretch: bool,
            stretch_in_lang: str,
            exit_segfail: bool,
    ) -> List[SubRipItem]:
        subs_list = []
        with _ThreadPoolExecutorLocal(
                queue_size=Predictor.__THREAD_QUEUE_SIZE,
                max_workers=Predictor.__THREAD_NUMBER
        ) as executor:
            lock = threading.RLock()
            network = self.__initialise_network(os.path.dirname(weights_file_path), self.__LOGGER)
            futures = []
            for segment_index in batch_idx:
                futures.append(
                    executor.submit(
                        Predictor._predict_in_multithreads,
                        self,
                        segment_index,
                        segment_starts,
                        segment_ends,
                        weights_file_path,
                        audio_file_path,
                        subs,
                        subs_copy,
                        stretch,
                        stretch_in_lang,
                        exit_segfail,
                        lock,
                        network
                    )
                )
            for i, future in enumerate(futures):
                try:
                    new_subs = future.result(timeout=Predictor.__SEGMENT_PREDICTION_TIMEOUT)
                except concurrent.futures.TimeoutError as e:
                    self.__cancel_futures(futures[i:], Predictor.__SEGMENT_PREDICTION_TIMEOUT)
                    message = "Segment alignment timed out after {} seconds".format(Predictor.__SEGMENT_PREDICTION_TIMEOUT)
                    self.__LOGGER.error(message)
                    raise TerminalException(message) from e
                except Exception as e:
                    self.__cancel_futures(futures[i:], Predictor.__SEGMENT_PREDICTION_TIMEOUT)
                    message = "Exception on segment alignment: {}\n{}".format(str(e), "".join(traceback.format_stack()))
                    self.__LOGGER.error(e, exc_info=True, stack_info=True)
                    traceback.print_tb(e.__traceback__)
                    if isinstance(e, TerminalException):
                        raise e
                    else:
                        raise TerminalException(message) from e
                except KeyboardInterrupt:
                    self.__cancel_futures(futures[i:], Predictor.__SEGMENT_PREDICTION_TIMEOUT)
                    raise TerminalException("Segment alignment interrupted by the user")
                else:
                    self.__LOGGER.debug("Segment aligned")
                    subs_list.extend(new_subs)
        return subs_list

    @staticmethod
    def _predict_in_multithreads(
            self,
            segment_index: int,
            segment_starts: List[str],
            segment_ends: List[str],
            weights_file_path: str,
            audio_file_path: str,
            subs: List[SubRipItem],
            subs_copy: List[SubRipItem],
            stretch: bool,
            stretch_in_lang: str,
            exit_segfail: bool,
            lock: threading.RLock,
            network: Network
    ) -> List[SubRipItem]:
        segment_path = ""
        try:
            if segment_index == (len(segment_starts) - 1):
                segment_path, segment_duration = self.__media_helper.extract_audio_from_start_to_end(
                    audio_file_path, segment_starts[segment_index], None
                )
            else:
                segment_path, segment_duration = self.__media_helper.extract_audio_from_start_to_end(
                    audio_file_path,
                    segment_starts[segment_index],
                    segment_ends[segment_index],
                )
            subtitle_duration = FeatureEmbedder.time_to_sec(
                subs[segment_index][len(subs[segment_index]) - 1].end
            ) - FeatureEmbedder.time_to_sec(subs[segment_index][0].start)
            if segment_duration is None:
                max_shift_secs = None
            else:
                max_shift_secs = segment_duration - subtitle_duration

            if segment_index == 0:
                previous_gap = 0.0
            else:
                previous_gap = FeatureEmbedder.time_to_sec(subs[segment_index][0].start) - FeatureEmbedder.time_to_sec(
                    subs[segment_index - 1][len(subs[segment_index - 1]) - 1].end
                )
            subs_new, _, voice_probabilities = self.__predict(
                video_file_path=None,
                subtitle_file_path=None,
                weights_file_path=weights_file_path,
                audio_file_path=segment_path,
                subtitles=subs_copy[segment_index],
                max_shift_secs=max_shift_secs,
                previous_gap=previous_gap,
                lock=lock,
                network=network
            )
            del voice_probabilities
            gc.collect()

            if stretch:
                subs_new = self.__adjust_durations(subs_new, audio_file_path, stretch_in_lang, lock)
                self.__LOGGER.info("[{}] Segment {} stretched".format(os.getpid(), segment_index))
            return subs_new
        except Exception as e:
            self.__LOGGER.error(
                "[{}] Alignment failed for segment {}: {}\n{}".format(
                    os.getpid(), segment_index, str(e), "".join(traceback.format_stack())
                )
            )
            traceback.print_tb(e.__traceback__)
            if exit_segfail:
                raise TerminalException("At least one of the segments failed on alignment. Exiting...") from e
            return subs[segment_index]
        finally:
            # Housekeep intermediate files
            if os.path.exists(segment_path):
                os.remove(segment_path)

    @staticmethod
    def __minibatch(total: int, batch_size: int) -> Iterable[List[int]]:
        batch: List = []
        for i in range(total):
            if len(batch) == batch_size:
                yield batch
                batch = []
            batch.append(i)
        if batch:
            yield batch

    @staticmethod
    def __initialise_network(weights_dir: str, logger: logging.Logger) -> Network:
        model_dir = weights_dir.replace("/weights", "/model").replace("\\weights", "\\model")
        config_dir = weights_dir.replace("/weights", "/config").replace("\\weights", "\\config")
        files = os.listdir(model_dir)
        model_files = [
            file
            for file in files
            if file.startswith("model")
        ]
        files = os.listdir(config_dir)
        hyperparams_files = [
            file
            for file in files
            if file.startswith("hyperparameters")
        ]

        if not model_files:
            raise TerminalException(
                "Cannot find model files at {}".format(weights_dir)
            )

        logger.debug("model files: {}".format(model_files))
        logger.debug("config files: {}".format(hyperparams_files))

        # Get the first file from the file lists
        model_path = os.path.join(model_dir, model_files[0])
        hyperparams_path = os.path.join(config_dir, hyperparams_files[0])
        hyperparams = Hyperparameters.from_file(hyperparams_path)
        return Network.get_from_model(model_path, hyperparams)

    @staticmethod
    def __get_weights_path(weights_dir: str) -> str:
        files = os.listdir(weights_dir)
        weights_files = [
            file
            for file in files
            if file.startswith("weights")
        ]

        if not weights_files:
            raise TerminalException(
                "Cannot find weights files at {}".format(weights_dir)
            )

        # Get the first file from the file lists
        weights_path = os.path.join(weights_dir, weights_files[0])

        return os.path.abspath(weights_path)

    def __predict_2nd_pass(self, audio_file_path: str, subs: List[SubRipItem], weights_file_path: str, stretch: bool, stretch_in_lang: str, exit_segfail: bool) -> List[SubRipItem]:
        """This function uses divide and conquer to align partial subtitle with partial video.

        Arguments:
            audio_file_path {string} -- The file path of the original audio.
            subs {list} -- A list of SubRip files.
            weights_file_path {string} --  The file path of the weights file.
            stretch {bool} -- True to stretch the subtitle segments.
            stretch_in_lang {str} -- The language used for stretching subtitles.
            exit_segfail {bool} -- True to exit on any segment alignment failures.
        """

        segment_starts, segment_ends, subs = self.__media_helper.get_audio_segment_starts_and_ends(subs)
        subs_copy = deepcopy(subs)

        for index, sub in enumerate(subs):
            self.__LOGGER.debug(
                "Subtitle chunk #{0}: start time: {1} ------> end time: {2}".format(
                    index, sub[0].start, sub[len(sub) - 1].end
                )
            )

        assert len(segment_starts) == len(
            segment_ends
        ), "Segment start times and end times do not match"
        assert len(segment_starts) == len(
            subs
        ), "Segment size and subtitle size do not match"

        subs_list = []

        max_workers = math.ceil(float(os.getenv("MAX_WORKERS", mp.cpu_count() / 2)))
        self.__LOGGER.debug("Number of workers: {}".format(max_workers))

        with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
        ) as executor:
            batch_size = max(math.floor(len(segment_starts) / max_workers), 1)
            futures = [
                executor.submit(
                    Predictor._predict_in_multiprocesses,
                    self,
                    batch_idx,
                    segment_starts,
                    segment_ends,
                    weights_file_path,
                    audio_file_path,
                    subs,
                    subs_copy,
                    stretch,
                    stretch_in_lang,
                    exit_segfail
                )
                for batch_idx in Predictor.__minibatch(len(segment_starts), batch_size)
            ]
            for i, future in enumerate(futures):
                try:
                    subs_list.extend(future.result(timeout=Predictor.__SEGMENT_PREDICTION_TIMEOUT * batch_size))
                except concurrent.futures.TimeoutError as e:
                    self.__cancel_futures(futures[i:], Predictor.__SEGMENT_PREDICTION_TIMEOUT * batch_size)
                    message = "Batch alignment timed out after {} seconds".format(Predictor.__SEGMENT_PREDICTION_TIMEOUT)
                    self.__LOGGER.error(message)
                    raise TerminalException(message) from e
                except Exception as e:
                    self.__cancel_futures(futures[i:], Predictor.__SEGMENT_PREDICTION_TIMEOUT * batch_size)
                    message = "Exception on batch alignment: {}\n{}".format(str(e), "".join(traceback.format_stack()))
                    self.__LOGGER.error(e, exc_info=True, stack_info=True)
                    traceback.print_tb(e.__traceback__)
                    if isinstance(e, TerminalException):
                        raise e
                    else:
                        raise TerminalException(message) from e
                except KeyboardInterrupt:
                    self.__cancel_futures(futures[i:], Predictor.__SEGMENT_PREDICTION_TIMEOUT * batch_size)
                    raise TerminalException("Batch alignment interrupted by the user")
                else:
                    self.__LOGGER.debug("Batch aligned")

        subs_list = [sub_item for sub_item in subs_list]
        self.__LOGGER.debug("All segments aligned")
        return subs_list

    def __cancel_futures(self, futures: List[concurrent.futures.Future], timeout: int) -> None:
        for future in futures:
            future.cancel()
        concurrent.futures.wait(futures, timeout=timeout)

    def __get_subtitle_mask(self, subs: List[SubRipItem]) -> np.ndarray:
        pos = self.__feature_embedder.time_to_position(subs[len(subs) - 1].end) - 1
        subtitle_mask = np.zeros(pos if pos > 0 else 0)

        for sub in subs:
            start_pos = self.__feature_embedder.time_to_position(sub.start)
            end_pos = self.__feature_embedder.time_to_position(sub.end)
            for i in np.arange(start_pos, end_pos):
                if i < len(subtitle_mask):
                    subtitle_mask[i] = 1
        return subtitle_mask

    def __on_frame_timecodes(self, subs: List[SubRipItem]) -> None:
        for sub in subs:
            millis_per_frame = self.__feature_embedder.step_sample * 1000
            new_start_millis = round(int(str(sub.start).split(",")[1]) / millis_per_frame + 0.5) * millis_per_frame
            new_start = str(sub.start).split(",")[0] + "," + str(int(new_start_millis)).zfill(3)
            new_end_millis = round(int(str(sub.end).split(",")[1]) / millis_per_frame - 0.5) * millis_per_frame
            new_end = str(sub.end).split(",")[0] + "," + str(int(new_end_millis)).zfill(3)
            sub.start = SubRipTime.coerce(new_start)
            sub.end = SubRipTime.coerce(new_end)

    def __adjust_durations(self, subs: List[SubRipItem], audio_file_path: str, stretch_in_lang: str, lock: threading.RLock) -> List[SubRipItem]:
        from aeneas.executetask import ExecuteTask
        from aeneas.task import Task
        from aeneas.runtimeconfiguration import RuntimeConfiguration
        from aeneas.logger import Logger as AeneasLogger

        # Initialise a DTW alignment task
        task_config_string = (
            "task_language={}|os_task_file_format=srt|is_text_type=subtitles".format(stretch_in_lang)
        )
        runtime_config_string = "dtw_algorithm=stripe"  # stripe or exact
        task = Task(config_string=task_config_string)

        try:
            with lock:
                segment_path, _ = self.__media_helper.extract_audio_from_start_to_end(
                    audio_file_path,
                    str(subs[0].start),
                    str(subs[len(subs) - 1].end),
                )

                # Create a text file for DTW alignments
                root, _ = os.path.splitext(segment_path)
                text_file_path = "{}.txt".format(root)

                with open(text_file_path, "w", encoding="utf8") as text_file:
                    for sub_new in subs:
                        text_file.write(sub_new.text)
                        text_file.write(os.linesep * 2)

                task.audio_file_path_absolute = segment_path
                task.text_file_path_absolute = text_file_path
                task.sync_map_file_path_absolute = "{}.srt".format(root)

                tee = self.__LOGGER.level == getattr(logging, 'DEBUG')

                # Execute the task
                ExecuteTask(
                    task=task,
                    rconf=RuntimeConfiguration(config_string=runtime_config_string),
                    logger=AeneasLogger(tee=tee),
                ).execute()

                # Output new subtitle segment to a file
                task.output_sync_map_file()

            # Load the above subtitle segment
            adjusted_subs = Subtitle.load(
                task.sync_map_file_path_absolute
            ).subs
            for index, sub_new_loaded in enumerate(adjusted_subs):
                sub_new_loaded.index = subs[index].index

            adjusted_subs.shift(
                seconds=self.__media_helper.get_duration_in_seconds(
                    start=None, end=str(subs[0].start)
                )
            )
            return adjusted_subs
        except KeyboardInterrupt:
            raise TerminalException("Subtitle stretch interrupted by the user")
        finally:
            # Housekeep intermediate files
            if task.audio_file_path_absolute is not None and os.path.exists(
                    task.audio_file_path_absolute
            ):
                os.remove(task.audio_file_path_absolute)
            if task.text_file_path_absolute is not None and os.path.exists(
                    task.text_file_path_absolute
            ):
                os.remove(task.text_file_path_absolute)
            if task.sync_map_file_path_absolute is not None and os.path.exists(task.sync_map_file_path_absolute):
                os.remove(task.sync_map_file_path_absolute)

    def __predict(
            self,
            video_file_path: Optional[str],
            subtitle_file_path: Optional[str],
            weights_file_path: str,
            audio_file_path: Optional[str] = None,
            subtitles: Optional[SubRipFile] = None,
            max_shift_secs: Optional[float] = None,
            previous_gap: Optional[float] = None,
            lock: Optional[threading.RLock] = None,
            network: Optional[Network] = None
    ) -> Tuple[List[SubRipItem], str, np.ndarray]:
        """Shift out-of-sync subtitle cues by sending the audio track of an video to the trained network.

        Arguments:
            video_file_path {string} -- The file path of the original video.
            subtitle_file_path {string} -- The file path of the out-of-sync subtitles.
            weights_file_path {string} -- The file path of the weights file.

        Keyword Arguments:
            audio_file_path {string} -- The file path of the original audio (default: {None}).
            subtitles {list} -- The list of SubRip files (default: {None}).
            max_shift_secs {float} -- The maximum seconds by which subtitle cues can be shifted (default: {None}).
            previous_gap {float} -- The duration between the start time of the audio segment and the start time of the subtitle segment (default: {None}).

        Returns:
            tuple -- The shifted subtitles, the audio file path and the voice probabilities of the original audio.
        """
        if network is None:
            network = self.__initialise_network(os.path.dirname(weights_file_path), self.__LOGGER)
        result: Dict[str, Any] = {}
        pred_start = datetime.datetime.now()
        if audio_file_path is not None:
            result["audio_file_path"] = audio_file_path
        elif video_file_path is not None:
            t = datetime.datetime.now()
            audio_file_path = self.__media_helper.extract_audio(
                video_file_path, True, 16000
            )
            self.__LOGGER.debug(
                "[{}] Audio extracted after {}".format(
                    os.getpid(), str(datetime.datetime.now() - t)
                )
            )
            result["video_file_path"] = video_file_path
        else:
            raise TerminalException("Neither audio nor video is passed in")

        if subtitle_file_path is not None:
            subs = Subtitle.load(subtitle_file_path).subs
            result["subtitle_file_path"] = subtitle_file_path
        elif subtitles is not None:
            subs = subtitles
        else:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            raise TerminalException("ERROR: No subtitles passed in")
        if lock is not None:
            with lock:
                try:
                    train_data, labels = self.__feature_embedder.extract_data_and_label_from_audio(
                        audio_file_path, None, subtitles=subs
                    )
                except TerminalException:
                    if os.path.exists(audio_file_path):
                        os.remove(audio_file_path)
                    raise
        else:
            try:
                train_data, labels = self.__feature_embedder.extract_data_and_label_from_audio(
                    audio_file_path, None, subtitles=subs
                )
            except TerminalException:
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
                raise

        train_data = np.array([np.rot90(val) for val in train_data])
        train_data = train_data - np.mean(train_data, axis=0)
        result["time_load_dataset"] = str(datetime.datetime.now() - pred_start)
        result["X_shape"] = train_data.shape

        # Load neural network
        input_shape = (train_data.shape[1], train_data.shape[2])
        self.__LOGGER.debug("[{}] input shape: {}".format(os.getpid(), input_shape))

        # Network class is not thread safe so a new graph is created for each thread
        pred_start = datetime.datetime.now()
        if lock is not None:
            with lock:
                try:
                    self.__LOGGER.debug("[{}] Start predicting...".format(os.getpid()))
                    voice_probabilities = network.get_predictions(train_data, weights_file_path)
                except Exception as e:
                    self.__LOGGER.error("[{}] Prediction failed: {}\n{}".format(os.getpid(), str(e), "".join(traceback.format_stack())))
                    traceback.print_tb(e.__traceback__)
                    raise TerminalException("Prediction failed") from e
                finally:
                    del train_data
                    del labels
                    gc.collect()
        else:
            try:
                self.__LOGGER.debug("[{}] Start predicting...".format(os.getpid()))
                voice_probabilities = network.get_predictions(train_data, weights_file_path)
            except Exception as e:
                self.__LOGGER.error(
                    "[{}] Prediction failed: {}\n{}".format(os.getpid(), str(e), "".join(traceback.format_stack())))
                traceback.print_tb(e.__traceback__)
                raise TerminalException("Prediction failed") from e
            finally:
                del train_data
                del labels
                gc.collect()

        if len(voice_probabilities) <= 0:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            raise TerminalException(
                "ERROR: Audio is too short and no voice was detected"
            )

        result["time_predictions"] = str(datetime.datetime.now() - pred_start)

        original_start = FeatureEmbedder.time_to_sec(subs[0].start)
        shifted_subs = deepcopy(subs)
        subs.shift(seconds=-original_start)

        self.__LOGGER.info("[{}] Aligning subtitle with video...".format(os.getpid()))

        if lock is not None:
            with lock:
                min_log_loss, min_log_loss_pos = self.get_min_log_loss_and_index(
                    voice_probabilities, subs
                )
        else:
            min_log_loss, min_log_loss_pos = self.get_min_log_loss_and_index(
                voice_probabilities, subs
            )

        pos_to_delay = min_log_loss_pos
        result["loss"] = min_log_loss

        self.__LOGGER.info("[{}] Subtitle aligned".format(os.getpid()))

        if subtitle_file_path is not None:  # for the first pass
            seconds_to_shift = (
                self.__feature_embedder.position_to_duration(pos_to_delay) - original_start
            )
        elif subtitles is not None:  # for each in second pass
            seconds_to_shift = self.__feature_embedder.position_to_duration(pos_to_delay) - previous_gap if previous_gap is not None else 0.0
        else:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            raise ValueError("ERROR: No subtitles passed in")

        if abs(seconds_to_shift) > Predictor.__MAX_SHIFT_IN_SECS:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            raise TerminalException(
                "Average shift duration ({} secs) have been reached".format(
                    Predictor.__MAX_SHIFT_IN_SECS
                )
            )

        result["seconds_to_shift"] = seconds_to_shift
        result["original_start"] = original_start
        total_elapsed_time = str(datetime.datetime.now() - pred_start)
        result["time_sync"] = total_elapsed_time
        self.__LOGGER.debug("[{}] Statistics: {}".format(os.getpid(), result))

        self.__LOGGER.debug("[{}] Total Time: {}".format(os.getpid(), total_elapsed_time))
        self.__LOGGER.debug(
            "[{}] Seconds to shift: {}".format(os.getpid(), seconds_to_shift)
        )

        # For each subtitle chunk, its end time should not be later than the end time of the audio segment
        if max_shift_secs is not None and seconds_to_shift <= max_shift_secs:
            shifted_subs.shift(seconds=seconds_to_shift)
        elif max_shift_secs is not None and seconds_to_shift > max_shift_secs:
            self.__LOGGER.warning(
                "[{}] Maximum {} seconds shift has reached".format(os.getpid(), max_shift_secs)
            )
            shifted_subs.shift(seconds=max_shift_secs)
        else:
            shifted_subs.shift(seconds=seconds_to_shift)
        self.__LOGGER.debug("[{}] Subtitle shifted".format(os.getpid()))
        return shifted_subs, audio_file_path, voice_probabilities


class _ThreadPoolExecutorLocal:

    def __init__(self, queue_size: int, max_workers: int):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = threading.BoundedSemaphore(queue_size + max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown(True)

    def submit(self, fn, *args, **kwargs):
        self.semaphore.acquire()
        try:
            future = self.executor.submit(fn, *args, **kwargs)
        except Exception:
            self.semaphore.release()
            raise
        else:
            future.add_done_callback(lambda x: self.semaphore.release())
            return future
