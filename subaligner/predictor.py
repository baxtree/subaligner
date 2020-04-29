import os
import datetime
import traceback
import threading
import concurrent.futures
import numpy as np
import multiprocessing as mp

from pysrt import SubRipTime
from sklearn.metrics import log_loss
from copy import deepcopy
from aeneas.executetask import ExecuteTask
from aeneas.task import Task
from aeneas.runtimeconfiguration import RuntimeConfiguration
from aeneas.logger import Logger as AeneasLogger
from .network import Network
from .embedder import FeatureEmbedder
from .media_helper import MediaHelper
from .singleton import Singleton
from .subtitle import Subtitle
from .hyperparameters import Hyperparameters
from .exception import TerminalException
from .logger import Logger


class Predictor(Singleton):
    """ Predictor for working out the time to shift subtitles
    """

    __LOGGER = Logger().get_logger(__name__)
    __MAX_SHIFT_IN_SECS = (
        100
    )
    __MAX_CHARS_PER_SEC = (
        50
    )  # Average 0.3 word per sec multiplies average 6 characters per word
    __MAX_HEAD_ROOM = 20000  # Maximum duration without subtitle (10 minutes)

    def __init__(self, **kwargs):
        """Feature predictor initialiser.

            Keyword Arguments:
                n_mfcc {int} -- The number of MFCC components (default: {13}).
                frequency {float} -- The sample rate  (default: {16000}).
                hop_len {int} -- The number of samples per frame (default: {512}).
                step_sample {float} -- The space (in seconds) between the begining of each sample (default: 1s / 25 FPS = 0.04s).
                len_sample {float} -- The length in seconds for the input samples (default: {0.075}).
        """

        self.__feature_embedder = FeatureEmbedder(**kwargs)
        self.__lock = threading.Lock()

    def predict_single_pass(
            self,
            video_file_path,
            subtitle_file_path,
            weights_dir="models/training/weights",
    ):
        """Predict time to shift with single pass

            Arguments:
                video_file_path {string} -- The input video file path.
                subtitle_file_path {string} -- The path to the subtitle file.
                weights_dir {string} -- The the model weights directory.

            Returns:
                tuple -- The shifted subtitles, the audio file path and the voice probabilities of the original audio.
        """

        weights_dir = os.path.join(os.path.dirname(__file__), weights_dir)
        audio_file_path = ""
        try:
            subs, audio_file_path, voice_probabilities = self.__predict(
                video_file_path, subtitle_file_path, weights_dir
            )
        except Exception:
            raise
        else:
            frame_rate = MediaHelper.get_frame_rate(video_file_path)
            self.__feature_embedder.step_sample = 1 / frame_rate
            self.__on_frame_timecodes(subs)
            return subs, audio_file_path, voice_probabilities
        finally:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)

    def predict_dual_pass(
            self,
            video_file_path,
            subtitle_file_path,
            weights_dir="models/training/weights",
            stretch=False,
    ):
        """Predict time to shift with single pass

            Arguments:
            video_file_path {string} -- The input video file path.
            subtitle_file_path {string} -- The path to the subtitle file.
            weights_dir {string} -- The the model weights directory.
            stretch {bool} -- True to stretch the subtitle segments (default: {False})

            Returns:
            tuple -- The shifted subtitles, the audio file path and the voice probabilities of the original audio.
        """

        weights_dir = os.path.join(os.path.dirname(__file__), weights_dir)
        audio_file_path = ""
        try:
            subs, audio_file_path, voice_probabilities = self.__predict(
                video_file_path, subtitle_file_path, weights_dir
            )
            new_subs = self.__predict_2nd_pass(
                audio_file_path, subs, weights_dir=weights_dir, stretch=stretch
            )
        except Exception:
            raise
        else:
            frame_rate = MediaHelper.get_frame_rate(video_file_path)
            self.__feature_embedder.step_sample = 1 / frame_rate
            self.__on_frame_timecodes(new_subs)
            return new_subs, subs, voice_probabilities
        finally:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)

    def get_log_loss(self, voice_probabilities, subs):
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
            local_vp = np.vstack(
                [
                    voice_probabilities,
                    [np.zeros(voice_probabilities.shape[1])] * (-head_room * 5),
                ]
            )
        else:
            local_vp = voice_probabilities
        result = log_loss(
            subtitle_mask, local_vp[: len(subtitle_mask)], labels=[0, 1]
        )
        Predictor.__LOGGER.debug("Log loss: {}".format(result))
        return result

    def get_min_log_loss_and_index(self, voice_probabilities, subs):
        """Returns the minimum loss value and its shift position
        after going through all possible shifts.

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
        Predictor.__LOGGER.debug("head room: {}".format(head_room))
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
            Predictor.__LOGGER.error("head room: {}".format(head_room))
            raise TerminalException(
                "Maximum head room reached due to the suspicious audio or subtitle duration"
            )

        log_losses = []
        Predictor.__LOGGER.debug(
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
            return min_log_loss, log_losses.index(min_log_loss)
        else:
            return None, 0

    def __predict(
        self,
        video_file_path,
        subtitle_file_path,
        model_weights_dir,
        audio_file_path=None,
        subtitles=None,
        max_shift_secs=None,
        previous_gap=None,
    ):
        """Shift out-of-sync subtitle cues by sending the audio track of an video to the trained network.

        Arguments:
            video_file_path {string} -- The file path of the original video.
            subtitle_file_path {string} -- The file path of the out-of-sync subtitles.
            model_weights_dir {string} -- The directory of the model and weights.

        Keyword Arguments:
            audio_file_path {string} -- The file path of the original audio (default: {None}).
            subtitles {list} -- The list of SubRip files (default: {None}).
            max_shift_secs {float} -- The maximum seconds by which subtitle cues can be shifted (default: {None}).
            previous_gap {float} -- The duration betwee the start time of the audio segment and the start time of the subtitle segment.

        Returns:
            tuple -- The shifted subtitles, the audio file path and the voice probabilities of the original audio.
        """

        model_dir = model_weights_dir.replace("/weights", "/model")
        config_dir = model_weights_dir.replace("/weights", "/config")
        files = os.listdir(model_dir)
        model_files = [
            file
            for file in files
            if file.startswith("model")
        ]
        files = os.listdir(model_weights_dir)
        weights_files = [
            file
            for file in files
            if file.startswith("weights")
        ]

        files = os.listdir(config_dir)
        hyperparams_files = [
            file
            for file in files
            if file.startswith("hyperparameters")
        ]

        if not model_files:
            raise TerminalException(
                "Cannot find model files at {}".format(model_weights_dir)
            )
        if not weights_files:
            raise TerminalException(
                "Cannot find weights files at {}".format(model_weights_dir)
            )

        Predictor.__LOGGER.debug("model files: {}".format(model_files))
        Predictor.__LOGGER.debug("weigths files: {}".format(weights_files))
        Predictor.__LOGGER.debug("config files: {}".format(hyperparams_files))

        # Get the first file from the file lists
        model_path = "{}/{}".format(model_dir, model_files[0])
        weights_path = "{}/{}".format(model_weights_dir, weights_files[0])
        hyperparams_path = "{}/{}".format(config_dir, hyperparams_files[0])

        result = {}
        pred_start = datetime.datetime.now()
        if audio_file_path is not None:
            result["audio_file_path"] = audio_file_path
        elif video_file_path is not None:
            t = datetime.datetime.now()
            audio_file_path = MediaHelper.extract_audio(
                video_file_path, True, 16000
            )
            Predictor.__LOGGER.debug(
                "Audio extracted after {}".format(
                    str(datetime.datetime.now() - t)
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
            raise TerminalException("Error: No subtitles passed in")

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
        Predictor.__LOGGER.debug("input shape: {}".format(input_shape))

        # Network class is not thread safe so a new graph is created for each thread
        self.__lock.acquire()
        try:
            hyperparams = Hyperparameters.from_file(hyperparams_path)
            network = Network.get_from_model(model_path, hyperparams)
            Predictor.__LOGGER.debug("Start predicting...")
            pred_start = datetime.datetime.now()
            voice_probabilities = network.get_predictions(train_data, weights_path)
        finally:
            self.__lock.release()

        if len(voice_probabilities) <= 0:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            raise TerminalException(
                "Error: Audio is too short and no voice was detected"
            )

        result["time_predictions"] = str(datetime.datetime.now() - pred_start)

        # for p in voice_probabilities: Predictor.__LOGGER.debug("{}, ".format(p))
        # Predictor.__LOGGER.debug("predictions: {}".format(voice_probabilities))

        original_start = FeatureEmbedder.time_to_sec(subs[0].start)
        Predictor.__LOGGER.debug("Start {}".format(original_start))
        shifted_subs = deepcopy(subs)
        subs.shift(seconds=-original_start)

        Predictor.__LOGGER.info("Aligning subtitle with video...")

        min_log_loss, min_log_loss_pos = self.get_min_log_loss_and_index(
            voice_probabilities, subs
        )

        pos_to_delay = min_log_loss_pos
        result["loss"] = min_log_loss

        Predictor.__LOGGER.info("Subtitle aligned")

        if subtitle_file_path is not None:  # for the first pass
            seconds_to_shift = (
                self.__feature_embedder.pos_to_sec(pos_to_delay) - original_start
            )
        elif subtitles is not None:  # for each in second pass
            seconds_to_shift = self.__feature_embedder.pos_to_sec(pos_to_delay) - previous_gap
        else:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            raise ValueError("Error: No subtitles passed in")

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
        Predictor.__LOGGER.debug("Statistics: {}".format(result))

        Predictor.__LOGGER.debug("Total Time: {}".format(total_elapsed_time))
        Predictor.__LOGGER.debug(
            "Seconds to shift: {}".format(seconds_to_shift)
        )

        # For each subtitle chunk, its end time should not be later than the end time of the audio segment
        if max_shift_secs is not None and seconds_to_shift <= max_shift_secs:
            shifted_subs.shift(seconds=seconds_to_shift)
        elif max_shift_secs is not None and seconds_to_shift > max_shift_secs:
            Predictor.__LOGGER.warning(
                "Maximum {} seconds shift has reached".format(max_shift_secs)
            )
            shifted_subs.shift(seconds=max_shift_secs)
        else:
            shifted_subs.shift(seconds=seconds_to_shift)
        return shifted_subs, audio_file_path, voice_probabilities

    def __predict_2nd_pass(self, audio_file_path, subs, weights_dir, stretch):
        """This function uses divide and conquer to align partial subtitle with partial video.

        Arguments:
            audio_file_path {string} -- The file path of the original audio.
            subs {list} -- A list of SubRip files.
            weights_dir {string} --  The directory of the model weights.
            stretch {bool} -- True to stretch the subtitle segments (default: {False})
        """

        segment_starts, segment_ends, subs = MediaHelper.get_audio_segment_starts_and_ends(
            subs
        )
        subs_copy = deepcopy(subs)

        for index, sub in enumerate(subs):
            Predictor.__LOGGER.debug(
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

        max_workers = int(os.getenv("MAX_WORKERS", mp.cpu_count() / 2))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(
                    Predictor.__predict_in_multithreads,
                    self,
                    i,
                    segment_starts,
                    segment_ends,
                    weights_dir,
                    audio_file_path,
                    subs,
                    subs_copy,
                    stretch
                )
                for i in range(len(segment_starts))
            ]
            concurrent.futures.wait(futures)
            for future in futures:
                try:
                    new_subs = future.result()
                except Exception as e:
                    Predictor.__LOGGER.error(
                        "Exception: {}\n{}".format(
                            str(e), traceback.format_stack()
                        )
                    )
                    raise
                else:
                    subs_list.append(new_subs)

        subs_list = [
            sub_item for subs in subs_list for sub_item in subs
        ]  # flatten the subs_list
        return subs_list

    def __normalise_seconds_to_shift(seconds_to_shift, step_sample):
        # Make sure each cue starts right on the beginning of a frame
        return round(seconds_to_shift / step_sample) * step_sample

    def __predict_in_multithreads(
        self,
        segment_index,
        segment_starts,
        segment_ends,
        weights_dir,
        audio_file_path,
        subs,
        subs_copy,
        stretch,
    ):
        segment_path = ""
        try:
            if segment_index == (len(segment_starts) - 1):
                segment_path, segment_duration = MediaHelper.extract_audio_from_start_to_end(
                    audio_file_path, segment_starts[segment_index], None
                )
            else:
                segment_path, segment_duration = MediaHelper.extract_audio_from_start_to_end(
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
                previous_gap = FeatureEmbedder.time_to_sec(
                    subs[segment_index][0].start
                ) - FeatureEmbedder.time_to_sec(
                    subs[segment_index - 1][
                        len(subs[segment_index - 1]) - 1
                    ].end
                )

            subs_new, _, _ = self.__predict(
                video_file_path=None,
                subtitle_file_path=None,
                model_weights_dir=weights_dir,
                audio_file_path=segment_path,
                subtitles=subs_copy[segment_index],
                max_shift_secs=max_shift_secs,
                previous_gap=previous_gap,
            )

            if stretch:
                subs_new = self.__adjust_durations(subs_new, audio_file_path)
            return subs_new
        finally:
            # Housekeep intermediate files
            if os.path.exists(segment_path):
                os.remove(segment_path)

    def __get_subtitle_mask(self, subs):
        pos = self.__feature_embedder.time_to_pos(subs[len(subs) - 1].end) - 1
        subtitle_mask = np.zeros(pos if pos > 0 else 0)

        for sub in subs:
            start_pos = self.__feature_embedder.time_to_pos(sub.start)
            end_pos = self.__feature_embedder.time_to_pos(sub.end)
            for i in np.arange(start_pos, end_pos):
                if i < len(subtitle_mask):
                    subtitle_mask[i] = 1
        return subtitle_mask

    def __adjust_durations(self, subs, audio_file_path):

        # Initialise a DTW alignment task
        task_config_string = (
            "task_language=eng|os_task_file_format=srt|is_text_type=subtitles"
        )
        runtime_config_string = "dtw_algorithm=stripe"  # stripe or exact
        task = Task(config_string=task_config_string)

        try:
            segment_path, _ = MediaHelper.extract_audio_from_start_to_end(
                audio_file_path,
                str(subs[0].start),
                str(subs[len(subs) - 1].end),
            )

            # Create a text file for DTW alignments
            root, _ = os.path.splitext(segment_path)
            text_file_path = "{}.txt".format(root)

            with open(text_file_path, "w") as text_file:
                for sub_new in subs:
                    text_file.write(sub_new.text)
                    text_file.write(os.linesep * 2)

            task.audio_file_path_absolute = segment_path
            task.text_file_path_absolute = text_file_path
            task.sync_map_file_path_absolute = "{}.srt".format(root)

            # Execute the task
            ExecuteTask(
                task=task,
                rconf=RuntimeConfiguration(config_string=runtime_config_string),
                logger=AeneasLogger(tee=Logger.VERBOSE),
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
                seconds=MediaHelper.get_duration_in_seconds(
                    start=None, end=str(subs[0].start)
                )
            )
            return adjusted_subs
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

    def __on_frame_timecodes(self, subs):
        for sub in subs:
            millis_per_frame = self.__feature_embedder.step_sample * 1000
            new_start_millis = round(int(str(sub.start).split(",")[1]) / millis_per_frame + 0.5) * millis_per_frame
            new_start = str(sub.start).split(",")[0] + "," + str(int(new_start_millis)).zfill(3)
            new_end_millis = round(int(str(sub.end).split(",")[1]) / millis_per_frame - 0.5) * millis_per_frame
            new_end = str(sub.end).split(",")[0] + "," + str(int(new_end_millis)).zfill(3)
            sub.start = SubRipTime.coerce(new_start)
            sub.end = SubRipTime.coerce(new_end)
