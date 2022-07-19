import datetime
import os
import h5py
import threading
import traceback
import concurrent.futures
import math
import numpy as np
import multiprocessing as mp

from typing import List, Tuple, Optional
from .network import Network
from .media_helper import MediaHelper
from .hyperparameters import Hyperparameters
from .embedder import FeatureEmbedder
from .exception import TerminalException
from .logger import Logger


class Trainer(object):
    """Network trainer.
    """

    EMBEDDING_TIMEOUT = 300  # time out for feature embedding of media files
    __MAX_BYTES = 2 ** 31 - 1

    def __init__(self, feature_embedder: FeatureEmbedder) -> None:
        """Initialiser for the training process.

        Arguments:
            feature_embedder {Embedder.FeatureEmbedder} -- the feature embedder object.

        Raises:
            NotImplementedError -- Thrown when any Trainer attributes are modified.
        """

        self.__feature_embedder = feature_embedder
        self.__lock = threading.RLock()
        self.__media_helper = MediaHelper()
        self.__LOGGER = Logger().get_logger(__name__)

    def train(
        self,
        av_file_paths: List[str],
        subtitle_file_paths: List[str],
        model_dir: str,
        weights_dir: str,
        config_dir: str,
        logs_dir: str,
        training_dump_dir: str,
        hyperparameters: Hyperparameters,
        training_log: str = "training.log",
        resume: bool = False,
        sound_effect_start_marker: Optional[str] = None,
        sound_effect_end_marker: Optional[str] = None
    ) -> None:
        """Trigger the training process.

        Arguments:
            av_file_paths {list} -- A list of paths to the input audio/video files.
            subtitle_file_paths {list} -- A list of paths to the subtitle files.
            model_dir {string} -- The directory of the model file.
            weights_dir {string} -- The directory of the weights file.
            config_dir {string} -- The directory of the hyperparameter file where hyperparameters will be saved.
            logs_dir {string} -- The directory of the log file.
            training_dump_dir {string} --  The directory of the training data dump file.
            hyperparameters {Hyperparameters} -- A configuration for hyperparameters used for training.
            training_log {string} -- The path to the log file of epoch results (default: {"training.log"}).
            resume {bool} -- True to continue with previous training result or False to start a new one (default: {False}).
            sound_effect_start_marker: {string} -- A string indicating the start of the ignored sound effect (default: {"("}).
            sound_effect_end_marker: {string} -- A string indicating the end of the ignored sound effect (default: {")"}).
        """

        training_start = datetime.datetime.now()
        model_filepath = os.path.join(os.path.abspath(model_dir), "model.hdf5")
        weights_filepath = os.path.join(os.path.abspath(weights_dir), "weights.hdf5")
        hyperparams_filepath = os.path.join(os.path.abspath(config_dir), "hyperparameters.json")

        if av_file_paths is None or subtitle_file_paths is None:
            # Load the data and labels dump from the disk
            training_dump = os.path.join(os.path.abspath(training_dump_dir), "training_dump.hdf5")
            self.__LOGGER.debug(
                "Resume training on data dump: ".format(
                    training_dump
                )
            )
            with h5py.File(training_dump, "r") as hf:
                train_data_raw = hf["train_data"]
                labels_raw = hf["labels"]

                if resume:
                    # Load hyperparameters from previous training
                    loaded_hyperparameters = Hyperparameters.from_file(hyperparams_filepath)

                    # Update the total epochs and save hyperparameters
                    loaded_hyperparameters.epochs = hyperparameters.epochs
                    loaded_hyperparameters.to_file(hyperparams_filepath)

                    network = Network.load_model_and_weights(model_filepath, weights_filepath, loaded_hyperparameters)
                else:
                    # Save hyperparameters before each new training
                    hyperparameters.to_file(hyperparams_filepath)

                    input_shape = (train_data_raw.shape[2], train_data_raw.shape[1])
                    self.__LOGGER.debug("input_shape: {}".format(input_shape))
                    network = Network.get_network(input_shape, hyperparameters)

                val_loss, val_acc = network.fit_with_generator(
                    train_data_raw,
                    labels_raw,
                    model_filepath,
                    weights_filepath,
                    logs_dir,
                    training_log,
                    resume,
                )
        else:
            train_data, labels = Trainer.__extract_data_and_label_from_avs(
                self, av_file_paths, subtitle_file_paths, sound_effect_start_marker, sound_effect_end_marker
            )

            # Dump extracted data and labels to files for re-training
            training_dump = os.path.join(os.path.abspath(training_dump_dir), "training_dump.hdf5")
            with h5py.File(training_dump, "w") as hf:
                hf.create_dataset("train_data", data=train_data)
                hf.create_dataset("labels", data=labels)

            rand = np.random.permutation(np.arange(len(labels)))
            train_data = train_data[rand]
            labels = labels[rand]

            train_data = np.array(
                [np.rot90(m=val, k=1, axes=(0, 1)) for val in train_data]
            )
            train_data = train_data - np.mean(train_data, axis=0)

            input_shape = (train_data.shape[1], train_data.shape[2])
            self.__LOGGER.debug("input_shape: {}".format(input_shape))

            # Save hyperparameters before each new training
            hyperparameters.to_file(hyperparams_filepath)

            network = Network.get_network(input_shape, hyperparameters)
            val_loss, val_acc = network.fit_and_get_history(
                train_data,
                labels,
                model_filepath,
                weights_filepath,
                logs_dir,
                training_log,
                False,
            )

        self.__LOGGER.debug("val_loss: {}".format(min(val_loss)))
        self.__LOGGER.debug("val_acc: {}".format(max(val_acc)))
        self.__LOGGER.info(
            "Total training time: {}".format(
                str(datetime.datetime.now() - training_start)
            )
        )

        # Save the model together with the weights after training
        combined_filepath = os.path.join(os.path.abspath(model_dir), "combined.hdf5")
        network.save_model_and_weights(
            model_filepath, weights_filepath, combined_filepath
        )

    def pre_train(
        self,
        av_file_paths: List[str],
        subtitle_file_paths: List[str],
        training_dump_dir: str,
        hyperparameters: Hyperparameters,
        sound_effect_start_marker: Optional[str] = None,
        sound_effect_end_marker: Optional[str] = None
    ) -> Tuple[List[float], List[float]]:
        """Trigger the training process.

        Arguments:
            av_file_paths {list} -- A list of paths to the input audio/video files.
            subtitle_file_paths {list} -- A list of paths to the subtitle files.
            training_dump_dir {string} --  The directory of the training data dump file.
            hyperparameters {Hyperparameters} -- A configuration for hyperparameters used for training.
            sound_effect_start_marker: {string} -- A string indicating the start of the ignored sound effect (default: {"("}).
            sound_effect_end_marker: {string} -- A string indicating the end of the ignored sound effect (default: {")"}).
        """

        training_dump = os.path.join(os.path.abspath(training_dump_dir), "training_dump.hdf5")
        if os.path.exists(training_dump):
            with h5py.File(training_dump, "r") as hf:
                train_data_raw = hf["train_data"]
                labels_raw = hf["labels"]

                input_shape = (train_data_raw.shape[2], train_data_raw.shape[1])
                self.__LOGGER.debug("input_shape: {}".format(input_shape))

                val_loss, val_acc = Network.simple_fit_with_generator(
                    input_shape,
                    train_data_raw,
                    labels_raw,
                    hyperparameters
                )
        else:
            train_data, labels = Trainer.__extract_data_and_label_from_avs(
                self, av_file_paths, subtitle_file_paths, sound_effect_start_marker, sound_effect_end_marker
            )
            with h5py.File(training_dump, "w") as hf:
                hf.create_dataset("train_data", data=train_data)
                hf.create_dataset("labels", data=labels)

            rand = np.random.permutation(np.arange(len(labels)))
            train_data = train_data[rand]
            labels = labels[rand]

            train_data = np.array(
                [np.rot90(m=val, k=1, axes=(0, 1)) for val in train_data]
            )
            train_data = train_data - np.mean(train_data, axis=0)

            input_shape = (train_data.shape[1], train_data.shape[2])
            self.__LOGGER.debug("input_shape: {}".format(input_shape))

            val_loss, val_acc = Network.simple_fit(
                input_shape,
                train_data,
                labels,
                hyperparameters
            )
        return val_loss, val_acc

    @staticmethod
    def get_done_epochs(training_log: str) -> int:
        """Get the number of finished epochs.

        Arguments:
            training_log {string} -- The path to the training log file.
        """
        if not os.path.isfile(training_log):
            return 0
        epochs_done = 0
        training_log_file = open(training_log)
        epochs_done += sum(1 for _ in training_log_file) - 1
        training_log_file.close()
        return epochs_done if epochs_done >= 0 else 0

    def __extract_data_and_label_from_avs(
        self,
        av_file_paths: List[str],
        subtitle_file_paths: List[str],
        sound_effect_start_marker: Optional[str],
        sound_effect_end_marker: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a training dataset and labels from audio/video files.

        Arguments:
            av_file_paths {list} -- A list of paths to the input audio/video files.
            subtitle_file_paths {list} -- A list of paths to the subtitle files.
            sound_effect_start_marker: {string} -- A string indicating the start of the ignored sound effect.
            sound_effect_end_marker: {string} -- A string indicating the end of the ignored sound effect.

        Returns:
            tuple -- The training data and labels.
        """

        train_data, labels = (
            [None] * len(av_file_paths),
            [None] * len(subtitle_file_paths),
        )

        extraction_start = datetime.datetime.now()
        max_workers = math.ceil(float(os.getenv("MAX_WORKERS", mp.cpu_count() / 2)))
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(
                    Trainer.__extract_in_multithreads,
                    self,
                    index,
                    av_file_paths[index],
                    subtitle_file_paths[index],
                    train_data,
                    labels,
                    sound_effect_start_marker,
                    sound_effect_end_marker
                )
                for index in range(len(av_file_paths))
            ]
            try:
                done, not_done = concurrent.futures.wait(futures, timeout=Trainer.EMBEDDING_TIMEOUT)
            except KeyboardInterrupt:
                for future in futures:
                    if not future.cancel():
                        self.__LOGGER.warning("Data and label extraction job cannot be cancelled")
                raise TerminalException("Data and label extraction interrupted by the user")
            for future in not_done:
                # Log undone audio files and continue
                try:
                    audio_file_path, subtitle_file_path = future.result()
                    self.__LOGGER.warning(
                        "Data and label extraction failed for: [Audio: {}, Subtitle: {}]".format(
                            audio_file_path, subtitle_file_path
                        )
                    )
                    if not future.cancel():
                        self.__LOGGER.warning("Data and label extraction job cannot be cancelled")
                except Exception as e:
                    self.__LOGGER.error(
                        "Unexpected exception during data and label extraction: {} stacktrace: {}".format(
                            str(e), "".join(traceback.format_stack())
                        )
                    )
                    traceback.print_tb(e.__traceback__)

        train_data = [x for x in train_data if x is not None]
        labels = [x for x in labels if x is not None]

        train_data: np.ndarray = np.concatenate(train_data)  # type: ignore
        labels: np.ndarray = np.concatenate(labels)  # type: ignore
        self.__LOGGER.debug(
            "Data and labels extracted after {} seconds".format(
                str(datetime.datetime.now() - extraction_start)
            )
        )

        return train_data, labels

    def __extract_in_multithreads(
        self,
        index: int,
        av_file_path: str,
        subtitle_file_path: str,
        train_data: np.ndarray,
        labels: np.ndarray,
        sound_effect_start_marker: Optional[str],
        sound_effect_end_marker: Optional[str]
    ) -> Tuple[str, str]:
        _, file_ext = os.path.splitext(av_file_path)

        try:
            if file_ext not in self.__media_helper.AUDIO_FILE_EXTENSION:
                t = datetime.datetime.now()
                audio_file_path = self.__media_helper.extract_audio(
                    av_file_path, True, 16000
                )
                self.__LOGGER.debug(
                    "- Audio extracted after {}".format(
                        str(datetime.datetime.now() - t)
                    )
                )
            else:
                audio_file_path = av_file_path
            with self.__lock:
                x, y = self.__feature_embedder.extract_data_and_label_from_audio(
                    audio_file_path,
                    subtitle_file_path,
                    subtitles=None,
                    sound_effect_start_marker=sound_effect_start_marker,
                    sound_effect_end_marker=sound_effect_end_marker
                )

        # Some media files are malformed and on occurring they will be logged
        # However, the training shall continue after expensive embedding on healthy media files.
        except Exception as e:
            # Log failed audio and subtitle files and continue
            self.__LOGGER.warning(
                "Exception: {}; stacktrace: {}".format(
                    str(e), "".join(traceback.format_stack())
                )
            )
            self.__LOGGER.warning(
                "[Audio: {}, Subtitle: {}]".format(
                    audio_file_path, subtitle_file_path
                )
            )
            traceback.print_tb(e.__traceback__)
        else:
            train_data[index] = x
            labels[index] = y
        return audio_file_path, subtitle_file_path
