import os
import whisper
import torch
import concurrent.futures
import math
import multiprocessing as mp
import torchaudio
import numpy as np
from functools import partial
from threading import Lock
from typing import Tuple, Optional, Dict, List
from pysrt import SubRipTime
from whisper import Whisper
from whisper.tokenizer import LANGUAGES
from tqdm import tqdm
from .subtitle import Subtitle
from .media_helper import MediaHelper
from .llm import TranscriptionRecipe, WhisperFlavour
from .logger import Logger
from .utils import Utils
from .exception import NoFrameRateException, TranscriptionException


class Transcriber(object):
    """Transcribe audiovisual content for subtitle generation.

    Arguments:
        recipe {string} -- the LLM recipe used for transcribing video files (default: "whisper").
        flavour {string} -- the flavour variation for a specific LLM recipe (default: "small").

    Raises:
        NotImplementedError: Thrown when the LLM recipe is unknown.
    """

    def __init__(self, recipe: str = TranscriptionRecipe.WHISPER.value, flavour: str = WhisperFlavour.SMALL.value) -> None:
        if recipe not in [r.value for r in TranscriptionRecipe]:
            raise NotImplementedError(f"Unknown recipe: {recipe}")
        if recipe == TranscriptionRecipe.WHISPER.value:
            if flavour not in [f.value for f in WhisperFlavour]:
                raise NotImplementedError(f"Unknown {recipe} flavour: {flavour}")
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
            self.__model = whisper.load_model(flavour, device=device)
        self.__recipe = recipe
        self.__flavour = flavour
        self.__media_helper = MediaHelper()
        self.__LOGGER = Logger().get_logger(__name__)
        self.__lock = Lock()

    def transcribe(self,
                   video_file_path: str,
                   language_code: str,
                   initial_prompt: Optional[str] = None,
                   max_char_length: Optional[int] = None,
                   with_word_time_codes: bool = False) -> Tuple[Subtitle, Optional[float]]:
        """Transcribe an audiovisual file and generate subtitles.

        Arguments:
            video_file_path {string} -- The input video file path.
            language_code {string} -- An alpha 3 language code derived from ISO 639-3.
            initial_prompt {string} -- Optional Text to provide the transcribing context or specific phrases.
            max_char_length {int} -- Optional Maximum number of characters for each generated subtitle segment.
            with_word_time_codes {bool} -- True to output time codes for each word (default: {False})

        Returns:
            tuple: Generated subtitle after transcription and the detected frame rate

        Raises:
            TranscriptionException: Thrown when transcription is failed.
            NotImplementedError: Thrown when the LLM recipe is not supported.
        """
        if self.__recipe == "whisper":
            lang = Utils.get_iso_639_alpha_2(language_code)
            if lang not in LANGUAGES:
                raise TranscriptionException(f'"{language_code}" is not supported by {self.__recipe} ({self.__flavour})')
            audio_file_path = self.__media_helper.extract_audio(video_file_path, True, 16000)
            try:
                audio = whisper.load_audio(audio_file_path)
                self.__LOGGER.info("Start transcribing the audio...")
                verbose = False if Logger.VERBOSE and not Logger.QUIET else None
                self.__LOGGER.debug("Prompting with: '%s'" % initial_prompt)
                result = self.__model.transcribe(audio,
                                                 task="transcribe",
                                                 language=LANGUAGES[lang],
                                                 verbose=verbose,
                                                 word_timestamps=True,
                                                 initial_prompt=initial_prompt)
                self.__LOGGER.info("Finished transcribing the audio")
                srt_str = ""
                srt_idx = 1
                for segment in result["segments"]:
                    if max_char_length is not None and len(segment["text"]) > max_char_length:
                        srt_str, srt_idx = self._chunk_segment(segment, srt_str, srt_idx, max_char_length)
                    else:
                        if with_word_time_codes:
                            for word in segment["words"]:
                                srt_str += f"{srt_idx}\n" \
                                           f"{Utils.format_timestamp(word['start'])} --> {Utils.format_timestamp(word['end'])}\n" \
                                           f"{word['word'].strip().replace('-->', '->')}\n" \
                                           "\n"
                                srt_idx += 1
                        else:
                            srt_str += f"{srt_idx}\n" \
                                       f"{Utils.format_timestamp(segment['words'][0]['start'])} --> {Utils.format_timestamp(segment['words'][-1]['end'])}\n" \
                                       f"{segment['text'].strip().replace('-->', '->')}\n" \
                                       "\n"
                            srt_idx += 1
                subtitle = Subtitle.load_subrip_str(srt_str)
                subtitle, frame_rate = self.__on_frame_timecodes(subtitle, video_file_path)
                self.__LOGGER.debug("Generated the raw subtitle")
                return subtitle, frame_rate
            finally:
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
        else:
            raise NotImplementedError(f"{self.__recipe} ({self.__flavour}) is not supported")

    def transcribe_with_subtitle_as_prompts(self,
                                            video_file_path: str,
                                            subtitle_file_path: str,
                                            language_code: str,
                                            max_char_length: Optional[int] = None,
                                            use_prior_prompting: bool = False,
                                            with_word_time_codes: bool = False) -> Tuple[Subtitle, Optional[float]]:
        """Transcribe an audiovisual file and generate subtitles using the original subtitle (with accurate time codes) as prompts.


        Arguments:
            video_file_path {string} -- The input video file path.
            subtitle_file_path {string} -- The input subtitle file path to provide prompts.
            language_code {string} -- An alpha 3 language code derived from ISO 639-3.
            max_char_length {int} -- Optional Maximum number of characters for each generated subtitle segment.
            use_prior_prompting {bool} -- Whether to use the previous subtitle cue as the current prompt.
            with_word_time_codes {bool} -- True to output time codes for each word (default: {False})

        Returns:
            tuple: Generated subtitle after transcription and the detected frame rate

        Raises:
            TranscriptionException: Thrown when transcription is failed.
            NotImplementedError: Thrown when the LLM recipe is not supported.
        """
        if self.__recipe == "whisper":
            lang = Utils.get_iso_639_alpha_2(language_code)
            if lang not in LANGUAGES:
                raise TranscriptionException(
                    f'"{language_code}" is not supported by {self.__recipe} ({self.__flavour})')
            audio_file_path = self.__media_helper.extract_audio(video_file_path, True, 16000)
            subtitle = Subtitle.load(subtitle_file_path)
            segment_paths: List[str] = []
            try:
                srt_str = ""
                srt_idx = 1
                self.__LOGGER.info("Start transcribing the audio...")
                segment_paths = []
                args = []
                longest_segment_char_length = 0
                prev_sub_text = ""
                for sub in tqdm(subtitle.subs, desc="Extracting audio segments"):
                    segment_path, _ = self.__media_helper.extract_audio_from_start_to_end(audio_file_path, str(sub.start), str(sub.end))
                    segment_paths.append(segment_path)
                    if use_prior_prompting:
                        args.append((segment_path, sub.text if prev_sub_text == "" else prev_sub_text, self.__lock, self.__LOGGER))
                        prev_sub_text = sub.text
                    else:
                        args.append((segment_path, sub.text, self.__lock, self.__LOGGER))
                    if len(sub.text) > longest_segment_char_length:
                        longest_segment_char_length = len(sub.text)
                max_subtitle_char_length = max_char_length or longest_segment_char_length

                max_workers = math.ceil(float(os.getenv("MAX_WORKERS", mp.cpu_count() / 2)))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(partial(self._whisper_transcribe, model=self.__model, lang=lang), args))

                for sub, result in zip(subtitle.subs, results):
                    original_start_in_secs = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000.0
                    original_end_in_secs = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000.0
                    if len(result["segments"]) == 0:
                        srt_str += f"{srt_idx}\n" \
                                   f"{Utils.format_timestamp(original_start_in_secs)} --> {Utils.format_timestamp(original_end_in_secs)}\n" \
                                   f"{sub.text.strip().replace('-->', '->')}\n" \
                                   "\n"
                        srt_idx += 1
                    else:
                        for segment in result["segments"]:
                            if segment["end"] <= segment["start"]:
                                continue
                            segment_end = min(original_start_in_secs + segment["end"], original_end_in_secs)
                            if with_word_time_codes:
                                for word in segment["words"]:
                                    word_end = original_start_in_secs + word["end"]
                                    if word_end > segment_end:
                                        break
                                    srt_str += f"{srt_idx}\n" \
                                               f"{Utils.format_timestamp(original_start_in_secs + word['start'])} --> {Utils.format_timestamp(word_end)}\n" \
                                               f"{word['word'].strip().replace('-->', '->')}\n" \
                                               "\n"
                                    srt_idx += 1
                            else:
                                if len(segment["text"]) > max_subtitle_char_length:
                                    srt_str, srt_idx = self._chunk_segment(segment,
                                                                           srt_str,
                                                                           srt_idx,
                                                                           max_subtitle_char_length,
                                                                           original_start_in_secs,
                                                                           original_end_in_secs)
                                else:
                                    srt_str += f"{srt_idx}\n" \
                                               f"{Utils.format_timestamp(original_start_in_secs + segment['start'])} --> {Utils.format_timestamp(segment_end)}\n" \
                                               f"{segment['text'].strip().replace('-->', '->')}\n" \
                                               "\n"
                                    srt_idx += 1
                            if segment_end == original_end_in_secs:
                                break
                self.__LOGGER.info("Finished transcribing the audio")
                subtitle = Subtitle.load_subrip_str(srt_str)
                subtitle, frame_rate = self.__on_frame_timecodes(subtitle, video_file_path)
                self.__LOGGER.debug("Generated the raw subtitle")
                return subtitle, frame_rate
            finally:
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
                for segment_path in segment_paths:
                    if os.path.exists(segment_path):
                        os.remove(segment_path)
        else:
            raise NotImplementedError(f"{self.__recipe} ({self.__flavour}) is not supported")

    @staticmethod
    def _whisper_transcribe(args: Tuple, model: Whisper, lang: str) -> Dict:
        segment_path, sub_text, lock, logger = args
        verbose = False if Logger.VERBOSE and not Logger.QUIET else None
        try:
            waveform, _ = torchaudio.load(segment_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0)
            waveform = waveform.numpy().astype(np.float32)
            with lock:
                result = model.transcribe(waveform,
                                          task="transcribe",
                                          language=LANGUAGES[lang],
                                          verbose=verbose,
                                          initial_prompt=sub_text,
                                          word_timestamps=True)
                logger.debug("Segment transcribed : %s", result)
            return result
        except Exception as e:
            logger.warning(f"Error while transcribing segment: {e}")
            return {"segments": []}

    @staticmethod
    def _chunk_segment(segment: Dict,
                       srt_str: str,
                       srt_idx: int,
                       max_subtitle_char_length: int,
                       start_offset: float = 0.0,
                       end_ceiling: float = float("inf")) -> Tuple[str, int]:
        chunked_text = ""
        chunk_start_in_secs = 0.0
        chunk_end_in_secs = 0.0
        chunk_char_length = 0

        for word in segment["words"]:
            if chunk_char_length + len(word["word"]) > max_subtitle_char_length and chunked_text.strip() != "":
                srt_str += f"{srt_idx}\n" \
                           f"{Utils.format_timestamp(start_offset + chunk_start_in_secs)} --> {Utils.format_timestamp(min(start_offset + chunk_end_in_secs, end_ceiling))}\n" \
                           f"{chunked_text.strip().replace('-->', '->')}\n" \
                           "\n"
                srt_idx += 1
                chunked_text = word["word"]
                chunk_start_in_secs = word["start"]
                chunk_char_length = len(word["word"])
            else:
                if chunk_start_in_secs == 0.0:
                    chunk_start_in_secs = word["start"]
                chunked_text += word["word"]
                chunk_char_length += len(word["word"])
            chunk_end_in_secs = word["end"]

        if len(chunked_text) > 0:
            srt_str += f"{srt_idx}\n" \
                       f"{Utils.format_timestamp(start_offset + chunk_start_in_secs)} --> {Utils.format_timestamp(min(start_offset + chunk_end_in_secs, end_ceiling))}\n" \
                       f"{chunked_text.strip().replace('-->', '->')}\n" \
                       "\n"
            srt_idx += 1

        return srt_str, srt_idx

    def __on_frame_timecodes(self, subtitle: Subtitle, video_file_path: str) -> Tuple[Subtitle, Optional[float]]:
        frame_rate = None
        try:
            frame_rate = self.__media_helper.get_frame_rate(video_file_path)
            frame_duration = 1.0 / frame_rate
            for sub in subtitle.subs:
                start_seconds = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000.0
                end_seconds = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000.0
                start_frames = int(start_seconds / frame_duration)
                end_frames = int(end_seconds / frame_duration)
                sub.start = SubRipTime(seconds=start_frames * frame_duration)
                sub.end = SubRipTime(seconds=end_frames * frame_duration)
        except NoFrameRateException:
            self.__LOGGER.warning("Cannot detect the frame rate for %s" % video_file_path)
        return subtitle, frame_rate
