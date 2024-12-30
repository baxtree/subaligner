import os
import whisper
import torch
from typing import Tuple, Optional
from pysrt import SubRipTime
from whisper.tokenizer import LANGUAGES
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

    def transcribe(self, video_file_path: str, language_code: str, initial_prompt: Optional[str] = None) -> Tuple[Subtitle, Optional[float]]:
        """Transcribe an audiovisual file and generate subtitles.

        Arguments:
            video_file_path {string} -- The input video file path.
            language_code {string} -- An alpha 3 language code derived from ISO 639-3.
            initial_prompt {string} -- Optional text to provide the transcribing context or specific phrases.

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
                result = self.__model.transcribe(audio, task="transcribe", language=LANGUAGES[lang], verbose=verbose, initial_prompt=initial_prompt)
                self.__LOGGER.info("Finished transcribing the audio")
                srt_str = ""
                for i, segment in enumerate(result["segments"], start=1):
                    srt_str += f"{i}\n" \
                               f"{Utils.format_timestamp(segment['start'])} --> {Utils.format_timestamp(segment['end'])}\n" \
                               f"{segment['text'].strip().replace('-->', '->')}\n" \
                               "\n"
                subtitle = Subtitle.load_subrip_str(srt_str)
                subtitle, frame_rate = self.__on_frame_timecodes(subtitle, video_file_path)
                self.__LOGGER.debug("Generated the raw subtitle")
                return subtitle, frame_rate
            finally:
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
        else:
            raise NotImplementedError(f"{self.__recipe} ({self.__flavour}) is not supported")

    def transcribe_with_subtitle_as_prompts(self, video_file_path: str, subtitle_file_path: str, language_code: str) -> Tuple[Subtitle, Optional[float]]:
        """Transcribe an audiovisual file and generate subtitles using the original subtitle as prompts.

        Arguments:
            video_file_path {string} -- The input video file path.
            subtitle_file_path {string} -- The input subtitle file path to provide prompts.
            language_code {string} -- An alpha 3 language code derived from ISO 639-3.

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
            segment_paths = []
            try:
                srt_str = ""
                srt_idx = 1
                self.__LOGGER.info("Start transcribing the audio...")
                verbose = False if Logger.VERBOSE and not Logger.QUIET else None
                for sub in subtitle.subs:
                    segment_path, _ = self.__media_helper.extract_audio_from_start_to_end(audio_file_path, str(sub.start), str(sub.end))
                    segment_paths.append(segment_path)
                    audio = whisper.load_audio(segment_path)
                    result = self.__model.transcribe(audio, task="transcribe", language=LANGUAGES[lang], verbose=verbose, initial_prompt=sub.text)
                    original_start_in_secs = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000.0
                    original_end_in_secs = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000.0
                    for segment in result["segments"]:
                        if segment["end"] <= segment["start"]:
                            continue
                        srt_str += f"{srt_idx}\n" \
                                   f"{Utils.format_timestamp(original_start_in_secs + segment['start'])} --> {Utils.format_timestamp(min(original_start_in_secs + segment['end'], original_end_in_secs))}\n" \
                                   f"{segment['text'].strip().replace('-->', '->')}\n" \
                                   "\n"
                        srt_idx += 1
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
