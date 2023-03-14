import os
import whisper
from typing import Tuple, Optional
from pysrt import SubRipTime
from whisper.tokenizer import LANGUAGES
from .translator import Translator
from .subtitle import Subtitle
from .media_helper import MediaHelper
from .llm import TranscriptionRecipe, WhisperFlavour
from .singleton import Singleton
from .logger import Logger
from .utils import Utils
from .exception import NoFrameRateException, TranscriptionException


class Transcriber(object):
    """Transcribe audiovisual content for subtitle generation.
    """

    def __init__(self, recipe: str = TranscriptionRecipe.WHISPER.value, flavour: str = WhisperFlavour.SMALL.value) -> None:
        """Initialiser for the transcribing process.

        Arguments:
            recipe {string} -- the LLM recipe used for transcribing video files (default: "whisper").
            flavour {string} -- the flavour variation for a specific LLM recipe (default: "small").
        Raises:
            NotImplementedError -- Thrown when the LLM recipe is unknown.
        """
        if recipe not in [r.value for r in TranscriptionRecipe]:
            raise NotImplementedError(f"Unknown recipe: {recipe}")
        if recipe == TranscriptionRecipe.WHISPER.value:
            if flavour not in [f.value for f in WhisperFlavour]:
                raise NotImplementedError(f"Unknown {recipe} flavour: {flavour}")
            self.__model = whisper.load_model(flavour)
        self.__recipe = recipe
        self.__flavour = flavour
        self.__media_helper = MediaHelper()
        self.__LOGGER = Logger().get_logger(__name__)

    def transcribe(self, video_file_path: str, language_code: str) -> Tuple[Subtitle, Optional[float]]:
        """Transcribe an audiovisual file and generate subtitles.

        Arguments:
            video_file_path {string} -- The input video file path.
            language_code {string} -- An alpha 3 language code derived from ISO 639-3.
        Raises:
            TranscriptionException -- Thrown when transcription is failed.
            NotImplementedError -- Thrown when the LLM recipe is not supported.
        """
        if self.__recipe == "whisper":
            lang = Utils.get_iso_639_alpha_2(language_code)
            if lang not in LANGUAGES:
                raise TranscriptionException(f'"{language_code}" is not supported by {self.__recipe} ({self.__flavour})')
            audio_file_path = self.__media_helper.extract_audio(video_file_path, True, 16000)
            try:
                audio = whisper.load_audio(audio_file_path)
                self.__LOGGER.debug("Start transcribing the audio...")
                result = self.__model.transcribe(audio, task="transcribe", language=LANGUAGES[lang])
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
