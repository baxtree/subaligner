import os
import whisper
from enum import Enum
from typing import Tuple, Optional
from pysrt import SubRipTime
from whisper.tokenizer import LANGUAGES
from .translator import Translator
from .subtitle import Subtitle
from .media_helper import MediaHelper
from .logger import Logger
from .exception import NoFrameRateException, TranscriptionException


class Transcriber(object):
    """Transcribe audiovisual content for subtitle generation.
    """

    def __init__(self, recipe: str = "whisper", flavour: str = "small") -> None:
        """Initialiser for the transcribing process.

        Arguments:
            recipe {string} -- the LLM recipe used for transcribing video files (default: "whisper").
            flavour {string} -- the flavour variation for a specific LLM recipe (default: "small").
        Raises:
            NotImplementedError -- Thrown when the LLM recipe is unknown.
        """
        if recipe not in [r.value for r in Recipe]:
            raise NotImplementedError(f"Unknown recipe: {recipe}")
        if recipe == Recipe.whisper.value:
            if flavour not in [f.value for f in WhisperFlavour]:
                raise NotImplementedError(f"Unknown {recipe} flavour: {flavour}")
            self.__model = whisper.load_model(flavour)
        self.recipe = recipe
        self.flavour = flavour
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
        if self.recipe == "whisper":
            lang = Translator.get_iso_639_alpha_2(language_code)
            if lang not in LANGUAGES:
                raise TranscriptionException(f'"{language_code}" is not supported by {self.recipe} ({self.flavour})')
            audio_file_path = self.__media_helper.extract_audio(video_file_path, True, 16000)
            try:
                audio = whisper.load_audio(audio_file_path)
                self.__LOGGER.debug("Start transcribing the audio...")
                result = self.__model.transcribe(audio, task="transcribe", language=LANGUAGES[lang])
                self.__LOGGER.info("Finished transcribing the audio")
                srt_str = ""
                for i, segment in enumerate(result["segments"], start=1):
                    srt_str += f"{i}\n" \
                               f"{self.__format_timestamp(segment['start'])} --> {self.__format_timestamp(segment['end'])}\n" \
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
            raise NotImplementedError(f"{self.recipe} ({self.flavour}) is not supported")

    @staticmethod
    def __format_timestamp(seconds: float) -> str:
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)
        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000
        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000
        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000
        hours_marker = f"{hours:02d}:"
        return f"{hours_marker}{minutes:02d}:{seconds:02d},{milliseconds:03d}"

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


class Recipe(str, Enum):
    whisper = "whisper"


class WhisperFlavour(str, Enum):
    tiny = "tiny"
    tiny_en = "tiny.en"
    small = "small"
    medium = "medium"
    medium_en = "medium.en"
    base = "base"
    base_en = "base.en"
    large_v1 = "large-v1"
    large_v2 = "large-v2"
    large = "large"
