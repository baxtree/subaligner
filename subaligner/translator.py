import math
import time
import whisper
from copy import deepcopy
from typing import List, Generator, Optional
from pysrt import SubRipItem
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from whisper.tokenizer import LANGUAGES
from .singleton import Singleton
from .llm import TranslationRecipe, HelsinkiNLPFlavour, WhisperFlavour
from .utils import Utils
from .subtitle import Subtitle
from .logger import Logger
from .exception import TranslationException


class Translator(object):
    """Translate subtitles.
    """

    __TENSOR_TYPE = "pt"
    __TRANSLATING_BATCH_SIZE = 10
    __LANGUAGE_CODE_MAPPER = {
        "bos": "zls",
        "cmn": "zho",
        "gla": "cel",
        "grc": "grk",
        "guj": "inc",
        "ina": "art",
        "jbo": "art",
        "kan": "dra",
        "kir": "trk",
        "lat": "itc",
        "lfn": "art",
        "mya": "sit",
        "nep": "inc",
        "ori": "inc",
        "sin": "inc",
        "srp": "zls",
        "tam": "dra",
        "tat": "trk",
        "tel": "dra",
        "yue": "zho"
    }
    __LANGUAGE_PAIR_MAPPER = {
        "eng-jpn": "eng-jap",
        "jpn-eng": "jap-eng"
    }

    def __init__(self,
                 src_language: str,
                 tgt_language: str,
                 recipe: str = TranslationRecipe.HELSINKI_NLP.value,
                 flavour: Optional[str] = None) -> None:
        """Initialiser for the subtitle translation.

        Arguments:
            src_language {string} -- The source language code derived from ISO 639-3.
            tgt_language {string} -- The target language code derived from ISO 639-3.
            recipe {string} -- the LLM recipe used for transcribing video files (default: "helsinki-nlp").
            flavour {string} -- the flavour variation for a specific LLM recipe (default: None).

        Raises:
            NotImplementedError -- Thrown when the model of the specified language pair is not found.
        """

        self.__LOGGER = Logger().get_logger(__name__)
        if recipe not in [r.value for r in TranslationRecipe]:
            raise NotImplementedError(f"Unknown recipe: {recipe}")
        self.__recipe = recipe
        self.__tgt_language = tgt_language
        self.__initialise_model(src_language, tgt_language, recipe, flavour)

    @staticmethod
    def normalise_single(language_code: str) -> str:
        """Normalise a single language code.

        Arguments:
            language_code {string} -- A language code derived from ISO 639-3.

        Returns:
            string -- The language code understood by the language model.
        """

        return Translator.__LANGUAGE_CODE_MAPPER[language_code] if language_code in Translator.__LANGUAGE_CODE_MAPPER else language_code

    @staticmethod
    def normalise_pair(src_language: str, tgt_language: str) -> List[str]:
        """Normalise a pair of language codes.

        Arguments:
            src_language {string} -- The source language code derived from ISO 639-3.
            tgt_language {string} -- The target language code derived from ISO 639-3.

        Returns:
            list -- The language code pair understood by the language model.
        """

        if "{}-{}".format(src_language, tgt_language) in Translator.__LANGUAGE_PAIR_MAPPER:
            return Translator.__LANGUAGE_PAIR_MAPPER["{}-{}".format(src_language, tgt_language)].split("-")
        else:
            return [src_language, tgt_language]

    def translate(self, subs: List[SubRipItem], video_file_path: Optional[str] = None) -> List[SubRipItem]:
        """Translate a list of subtitle cues.

        Arguments:
            subs {list} -- A list of SubRipItems.
            video_file_path {string} -- The input video file path (default: None)..

        Returns:
            {list} -- A list of new SubRipItems holding the translation results.
        """

        if self.__recipe == TranslationRecipe.HELSINKI_NLP.value:
            translated_texts = []
            self.lang_model.eval()
            new_subs = deepcopy(subs)
            src_texts = [sub.text for sub in new_subs]
            num_of_batches = math.ceil(len(src_texts) / Translator.__TRANSLATING_BATCH_SIZE)
            self.__LOGGER.info("Translating %s subtitle cue(s)..." % len(src_texts))
            for batch in tqdm(Translator.__batch(src_texts, Translator.__TRANSLATING_BATCH_SIZE), total=num_of_batches):
                input_ids = self.tokenizer(batch, return_tensors=Translator.__TENSOR_TYPE, padding=True)
                translated = self.lang_model.generate(**input_ids)
                translated_texts.extend([self.tokenizer.decode(t, skip_special_tokens=True) for t in translated])
            for index in range(len(new_subs)):
                new_subs[index].text = translated_texts[index]
            self.__LOGGER.info("Subtitle translated")
            return new_subs
        elif self.__recipe == TranslationRecipe.WHISPER.value:
            assert video_file_path is not None
            lang = Utils.get_iso_639_alpha_2(self.__tgt_language)
            if lang not in LANGUAGES:
                raise TranslationException(f'"{self.__tgt_language}" is not supported by {self.__recipe}')
            audio = whisper.load_audio(video_file_path)
            self.__LOGGER.debug("Start translating the audio...")
            result = self.lang_model.transcribe(audio, task="translate", language=LANGUAGES[lang])
            self.__LOGGER.info("Finished translating the audio")
            srt_str = ""
            for i, segment in enumerate(result["segments"], start=1):
                srt_str += f"{i}\n" \
                           f"{Utils.format_timestamp(segment['start'])} --> {Utils.format_timestamp(segment['end'])}\n" \
                           f"{segment['text'].strip().replace('-->', '->')}\n" \
                           "\n"
            subtitle = Subtitle.load_subrip_str(srt_str)
            return subtitle.subs
        else:
            return []

    def __initialise_model(self, src_lang: str, tgt_lang: str, recipe: str, flavour: Optional[str]) -> None:
        if recipe == TranslationRecipe.HELSINKI_NLP.value:
            src_lang = Translator.normalise_single(src_lang)
            tgt_lang = Translator.normalise_single(tgt_lang)
            src_lang, tgt_lang = Translator.normalise_pair(src_lang, tgt_lang)

            if self.__download_mt_model(src_lang, tgt_lang, HelsinkiNLPFlavour.OPUS_MT.value):
                return
            elif self.__download_mt_model(src_lang, tgt_lang, HelsinkiNLPFlavour.OPUS_TATOEBA.value):
                return
            elif self.__download_mt_model(src_lang, tgt_lang, HelsinkiNLPFlavour.OPUS_MT_TC_BIG.value):
                return
            else:
                message = 'Cannot find the MT model for source language "{}" and destination language "{}"'.format(src_lang, tgt_lang)
                self.__LOGGER.error(message)
                raise NotImplementedError(message)
        elif recipe == TranslationRecipe.WHISPER.value:
            if flavour in [f.value for f in WhisperFlavour]:
                # self.__download_whisper_model(flavour)
                self.__download_whisper_model("medium")  # works for translation target other than English
            else:
                raise NotImplementedError(f"Unknown {recipe} flavour: {flavour}")

    def __download_mt_model(self, src_lang: str, tgt_lang: str, flavour: str) -> bool:
        try:
            mt_model_name = flavour.format(Utils.get_iso_639_alpha_2(src_lang), Utils.get_iso_639_alpha_2(tgt_lang))
            self.__download_by_mt_name(mt_model_name)
            return True
        except OSError:
            self.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = flavour.format(src_lang, Utils.get_iso_639_alpha_2(tgt_lang))
            self.__download_by_mt_name(mt_model_name)
            return True
        except OSError:
            self.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = flavour.format(Utils.get_iso_639_alpha_2(src_lang), tgt_lang)
            self.__download_by_mt_name(mt_model_name)
            return True
        except OSError:
            self.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = flavour.format(src_lang, tgt_lang)
            self.__download_by_mt_name(mt_model_name)
            return True
        except OSError:
            self.__log_and_back_off(mt_model_name)
        return False

    def __download_whisper_model(self, flavour: str) -> None:
        self.lang_model = whisper.load_model(flavour)

    def __download_by_mt_name(self, mt_model_name: str) -> None:
        self.__LOGGER.debug("Trying to download the MT model %s" % mt_model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
        self.lang_model = MarianMTModel.from_pretrained(mt_model_name)
        self.__LOGGER.debug("MT model %s downloaded" % mt_model_name)

    def __log_and_back_off(self, mt_model_name: str):
        self.__LOGGER.debug("Cannot download the MT model %s" % mt_model_name)
        time.sleep(1)

    @staticmethod
    def __batch(data: List, size: int = 1) -> Generator:
        total = len(data)
        for ndx in range(0, total, size):
            yield data[ndx:min(ndx + size, total)]
