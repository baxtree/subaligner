import math
import pycountry
import time
from copy import deepcopy
from pysrt import SubRipItem
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from typing import List, Generator
from .singleton import Singleton
from .logger import Logger


class Translator(metaclass=Singleton):
    """Translate subtitles.
    """

    __TENSOR_TYPE = "pt"
    __OPUS_MT = "Helsinki-NLP/opus-mt-{}-{}"
    __OPUS_TATOEBA = "Helsinki-NLP/opus-tatoeba-{}-{}"
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

    def __init__(self, src_language, tgt_language) -> None:
        """Initialiser for the subtitle translation.

        Arguments:
            src_language {string} -- The source language code derived from ISO 639-3.
            tgt_language {string} -- The target language code derived from ISO 639-3.

        Raises:
            NotImplementedError -- Thrown when the model of the specified language pair is not found.
        """

        self.__LOGGER = Logger().get_logger(__name__)
        self.__initialise_model(src_language, tgt_language)

    @staticmethod
    def get_iso_639_alpha_2(language_code: str) -> str:
        """Find the alpha 2 language code based on an alpha 3 one.

        Arguments:
            language_code {string} -- An alpha 3 language code derived from ISO 639-3.

        Returns:
            string -- The alpha 2 language code if exists otherwise the alpha 3 one.

        Raises:
            ValueError -- Thrown when the input language code cannot be recognised.
        """

        lang = pycountry.languages.get(alpha_3=language_code)
        if lang is None:
            return language_code
        elif hasattr(lang, "alpha_2"):
            return lang.alpha_2
        else:
            return lang.alpha_3

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

    def translate(self, subs: List[SubRipItem]) -> List[SubRipItem]:
        """Translate a list of subtitle cues.

        Arguments:
            subs {list} -- A list of SubRipItems.

        Returns:
            {list} -- A list of new SubRipItems holding the translation results.
        """

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

    def __initialise_model(self, src_lang: str, tgt_lang: str) -> None:
        src_lang = Translator.normalise_single(src_lang)
        tgt_lang = Translator.normalise_single(tgt_lang)
        src_lang, tgt_lang = Translator.normalise_pair(src_lang, tgt_lang)
        try:
            mt_model_name = Translator.__OPUS_MT.format(Translator.get_iso_639_alpha_2(src_lang), Translator.get_iso_639_alpha_2(tgt_lang))
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            self.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = Translator.__OPUS_MT.format(src_lang, Translator.get_iso_639_alpha_2(tgt_lang))
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            self.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = Translator.__OPUS_MT.format(Translator.get_iso_639_alpha_2(src_lang), tgt_lang)
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            self.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = Translator.__OPUS_MT.format(src_lang, tgt_lang)
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            self.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = Translator.__OPUS_TATOEBA.format(Translator.get_iso_639_alpha_2(src_lang), Translator.get_iso_639_alpha_2(tgt_lang))
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            self.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = Translator.__OPUS_TATOEBA.format(src_lang, Translator.get_iso_639_alpha_2(tgt_lang))
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            self.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = Translator.__OPUS_TATOEBA.format(Translator.get_iso_639_alpha_2(src_lang), tgt_lang)
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            self.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = Translator.__OPUS_TATOEBA.format(src_lang, tgt_lang)
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            self.__LOGGER.debug("Cannot download the MT model %s" % mt_model_name)
            message = 'Cannot find the MT model for source language "{}" and destination language "{}"'.format(src_lang, tgt_lang)
            self.__LOGGER.error(message)
            raise NotImplementedError(message)

    def __download_mt_model(self, mt_model_name: str) -> None:
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
