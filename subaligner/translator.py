import math
import pycountry
import time
from copy import deepcopy
from pysrt import SubRipItem
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from typing import List
from .singleton import Singleton
from .logger import Logger


class Translator(Singleton):

    __LOGGER = Logger().get_logger(__name__)
    __TENSOR_TYPE = "pt"
    __OPUS_MT = "Helsinki-NLP/opus-mt-{}-{}"
    __OPUS_TATOEBA = "Helsinki-NLP/opus-tatoeba-{}-{}"
    __TRANSLATING_BATCH_SIZE = 10

    def __init__(self, source_language, target_lang) -> None:
        self.__initialise_model(source_language, target_lang)

    @staticmethod
    def get_iso_639_alpha_2(language_code: str) -> str:
        lang = pycountry.languages.get(alpha_3=language_code)
        if lang is None:
            raise ValueError("Cannot recognise %s as an ISO 639-3 language code" % language_code)
        elif hasattr(lang, "alpha_2"):
            return lang.alpha_2
        else:
            return lang.alpha_3

    def translate_subs(self, subs: List[SubRipItem]) -> List[SubRipItem]:
        translated_texts = []
        self.lang_model.eval()
        new_subs = deepcopy(subs)
        src_texts = [sub.text for sub in new_subs]
        num_of_batches = math.ceil(len(src_texts) / Translator.__TRANSLATING_BATCH_SIZE)
        Translator.__LOGGER.info("Translating %s subtitle cue(s)..." % len(src_texts))
        for batch in tqdm(Translator.__batch(src_texts, Translator.__TRANSLATING_BATCH_SIZE), total=num_of_batches):
            tokenizer = self.tokenizer(batch, return_tensors=Translator.__TENSOR_TYPE, padding=True)
            translated = self.lang_model.generate(**tokenizer)
            translated_texts.extend([self.tokenizer.decode(t, skip_special_tokens=True) for t in translated])
        for index in range(len(new_subs)):
            new_subs[index].text = translated_texts[index]
        Translator.__LOGGER.info("Subtitle translated")
        return new_subs

    def __initialise_model(self, src_lang, des_lang):
        try:
            mt_model_name = Translator.__OPUS_MT.format(Translator.get_iso_639_alpha_2(src_lang), Translator.get_iso_639_alpha_2(des_lang))
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            Translator.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = Translator.__OPUS_MT.format(src_lang, Translator.get_iso_639_alpha_2(des_lang))
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            Translator.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = Translator.__OPUS_MT.format(Translator.get_iso_639_alpha_2(src_lang), des_lang)
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            Translator.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = Translator.__OPUS_MT.format(src_lang, des_lang)
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            Translator.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = Translator.__OPUS_TATOEBA.format(Translator.get_iso_639_alpha_2(src_lang), Translator.get_iso_639_alpha_2(des_lang))
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            Translator.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = Translator.__OPUS_TATOEBA.format(src_lang, Translator.get_iso_639_alpha_2(des_lang))
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            Translator.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = Translator.__OPUS_TATOEBA.format(Translator.get_iso_639_alpha_2(src_lang), des_lang)
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            Translator.__log_and_back_off(mt_model_name)
        try:
            mt_model_name = Translator.__OPUS_TATOEBA.format(src_lang, des_lang)
            self.__download_mt_model(mt_model_name)
            return
        except OSError:
            Translator.__LOGGER.debug("Cannot download the MT model %s" % mt_model_name)
            message = 'Cannot find the MT model for source language "{}" and destination language "{}"'.format(src_lang, des_lang)
            Translator.__LOGGER.error(message)
            raise NotImplementedError(message)

    def __download_mt_model(self, mt_model_name):
        Translator.__LOGGER.debug("Trying to download the MT model %s" % mt_model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
        self.lang_model = MarianMTModel.from_pretrained(mt_model_name)
        Translator.__LOGGER.debug("MT model %s downloaded" % mt_model_name)

    @staticmethod
    def __log_and_back_off(mt_model_name):
        Translator.__LOGGER.debug("Cannot download the MT model %s" % mt_model_name)
        time.sleep(1)

    @staticmethod
    def __batch(iterable, size=1):
        total = len(iterable)
        for ndx in range(0, total, size):
            yield iterable[ndx:min(ndx + size, total)]
