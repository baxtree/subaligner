import math
import time
import whisper
from copy import deepcopy
from typing import List, Generator, Optional, Tuple
from pysrt import SubRipItem
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    MarianMTModel,
    MarianTokenizer,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
)
from whisper.tokenizer import LANGUAGES
from .singleton import Singleton
from .llm import TranslationRecipe, HelsinkiNLPFlavour, WhisperFlavour, FacebookMbartFlavour
from .utils import Utils
from .subtitle import Subtitle
from .logger import Logger
from .exception import TranslationException


class Translator(object):
    """Translate subtitles.
    """

    __TENSOR_TYPE = "pt"
    __TRANSLATING_BATCH_SIZE = 10
    __HELSINKI_LANGUAGE_CODE_MAPPER = {
        "bos": "zls", "cmn": "zho", "gla": "cel", "grc": "grk", "guj": "inc", "ina": "art", "jbo": "art", "kan": "dra",
        "kir": "trk", "lat": "itc", "lfn": "art", "mya": "sit", "nep": "inc", "ori": "inc", "sin": "inc", "srp": "zls",
        "tam": "dra", "tat": "trk", "tel": "dra", "yue": "zho"
    }
    __HELSINKI_LANGUAGE_PAIR_MAPPER = {
        "eng-jpn": "eng-jap",
        "jpn-eng": "jap-eng"
    }
    __MBART_LANGUAGE_CODE_MAPPER = {
        "ara": "ar_AR", "ces": "cs_CZ", "deu": "de_DE", "eng": "en_XX", "spa": "es_XX", "est": "et_EE", "fin": "fi_FI",
        "fra": "fr_XX", "guj": "gu_IN", "hin": "hi_IN", "ita": "it_IT", "jpn": "ja_XX", "kaz": "kk_KZ", "kor": "ko_KR",
        "lit": "lt_LT", "lav": "lv_LV", "mya": "my_MM", "nep": "ne_NP", "nld": "nl_XX", "ron": "ro_RO", "rus": "ru_RU",
        "sin": "si_LK", "tur": "tr_TR", "vie": "vi_VN", "zho": "zh_CN", "afr": "af_ZA", "aze": "az_AZ", "ben": "bn_IN",
        "fas": "fa_IR", "heb": "he_IL", "hrv": "hr_HR", "ind": "id_ID", "kat": "ka_GE", "khm": "km_KH", "mkd": "mk_MK",
        "mal": "ml_IN", "mon": "mn_MN", "mar": "mr_IN", "pol": "pl_PL", "pus": "ps_AF", "por": "pt_XX", "swe": "sv_SE",
        "swa": "sw_KE", "tam": "ta_IN", "tel": "te_IN", "tha": "th_TH", "tgl": "tl_XX", "ukr": "uk_UA", "urd": "ur_PK",
        "xho": "xh_ZA", "glg": "gl_ES", "slv": "sl_SI"
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
        self.__src_language = src_language
        self.__tgt_language = tgt_language
        self.__tokenizer: PreTrainedTokenizer = None
        self.__lang_model: PreTrainedModel = None
        self.__initialise_model(src_language, tgt_language, recipe, flavour)

    def translate(self,
                  subs: List[SubRipItem],
                  video_file_path: Optional[str] = None,
                  language_pair: Optional[Tuple[str, str]] = None) -> List[SubRipItem]:
        """Translate a list of subtitle cues.

        Arguments:
            subs {list} -- A list of SubRipItems.

        Keyword Arguments:
            video_file_path {string} -- The input video file path (default: None).
            language_pair {Tuple[str, str]} -- Used for overriding the default language pair (default: None).

        Returns:
            {list} -- A list of new SubRipItems holding the translation results.
        """

        if self.__recipe == TranslationRecipe.HELSINKI_NLP.value:
            if language_pair is not None:
                self.__LOGGER.debug(f"Language pair ignored: {language_pair}")
            translated_texts = []
            self.__lang_model.eval()
            new_subs = deepcopy(subs)
            src_texts = [sub.text for sub in new_subs]
            num_of_batches = math.ceil(len(src_texts) / Translator.__TRANSLATING_BATCH_SIZE)
            self.__LOGGER.info("Translating %s subtitle cue(s)..." % len(src_texts))
            for batch in tqdm(Translator.__batch(src_texts, Translator.__TRANSLATING_BATCH_SIZE), total=num_of_batches):
                input_ids = self.__tokenizer(batch, return_tensors=Translator.__TENSOR_TYPE, padding=True)
                translated = self.__lang_model.generate(**input_ids)
                translated_texts.extend([self.__tokenizer.decode(t, skip_special_tokens=True) for t in translated])
            for index in range(len(new_subs)):
                new_subs[index].text = translated_texts[index]
            self.__LOGGER.info("Subtitle translated")
            return new_subs
        elif self.__recipe == TranslationRecipe.WHISPER.value:
            assert video_file_path is not None
            lang = Utils.get_iso_639_alpha_2(self.__tgt_language)
            if lang not in LANGUAGES or lang != "en":
                raise TranslationException(f'"{self.__tgt_language}" is not supported by {self.__recipe} as a translation target by {self.__recipe}')
            audio = whisper.load_audio(video_file_path)
            self.__LOGGER.debug("Start translating the audio...")
            result = self.__lang_model.transcribe(audio, task="translate")
            self.__LOGGER.info("Finished translating the audio")
            srt_str = ""
            for i, segment in enumerate(result["segments"], start=1):
                srt_str += f"{i}\n" \
                           f"{Utils.format_timestamp(segment['start'])} --> {Utils.format_timestamp(segment['end'])}\n" \
                           f"{segment['text'].strip().replace('-->', '->')}\n" \
                           "\n"
            subtitle = Subtitle.load_subrip_str(srt_str)
            return subtitle.subs
        elif self.__recipe == TranslationRecipe.FACEBOOK_MBART.value:
            src_lang, tgt_lang = language_pair if language_pair is not None else (self.__src_language, self.__tgt_language)
            self.__tokenizer.src_lang = Translator.__MBART_LANGUAGE_CODE_MAPPER.get(src_lang, None)
            lang_code = Translator.__MBART_LANGUAGE_CODE_MAPPER.get(tgt_lang, None)
            if src_lang is None or tgt_lang is None:
                raise NotImplementedError(f"Language pair of {src_lang} and {src_lang} is not supported by {self.__recipe}")
            translated_texts = []
            self.__lang_model.eval()
            new_subs = deepcopy(subs)
            src_texts = [sub.text for sub in new_subs]
            num_of_batches = math.ceil(len(src_texts) / Translator.__TRANSLATING_BATCH_SIZE)
            self.__LOGGER.info("Translating %s subtitle cue(s)..." % len(src_texts))
            for batch in tqdm(Translator.__batch(src_texts, Translator.__TRANSLATING_BATCH_SIZE), total=num_of_batches):
                input_ids = self.__tokenizer(batch, return_tensors=Translator.__TENSOR_TYPE, padding=True)
                translated = self.__lang_model.generate(**input_ids, forced_bos_token_id=self.__tokenizer.lang_code_to_id[lang_code])
                translated_texts.extend([self.__tokenizer.decode(t, skip_special_tokens=True) for t in translated])
            for index in range(len(new_subs)):
                new_subs[index].text = translated_texts[index]
            self.__LOGGER.info("Subtitle translated")
            return new_subs
        else:
            return []

    def __initialise_model(self, src_lang: str, tgt_lang: str, recipe: str, flavour: Optional[str]) -> None:
        if recipe == TranslationRecipe.HELSINKI_NLP.value:
            src_lang = Translator.__HELSINKI_LANGUAGE_CODE_MAPPER.get(src_lang, src_lang)
            tgt_lang = Translator.__HELSINKI_LANGUAGE_CODE_MAPPER.get(tgt_lang, tgt_lang)
            lang_pair = "{}-{}".format(src_lang, tgt_lang)
            src_lang, tgt_lang = Translator.__HELSINKI_LANGUAGE_PAIR_MAPPER.get(lang_pair, lang_pair).split("-")

            if self.__download_mt_model(src_lang, tgt_lang, HelsinkiNLPFlavour.OPUS_MT.value):
                return
            elif self.__download_mt_model(src_lang, tgt_lang, HelsinkiNLPFlavour.OPUS_TATOEBA.value):
                return
            elif self.__download_mt_model(src_lang, tgt_lang, HelsinkiNLPFlavour.OPUS_MT_TC_BIG.value):
                return
            else:
                message = f'Cannot find the {recipe} MT model for source language "{src_lang}" and destination language "{tgt_lang}"'
                self.__LOGGER.error(message)
                raise NotImplementedError(message)
        elif recipe == TranslationRecipe.WHISPER.value:
            if flavour in [f.value for f in WhisperFlavour]:
                # self.__download_whisper_model(flavour)
                self.__download_whisper_model("medium")  # works for translation target other than English
            else:
                raise NotImplementedError(f"Unknown {recipe} flavour: {flavour}")
        elif recipe == TranslationRecipe.FACEBOOK_MBART.value:
            if flavour in [f.value for f in FacebookMbartFlavour]:
                self.__download_mbart_model(flavour)
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
        self.__lang_model = whisper.load_model(flavour)

    def __download_mbart_model(self, flavour: str) -> None:
        mbart_model_name = f"facebook/mbart-{flavour}-50-many-to-many-mmt"
        self.__LOGGER.debug("Trying to download the mBART model %s" % mbart_model_name)
        self.__tokenizer = MBart50TokenizerFast.from_pretrained(mbart_model_name)
        self.__lang_model = MBartForConditionalGeneration.from_pretrained(mbart_model_name)
        self.__LOGGER.debug("mBART model %s downloaded" % mbart_model_name)

    def __download_by_mt_name(self, mt_model_name: str) -> None:
        self.__LOGGER.debug("Trying to download the MT model %s" % mt_model_name)
        self.__tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
        self.__lang_model = MarianMTModel.from_pretrained(mt_model_name)
        self.__LOGGER.debug("MT model %s downloaded" % mt_model_name)

    def __log_and_back_off(self, mt_model_name: str):
        self.__LOGGER.debug("Cannot download the MT model %s" % mt_model_name)
        time.sleep(1)

    @staticmethod
    def __batch(data: List, size: int = 1) -> Generator:
        total = len(data)
        for ndx in range(0, total, size):
            yield data[ndx:min(ndx + size, total)]
