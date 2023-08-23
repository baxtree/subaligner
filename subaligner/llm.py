from enum import Enum


class TranscriptionRecipe(Enum):
    WHISPER = "whisper"


class TranslationRecipe(Enum):
    HELSINKI_NLP = "helsinki-nlp"
    WHISPER = "whisper"
    FACEBOOK_MBART = "facebook-mbart"


class WhisperFlavour(Enum):
    TINY = "tiny"
    TINY_EN = "tiny.en"
    SMALL = "small"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    BASE = "base"
    BASE_EN = "base.en"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE = "large"


class HelsinkiNLPFlavour(Enum):
    OPUS_MT = "Helsinki-NLP/opus-mt-{}-{}"
    OPUS_MT_TC_BIG = "Helsinki-NLP/opus-mt-tc-big-{}-{}"
    OPUS_TATOEBA = "Helsinki-NLP/opus-tatoeba-{}-{}"


class FacebookMbartFlavour(Enum):
    LARGE = "large"
