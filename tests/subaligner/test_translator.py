import os
import unittest
from mock import Mock, patch
from parameterized import parameterized
from subaligner.subtitle import Subtitle
from subaligner.llm import TranslationRecipe, HelsinkiNLPFlavour, WhisperFlavour, FacebookMbartFlavour
from subaligner.exception import TranslationException
from subaligner.translator import Translator as Undertest


class TranslatorTests(unittest.TestCase):

    def setUp(self):
        self.srt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.srt"
        )

    @patch("transformers.MarianTokenizer.from_pretrained")
    @patch("transformers.MarianMTModel.from_pretrained")
    def test_translate_hel_nlp(self, model_from_pretrained, tokenizer_from_pretrained):
        subs = Subtitle.load(self.srt_file_path).subs
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": None, "attention_mask": None}
        mock_tokenizer.decode.return_value = "translated"
        mock_model = Mock()
        mock_model.generate.return_value = [None] * len(subs)
        tokenizer_from_pretrained.return_value = mock_tokenizer
        model_from_pretrained.return_value = mock_model

        undertest = Undertest("eng", "zho", recipe=TranslationRecipe.HELSINKI_NLP.value)
        translated_subs = undertest.translate(subs)

        self.assertEqual(["translated"] * len(subs), [*map(lambda x: x.text, translated_subs)])

    @patch("whisper.load_audio")
    @patch("whisper.load_model")
    def test_translate_whisper(self, load_model, load_audio):
        subs = Subtitle.load(self.srt_file_path).subs
        model = Mock()
        load_model.return_value = model
        model.transcribe.return_value = {"segments": [{"start": 0, "end": 1, "text": "translated"}]}

        undertest = Undertest("eng", "eng", recipe=TranslationRecipe.WHISPER.value, flavour=WhisperFlavour.TINY.value)
        translated_subs = undertest.translate(subs, "video_path")

        self.assertEqual(["translated"], [*map(lambda x: x.text, translated_subs)])

    @patch("transformers.MBart50TokenizerFast.from_pretrained")
    @patch("transformers.MBartForConditionalGeneration.from_pretrained")
    def test_translate_fb_mbart(self, model_from_pretrained, tokenizer_from_pretrained):
        subs = Subtitle.load(self.srt_file_path).subs
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": None, "attention_mask": None}
        mock_tokenizer.decode.return_value = "translated"
        mock_tokenizer.lang_code_to_id = {"zh_CN": 250025}
        mock_model = Mock()
        mock_model.generate.return_value = [None] * len(subs)
        tokenizer_from_pretrained.return_value = mock_tokenizer
        model_from_pretrained.return_value = mock_model

        undertest = Undertest("eng", "zho", recipe=TranslationRecipe.FACEBOOK_MBART.value, flavour=FacebookMbartFlavour.LARGE.value)
        translated_subs = undertest.translate(subs)

        self.assertEqual(["translated"] * len(subs), [*map(lambda x: x.text, translated_subs)])

    @patch("transformers.MarianTokenizer.from_pretrained", side_effect=OSError)
    def test_throw_exception_on_translating_subs(self, mock_tokenizer_from_pretrained):
        subs = Subtitle.load(self.srt_file_path).subs
        try:
            Undertest("eng", "aar").translate(subs)
        except Exception as e:
            self.assertTrue(mock_tokenizer_from_pretrained.called)
            self.assertTrue(isinstance(e, NotImplementedError))
        else:
            self.fail("Should have thrown exception")

    @patch("whisper.load_model")
    def test_throw_exception_on_unsupported_whisper_translation_target(self, load_model):
        subs = Subtitle.load(self.srt_file_path).subs
        model = Mock()
        load_model.return_value = model
        model.transcribe.return_value = {"segments": [{"start": 0, "end": 1, "text": "translated"}]}

        try:
            Undertest("eng", "unk", recipe=TranslationRecipe.WHISPER.value, flavour=WhisperFlavour.TINY.value).translate(subs, "video_path")
        except Exception as e:
            self.assertTrue(isinstance(e, TranslationException))
        else:
            self.fail("Should have thrown exception")
