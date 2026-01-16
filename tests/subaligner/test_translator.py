import os
import torch
import unittest
import numpy as np
from mock import Mock, patch
from subaligner.subtitle import Subtitle
from subaligner.llm import TranslationRecipe, WhisperFlavour, FacebookMbartFlavour
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

    @patch("librosa.load")
    @patch("transformers.WhisperProcessor.from_pretrained")
    @patch("transformers.WhisperForConditionalGeneration.from_pretrained")
    @patch("subaligner.utils.Utils.vad_segment")
    @patch("torch.ones")
    @patch("torch.no_grad")
    def test_translate_whisper(self, mock_no_grad, mock_ones, mock_vad_segment, model_from_pretrained, processor_from_pretrained, mock_librosa_load):
        subs = Subtitle.load(self.srt_file_path).subs
        mock_processor = Mock()
        mock_model = Mock()
        mock_audio = np.array([0.5, 0.3, 0.8, 0.2], dtype=np.float32)
        processor_from_pretrained.return_value = mock_processor
        model_from_pretrained.return_value = mock_model
        mock_librosa_load.return_value = (mock_audio, 16000)
        mock_vad_segment.return_value = [(0, 4)]
        mock_input_features = Mock()
        mock_input_features.shape = (1, 80, 3000)
        mock_input_features.to.return_value = mock_input_features  # Mock the .to() method
        mock_processor.return_value = Mock(input_features=mock_input_features)
        mock_output = Mock()
        mock_output.sequences = Mock()
        mock_output.sequences.cpu.return_value = Mock(return_value=[[1, 2, 3]])
        mock_model.generate.return_value = mock_output
        mock_processor.batch_decode.return_value = ["translated"]
        mock_no_grad.return_value.__enter__ = Mock(return_value=None)
        mock_no_grad.return_value.__exit__ = Mock(return_value=None)
        mock_attention_mask = Mock()
        mock_ones.return_value = mock_attention_mask

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

    @patch("transformers.WhisperProcessor.from_pretrained")
    @patch("transformers.WhisperForConditionalGeneration.from_pretrained")
    def test_throw_exception_on_unsupported_whisper_translation_target(self, model_from_pretrained, processor_from_pretrained):
        subs = Subtitle.load(self.srt_file_path).subs
        mock_processor = Mock()
        mock_model = Mock()
        processor_from_pretrained.return_value = mock_processor
        model_from_pretrained.return_value = mock_model

        try:
            Undertest("eng", "unk", recipe=TranslationRecipe.WHISPER.value, flavour=WhisperFlavour.TINY.value).translate(subs, "video_path")
        except Exception as e:
            self.assertTrue(isinstance(e, TranslationException))
        else:
            self.fail("Should have thrown exception")
