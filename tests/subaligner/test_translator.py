import os
import unittest
from mock import Mock, patch
from parameterized import parameterized
from transformers import MarianMTModel, MarianTokenizer
from subaligner.subtitle import Subtitle
from subaligner.translator import Translator as Undertest


class TranslatorTests(unittest.TestCase):

    def setUp(self):
        self.srt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.srt"
        )

    def test_get_iso_639_alpha_2(self):
        self.assertEqual("en", Undertest.get_iso_639_alpha_2("eng"))
        self.assertEqual("ada", Undertest.get_iso_639_alpha_2("ada"))
        self.assertEqual("unk", Undertest.get_iso_639_alpha_2("unk"))

    @patch("transformers.MarianMTModel.from_pretrained")
    @patch("transformers.MarianTokenizer.from_pretrained")
    def test_translate_subs(self, tokenizer_from_pretrained, model_from_pretrained):
        subs = Subtitle.load(self.srt_file_path).subs
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {"input_ids": None, "attention_mask": None}
        mock_tokenizer.decode.return_value = "translated"
        mock_model = Mock()
        mock_model.generate.return_value = [None] * len(subs)
        tokenizer_from_pretrained.return_value = mock_tokenizer
        model_from_pretrained.return_value = mock_model

        translated_subs = Undertest("eng", "zho").translate_subs(subs)

        self.assertEqual(["translated"] * len(subs), [*map(lambda x: x.text, translated_subs)])

    @parameterized.expand([
        ["bos", "zls"],
        ["cmn", "zho"],
        ["gla", "cel"],
        ["grc", "grk"],
        ["guj", "inc"],
        ["ina", "art"],
        ["jbo", "art"],
        ["kan", "dra"],
        ["kir", "trk"],
        ["lat", "itc"],
        ["lfn", "art"],
        ["mya", "sit"],
        ["nep", "inc"],
        ["ori", "inc"],
        ["sin", "inc"],
        ["srp", "zls"],
        ["tam", "dra"],
        ["tat", "trk"],
        ["tel", "dra"],
        ["yue", "zho"],
    ])
    def test_normalise_single(self, original, normalised):
        self.assertEqual(normalised, Undertest.normalise_single(original))

    @parameterized.expand([
        ["eng-jpn", "eng-jap"]
    ])
    def test_normalise_pair(self, original, normalised):
        self.assertEqual(normalised, "-".join(Undertest.normalise_pair(*original.split("-"))))

    @patch("transformers.MarianTokenizer.from_pretrained", side_effect=OSError)
    def test_throw_exception_on_translating_subs(self, mock_tokenizer_from_pretrained):
        subs = Subtitle.load(self.srt_file_path).subs
        try:
            Undertest("eng", "aar").translate_subs(subs)
        except Exception as e:
            self.assertTrue(mock_tokenizer_from_pretrained.called)
            self.assertTrue(isinstance(e, NotImplementedError))
        else:
            self.fail("Should have thrown exception")
