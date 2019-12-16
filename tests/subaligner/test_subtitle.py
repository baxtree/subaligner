import unittest
import os
from subaligner.subtitle import Subtitle as Undertest
from subaligner.exception import UnsupportedFormatException


class SubtitleTests(unittest.TestCase):
    def setUp(self):
        self.__srt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/subtitle.srt"
        )
        self.__ttml_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/subtitle.xml"
        )
        self.__another_ttml_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/subtitle.ttml"
        )

    def test_get_srt_file_path(self):
        subtitle = Undertest.load_subrip(self.__srt_file_path)
        self.assertEqual(self.__srt_file_path, subtitle.subtitle_file_path)

    def test_get_ttml_file_path_xml(self):
        subtitle = Undertest.load_ttml(self.__ttml_file_path)
        self.assertEqual(self.__ttml_file_path, subtitle.subtitle_file_path)

    def test_get_ttml_file_path_ttml(self):
        subtitle = Undertest.load_ttml(self.__another_ttml_file_path)
        self.assertEqual(self.__another_ttml_file_path, subtitle.subtitle_file_path)

    def test_load_srt_subs(self):
        subtitle = Undertest.load_subrip(self.__srt_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_ttml_subs(self):
        subtitle = Undertest.load_ttml(self.__ttml_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load(self):
        srt_subtitle = Undertest.load(self.__srt_file_path)
        ttml_subtitle = Undertest.load(self.__ttml_file_path)
        self.assertEqual(len(srt_subtitle.subs), len(ttml_subtitle.subs))

    def test_shift_ttml_subtitle(self):
        shifted_ttml_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/subtitle_test.xml"
        )
        Undertest.shift_subtitle(
            self.__ttml_file_path, 2, self.__ttml_file_path, suffix="_test"
        )
        with open(self.__ttml_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(shifted_ttml_file_path) as shifted:
            for j, ls in enumerate(shifted):
                pass
        original_line_num = i + 1
        shifted_line_num = j + 1
        self.assertEqual(original_line_num, shifted_line_num)
        os.remove(shifted_ttml_file_path)

    def test_export_subtitle(self):
        target_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/subtitle_test.xml"
        )
        Undertest.export_subtitle(
            self.__ttml_file_path,
            Undertest.load(self.__ttml_file_path).subs,
            target_file_path,
        )
        with open(self.__ttml_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(target_file_path) as target:
            for j, lt in enumerate(target):
                pass
        original_line_num = i + 1
        target_line_num = j + 1
        self.assertEqual(original_line_num, target_line_num)
        os.remove(target_file_path)

    def test_remove_sound_effects_with_affixes(self):
        subtitle = Undertest.load(self.__srt_file_path)
        new_subs = Undertest.remove_sound_effects_by_affixes(
            subtitle.subs, se_prefix="(", se_suffix=")"
        )
        self.assertEqual(len(subtitle.subs) - 2, len(new_subs))

    def test_remove_sound_effects_with_out_suffix(self):
        subtitle = Undertest.load(self.__srt_file_path)
        new_subs = Undertest.remove_sound_effects_by_affixes(
            subtitle.subs, se_prefix="("
        )
        self.assertEqual(len(subtitle.subs) - 2, len(new_subs))

    def test_remove_sound_effects_with_uppercase(self):
        subtitle = Undertest.load(self.__srt_file_path)
        new_subs = Undertest.remove_sound_effects_by_case(
            subtitle.subs, se_uppercase=True
        )
        self.assertEqual(len(subtitle.subs) - 2, len(new_subs))

    def test_throw_exception_on_missing_subtitle(self):
        try:
            unknown_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "resource/subtitle_test.unknown",
            )
            Undertest.export_subtitle(unknown_file_path, None, "")
        except Exception as e:
            self.assertTrue(isinstance(e, UnsupportedFormatException))
        else:
            self.fail("Should have thrown exception")

    def test_throw_exception_on_unknown_subtitle(self):
        try:
            unknown_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "resource/subtitle_test.unknown",
            )
            Undertest.shift_subtitle(unknown_file_path, 2, "", "")
        except Exception as e:
            self.assertTrue(isinstance(e, UnsupportedFormatException))
        else:
            self.fail("Should have thrown exception")


if __name__ == "__main__":
    unittest.main()
