import unittest
import os
import shutil
from subaligner.subtitle import Subtitle as Undertest
from subaligner.exception import UnsupportedFormatException


class SubtitleTests(unittest.TestCase):
    def setUp(self):
        self.__srt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.srt"
        )
        self.__ttml_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.xml"
        )
        self.__another_ttml_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ttml"
        )
        self.__vtt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.vtt"
        )
        self.__ass_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ass"
        )
        self.__ssa_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ssa"
        )
        self.__microdvd_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.sub"
        )
        self.__mpl2_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.mpl2.txt"
        )
        self.__tmp_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.tmp"
        )
        self.__subtxt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.txt"
        )
        self.__resource_tmp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/tmp"
        )
        if os.path.exists(self.__resource_tmp):
            shutil.rmtree(self.__resource_tmp)
        os.mkdir(self.__resource_tmp)

    def tearDown(self):
        shutil.rmtree(self.__resource_tmp)

    def test_get_srt_file_path(self):
        subtitle = Undertest.load_subrip(self.__srt_file_path)
        self.assertEqual(self.__srt_file_path, subtitle.subtitle_file_path)

    def test_get_ttml_file_path_xml(self):
        subtitle = Undertest.load_ttml(self.__ttml_file_path)
        self.assertEqual(self.__ttml_file_path, subtitle.subtitle_file_path)

    def test_get_ttml_file_path_ttml(self):
        subtitle = Undertest.load_ttml(self.__another_ttml_file_path)
        self.assertEqual(self.__another_ttml_file_path, subtitle.subtitle_file_path)

    def test_get_vtt_file_path(self):
        subtitle = Undertest.load_webvtt(self.__srt_file_path)
        self.assertEqual(self.__srt_file_path, subtitle.subtitle_file_path)

    def test_get_ass_file_path(self):
        subtitle = Undertest.load_ass(self.__ass_file_path)
        self.assertEqual(self.__ass_file_path, subtitle.subtitle_file_path)

    def test_get_ssa_file_path(self):
        subtitle = Undertest.load_ssa(self.__ssa_file_path)
        self.assertEqual(self.__ssa_file_path, subtitle.subtitle_file_path)

    def test_get_microdvd_file_path(self):
        subtitle = Undertest.load_microdvd(self.__microdvd_file_path)
        self.assertEqual(self.__microdvd_file_path, subtitle.subtitle_file_path)

    def test_get_mpl2_file_path(self):
        subtitle = Undertest.load_mpl2(self.__mpl2_file_path)
        self.assertEqual(self.__mpl2_file_path, subtitle.subtitle_file_path)

    def test_get_tmp_file_path(self):
        subtitle = Undertest.load_tmp(self.__tmp_file_path)
        self.assertEqual(self.__tmp_file_path, subtitle.subtitle_file_path)

    def test_load_srt_subs(self):
        subtitle = Undertest.load_subrip(self.__srt_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_ttml_subs(self):
        subtitle = Undertest.load_ttml(self.__ttml_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_vtt_subs(self):
        subtitle = Undertest.load_webvtt(self.__vtt_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_ass_subs(self):
        subtitle = Undertest.load_ass(self.__ass_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_ssa_subs(self):
        subtitle = Undertest.load_ssa(self.__ssa_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_microdvd_subs(self):
        subtitle = Undertest.load_ssa(self.__microdvd_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_mpl2_subs(self):
        subtitle = Undertest.load_ssa(self.__mpl2_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_tmp_subs(self):
        subtitle = Undertest.load_tmp(self.__tmp_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load(self):
        srt_subtitle = Undertest.load(self.__srt_file_path)
        ttml_subtitle = Undertest.load(self.__ttml_file_path)
        vtt_subtitle = Undertest.load(self.__vtt_file_path)
        ass_subtitle = Undertest.load(self.__ass_file_path)
        ssa_subtitle = Undertest.load(self.__ssa_file_path)
        microdvd_subtitle = Undertest.load(self.__microdvd_file_path)
        mp2_subtitle = Undertest.load(self.__mpl2_file_path)
        tmp_subtitle = Undertest.load(self.__tmp_file_path)

        self.assertEqual(len(srt_subtitle.subs), len(ttml_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(vtt_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(ass_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(ssa_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(microdvd_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(mp2_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(tmp_subtitle.subs))

    def test_shift_srt_subtitle(self):
        shifted_srt_file_path = os.path.join(self.__resource_tmp, "subtitle_test.srt")
        Undertest.shift_subtitle(
            self.__srt_file_path, 2, shifted_srt_file_path, suffix="_test"
        )
        with open(self.__srt_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(shifted_srt_file_path) as shifted:
            for j, ls in enumerate(shifted):
                pass
        original_line_num = i + 1
        shifted_line_num = j + 1
        self.assertEqual(original_line_num, shifted_line_num)

    def test_shift_ttml_subtitle(self):
        shifted_ttml_file_path = os.path.join(self.__resource_tmp, "subtitle_test.xml")
        Undertest.shift_subtitle(
            self.__ttml_file_path, 2, shifted_ttml_file_path, suffix="_test"
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

    def test_shift_vtt_subtitle(self):
        shifted_vtt_file_path = os.path.join(self.__resource_tmp, "subtitle_test.vtt")
        Undertest.shift_subtitle(
            self.__vtt_file_path, 2, shifted_vtt_file_path, suffix="_test"
        )
        with open(self.__vtt_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(shifted_vtt_file_path) as shifted:
            for j, ls in enumerate(shifted):
                pass
        original_line_num = i + 1
        shifted_line_num = j + 1
        self.assertEqual(original_line_num, shifted_line_num)

    def test_shift_ass_subtitle(self):
        shifted_ass_file_path = os.path.join(self.__resource_tmp, "subtitle_test.ass")
        Undertest.shift_subtitle(
            self.__ass_file_path, 2, shifted_ass_file_path, suffix="_test"
        )
        with open(self.__ass_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(shifted_ass_file_path) as shifted:
            for j, ls in enumerate(shifted):
                pass
        original_line_num = i + 1
        shifted_line_num = j + 1
        self.assertEqual(original_line_num, shifted_line_num)

    def test_shift_ssa_subtitle(self):
        shifted_ssa_file_path = os.path.join(self.__resource_tmp, "subtitle_test.ssa")
        Undertest.shift_subtitle(
            self.__ssa_file_path, 2, shifted_ssa_file_path, suffix="_test"
        )
        with open(self.__ssa_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(shifted_ssa_file_path) as shifted:
            for j, ls in enumerate(shifted):
                pass
        original_line_num = i + 1
        shifted_line_num = j + 1
        self.assertEqual(original_line_num, shifted_line_num)

    def test_shift_mpl2_subtitle(self):
        shifted_mpl2_file_path = os.path.join(self.__resource_tmp, "subtitle_test.mpl2.txt")
        Undertest.shift_subtitle(
            self.__mpl2_file_path, 2, shifted_mpl2_file_path, suffix="_test"
        )
        with open(self.__mpl2_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(shifted_mpl2_file_path) as shifted:
            for j, ls in enumerate(shifted):
                pass
        original_line_num = i + 1
        shifted_line_num = j + 1
        self.assertEqual(original_line_num, shifted_line_num)

    def test_shift_tmp_subtitle(self):
        shifted_tmp_file_path = os.path.join(self.__resource_tmp, "subtitle_test.tmp.txt")
        Undertest.shift_subtitle(
            self.__tmp_file_path, 2, shifted_tmp_file_path, suffix="_test"
        )
        with open(self.__tmp_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(shifted_tmp_file_path) as shifted:
            for j, ls in enumerate(shifted):
                pass
        original_line_num = i + 1
        shifted_line_num = j + 1
        self.assertEqual(original_line_num, shifted_line_num)

    def test_export_ttml_subtitle(self):
        target_file_path = os.path.join(self.__resource_tmp, "subtitle_test.xml")
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

    def test_export_vtt_subtitle(self):
        target_file_path = os.path.join(self.__resource_tmp, "subtitle_test.vtt")
        Undertest.export_subtitle(
            self.__vtt_file_path,
            Undertest.load(self.__vtt_file_path).subs,
            target_file_path,
        )
        with open(self.__vtt_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(target_file_path) as target:
            for j, lt in enumerate(target):
                pass
        original_line_num = i + 1
        target_line_num = j + 1
        self.assertEqual(original_line_num, target_line_num)

    def test_export_ass_subtitle(self):
        target_file_path = os.path.join(self.__resource_tmp, "subtitle_test.ass")
        Undertest.export_subtitle(
            self.__ass_file_path,
            Undertest.load(self.__ass_file_path).subs,
            target_file_path,
        )
        with open(self.__ass_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(target_file_path) as target:
            for j, lt in enumerate(target):
                pass
        original_line_num = i + 1
        target_line_num = j + 1
        self.assertEqual(original_line_num, target_line_num)

    def test_export_ssa_subtitle(self):
        target_file_path = os.path.join(self.__resource_tmp, "subtitle_test.ssa")
        Undertest.export_subtitle(
            self.__ssa_file_path,
            Undertest.load(self.__ssa_file_path).subs,
            target_file_path,
        )
        with open(self.__ssa_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(target_file_path) as target:
            for j, lt in enumerate(target):
                pass
        original_line_num = i + 1
        target_line_num = j + 1
        self.assertEqual(original_line_num, target_line_num)

    def test_export_microdvd_subtitle(self):
        target_file_path = os.path.join(self.__resource_tmp, "subtitle_test.sub")
        Undertest.export_subtitle(
            self.__microdvd_file_path,
            Undertest.load(self.__microdvd_file_path).subs,
            target_file_path,
        )
        with open(self.__microdvd_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(target_file_path) as target:
            for j, lt in enumerate(target):
                pass
        original_line_num = i + 1
        target_line_num = j + 1
        self.assertEqual(original_line_num, target_line_num)

    def test_export_mpl2_subtitle(self):
        target_file_path = os.path.join(self.__resource_tmp, "subtitle_test.mpl2.txt")
        Undertest.export_subtitle(
            self.__mpl2_file_path,
            Undertest.load(self.__mpl2_file_path).subs,
            target_file_path,
        )
        with open(self.__mpl2_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(target_file_path) as target:
            for j, lt in enumerate(target):
                pass
        original_line_num = i + 1
        target_line_num = j + 1
        self.assertEqual(original_line_num, target_line_num)

    def test_export_tmp_subtitle(self):
        target_file_path = os.path.join(self.__resource_tmp, "subtitle_test.tmp.txt")
        Undertest.export_subtitle(
            self.__tmp_file_path,
            Undertest.load(self.__tmp_file_path).subs,
            target_file_path,
        )
        with open(self.__tmp_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(target_file_path) as target:
            for j, lt in enumerate(target):
                pass
        original_line_num = i + 1
        target_line_num = j + 1
        self.assertEqual(original_line_num, target_line_num)

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

    def test_extract_text_from_srt(self):
        text = Undertest.extract_text(self.__srt_file_path)
        with open(self.__subtxt_file_path) as target:
            expected_text = target.read()
        self.assertEqual(expected_text, text)

    def test_extract_text_from_ttml(self):
        text = Undertest.extract_text(self.__ttml_file_path)
        with open(self.__subtxt_file_path) as target:
            expected_text = target.read()
        self.assertEqual(expected_text, text)

    def test_extract_text_from_vtt(self):
        text = Undertest.extract_text(self.__vtt_file_path)
        with open(self.__subtxt_file_path) as target:
            expected_text = target.read()
        self.assertEqual(expected_text, text)

    def test_extract_text_from_ass(self):
        text = Undertest.extract_text(self.__ass_file_path)
        with open(self.__subtxt_file_path) as target:
            expected_text = target.read()
        self.assertEqual(expected_text, text)

    def test_extract_text_from_ssa(self):
        text = Undertest.extract_text(self.__ssa_file_path)
        with open(self.__subtxt_file_path) as target:
            expected_text = target.read()
        self.assertEqual(expected_text, text)

    def test_extract_text_from_microdvd(self):
        text = Undertest.extract_text(self.__microdvd_file_path)
        with open(self.__subtxt_file_path) as target:
            expected_text = target.read()
        self.assertEqual(expected_text, text)

    def test_extract_text_from_mpl2(self):
        text = Undertest.extract_text(self.__mpl2_file_path)
        with open(self.__subtxt_file_path) as target:
            expected_text = target.read()
        self.assertEqual(expected_text, text)

    def test_extract_text_from_tmp(self):
        text = Undertest.extract_text(self.__tmp_file_path)
        with open(self.__subtxt_file_path) as target:
            expected_text = target.read()
        self.assertEqual(expected_text, text)

    def test_throw_exception_on_missing_subtitle(self):
        try:
            unknown_file_path = os.path.join(self.__resource_tmp, "subtitle_test.unknown")
            Undertest.export_subtitle(unknown_file_path, None, "")
        except Exception as e:
            self.assertTrue(isinstance(e, UnsupportedFormatException))
        else:
            self.fail("Should have thrown exception")

    def test_throw_exception_on_loading_unknown_subtitle(self):
        try:
            unknown_file_path = os.path.join(self.__resource_tmp, "subtitle_test.unknown")
            Undertest.shift_subtitle(unknown_file_path, 2, "", "")
        except Exception as e:
            self.assertTrue(isinstance(e, UnsupportedFormatException))
        else:
            self.fail("Should have thrown exception")

    def test_throw_exception_on_shifting_unknown_subtitle(self):
        try:
            unknown_file_path = os.path.join(self.__resource_tmp, "subtitle_test.unknown")
            Undertest.load(unknown_file_path)
        except Exception as e:
            self.assertTrue(isinstance(e, UnsupportedFormatException))
        else:
            self.fail("Should have thrown exception")

    def test_throw_exception_on_exporting_unknown_subtitle(self):
        try:
            unknown_file_path = os.path.join(self.__resource_tmp, "subtitle_test.unknown")
            Undertest.export_subtitle(
                unknown_file_path,
                Undertest.load(self.__ttml_file_path).subs,
                "target",
            )
        except Exception as e:
            self.assertTrue(isinstance(e, UnsupportedFormatException))
        else:
            self.fail("Should have thrown exception")


if __name__ == "__main__":
    unittest.main()
