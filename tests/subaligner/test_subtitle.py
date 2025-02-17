import unittest
import os
import shutil
from pathlib import Path
from parameterized import parameterized
from subaligner.subtitle import Subtitle as Undertest
from subaligner.exception import UnsupportedFormatException


class SubtitleTests(unittest.TestCase):
    def setUp(self):
        self.srt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.srt"
        )
        self.ttml_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.xml"
        )
        self.another_ttml_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ttml"
        )
        self.vtt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.vtt"
        )
        self.ass_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ass"
        )
        self.ssa_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ssa"
        )
        self.microdvd_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.sub"
        )
        self.mpl2_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test_mpl2.txt"
        )
        self.tmp_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.tmp"
        )
        self.sami_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.smi"
        )
        self.stl_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.stl"
        )
        self.scc_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.scc"
        )
        self.sbv_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.sbv"
        )
        self.ytt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.ytt"
        )
        self.subtxt_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.txt"
        )
        self.json_file_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/test.json"
        )
        self.resource_tmp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "resource/tmp"
        )
        if os.path.exists(self.resource_tmp):
            shutil.rmtree(self.resource_tmp)
        os.mkdir(self.resource_tmp)

    def tearDown(self):
        shutil.rmtree(self.resource_tmp)

    def test_get_srt_file_path(self):
        subtitle = Undertest.load_subrip(self.srt_file_path)
        self.assertEqual(self.srt_file_path, subtitle.subtitle_file_path)

    def test_get_ttml_file_path_xml(self):
        subtitle = Undertest.load_ttml(self.ttml_file_path)
        self.assertEqual(self.ttml_file_path, subtitle.subtitle_file_path)

    def test_get_ttml_file_path_ttml(self):
        subtitle = Undertest.load_ttml(self.another_ttml_file_path)
        self.assertEqual(self.another_ttml_file_path, subtitle.subtitle_file_path)

    def test_get_vtt_file_path(self):
        subtitle = Undertest.load_webvtt(self.srt_file_path)
        self.assertEqual(self.srt_file_path, subtitle.subtitle_file_path)

    def test_get_ass_file_path(self):
        subtitle = Undertest.load_ass(self.ass_file_path)
        self.assertEqual(self.ass_file_path, subtitle.subtitle_file_path)

    def test_get_ssa_file_path(self):
        subtitle = Undertest.load_ssa(self.ssa_file_path)
        self.assertEqual(self.ssa_file_path, subtitle.subtitle_file_path)

    def test_get_microdvd_file_path(self):
        subtitle = Undertest.load_microdvd(self.microdvd_file_path)
        self.assertEqual(self.microdvd_file_path, subtitle.subtitle_file_path)

    def test_get_mpl2_file_path(self):
        subtitle = Undertest.load_mpl2(self.mpl2_file_path)
        self.assertEqual(self.mpl2_file_path, subtitle.subtitle_file_path)

    def test_get_tmp_file_path(self):
        subtitle = Undertest.load_tmp(self.tmp_file_path)
        self.assertEqual(self.tmp_file_path, subtitle.subtitle_file_path)

    def test_get_sami_file_path(self):
        subtitle = Undertest.load_sami(self.sami_file_path)
        self.assertEqual(self.sami_file_path, subtitle.subtitle_file_path)

    def test_get_stl_file_path(self):
        subtitle = Undertest.load_stl(self.stl_file_path)
        self.assertEqual(self.stl_file_path, subtitle.subtitle_file_path)

    def test_get_scc_file_path(self):
        subtitle = Undertest.load_scc(self.scc_file_path)
        self.assertEqual(self.scc_file_path, subtitle.subtitle_file_path)

    def test_get_sbv_file_path(self):
        subtitle = Undertest.load_sbv(self.sbv_file_path)
        self.assertEqual(self.sbv_file_path, subtitle.subtitle_file_path)

    def test_get_ytt_file_path(self):
        subtitle = Undertest.load_ytt(self.ytt_file_path)
        self.assertEqual(self.ytt_file_path, subtitle.subtitle_file_path)

    def test_get_json_file_path(self):
        subtitle = Undertest.load_json(self.json_file_path)
        self.assertEqual(self.json_file_path, subtitle.subtitle_file_path)

    def test_load_srt_subs(self):
        subtitle = Undertest.load_subrip(self.srt_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_ttml_subs(self):
        subtitle = Undertest.load_ttml(self.ttml_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_vtt_subs(self):
        subtitle = Undertest.load_webvtt(self.vtt_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_ass_subs(self):
        subtitle = Undertest.load_ass(self.ass_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_ssa_subs(self):
        subtitle = Undertest.load_ssa(self.ssa_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_microdvd_subs(self):
        subtitle = Undertest.load_microdvd(self.microdvd_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_mpl2_subs(self):
        subtitle = Undertest.load_mpl2(self.mpl2_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_tmp_subs(self):
        subtitle = Undertest.load_tmp(self.tmp_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_sami_subs(self):
        subtitle = Undertest.load_sami(self.sami_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_stl_subs(self):
        subtitle = Undertest.load_stl(self.stl_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_scc_subs(self):
        subtitle = Undertest.load_scc(self.scc_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_sbv_subs(self):
        subtitle = Undertest.load_sbv(self.sbv_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_sbv_subs(self):
        subtitle = Undertest.load_ytt(self.ytt_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load_json_subs(self):
        subtitle = Undertest.load_json(self.json_file_path)
        self.assertGreater(len(subtitle.subs), 0)

    def test_load(self):
        srt_subtitle = Undertest.load(self.srt_file_path)
        ttml_subtitle = Undertest.load(self.ttml_file_path)
        vtt_subtitle = Undertest.load(self.vtt_file_path)
        ass_subtitle = Undertest.load(self.ass_file_path)
        ssa_subtitle = Undertest.load(self.ssa_file_path)
        microdvd_subtitle = Undertest.load(self.microdvd_file_path)
        mp2_subtitle = Undertest.load(self.mpl2_file_path)
        tmp_subtitle = Undertest.load(self.tmp_file_path)
        sami_subtitle = Undertest.load(self.sami_file_path)
        stl_subtitle = Undertest.load(self.stl_file_path)
        scc_subtitle = Undertest.load(self.scc_file_path)
        sbv_subtitle = Undertest.load(self.sbv_file_path)
        ytt_subtitle = Undertest.load(self.ytt_file_path)
        json_subtitle = Undertest.load(self.json_file_path)

        self.assertEqual(len(srt_subtitle.subs), len(ttml_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(vtt_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(ass_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(ssa_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(microdvd_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(mp2_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(tmp_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(sami_subtitle.subs))
        self.assertEqual(7, len(stl_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(scc_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(sbv_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(ytt_subtitle.subs))
        self.assertEqual(len(srt_subtitle.subs), len(json_subtitle.subs))

    @parameterized.expand([
        ["srt_file_path", "subtitle_test.srt"],
        ["ttml_file_path", "subtitle_test.xml"],
        ["vtt_file_path", "subtitle_test.vtt"],
        ["ass_file_path", "subtitle_test.ass"],
        ["ssa_file_path", "subtitle_test.ssa"],
        ["microdvd_file_path", "subtitle_test.sub"],
        ["mpl2_file_path", "subtitle_test.mpl2.txt"],
        ["tmp_file_path", "subtitle_test.tmp"],
        ["sami_file_path", "subtitle_test.sami"],
        ["scc_file_path", "subtitle_test.scc"],
        ["sbv_file_path", "subtitle_test.sbv"],
        ["ytt_file_path", "subtitle_test.ytt"],
        ["json_file_path", "subtitle_test.json"],
    ])
    def test_shift_subtitle(self, subtitle_file_path, shifted_subtitle_file_name):
        shifted_srt_file_path = os.path.join(self.resource_tmp, shifted_subtitle_file_name)
        Undertest.shift_subtitle(
            getattr(self, subtitle_file_path), 2, shifted_srt_file_path, suffix="_test"
        )
        with open(getattr(self, subtitle_file_path)) as original:
            for i, lo in enumerate(original):
                pass
        with open(shifted_srt_file_path) as shifted:
            for j, ls in enumerate(shifted):
                pass
        original_line_num = i + 1
        shifted_line_num = j + 1
        self.assertEqual(original_line_num, shifted_line_num)

    def test_shift_stl_subtitle(self):
        shifted_stl_file_path = os.path.join(self.resource_tmp, "subtitle_test.srt")
        Undertest.shift_subtitle(
            self.stl_file_path, 2, shifted_stl_file_path, suffix="_test"
        )
        with open(shifted_stl_file_path) as shifted:
            for j, ls in enumerate(shifted):
                pass
        shifted_line_num = j + 1
        self.assertEqual(32, shifted_line_num)

    @parameterized.expand([
        ["srt_file_path", "subtitle_test.srt", "subtitle_test_aligned.srt"],
        ["ttml_file_path", "subtitle_test.xml", "subtitle_test_aligned.xml"],
        ["vtt_file_path", "subtitle_test.vtt", "subtitle_test_aligned.vtt"],
        ["ass_file_path", "subtitle_test.ass", "subtitle_test_aligned.ass"],
        ["ssa_file_path", "subtitle_test.ssa", "subtitle_test_aligned.ssa"],
        ["microdvd_file_path", "subtitle_test.sub", "subtitle_test_aligned.sub"],
        ["mpl2_file_path", "subtitle_test_mpl2.txt", "subtitle_test_mpl2_aligned.txt"],
        ["tmp_file_path", "subtitle_test.tmp", "subtitle_test_aligned.tmp"],
        ["sami_file_path", "subtitle_test.sami", "subtitle_test_aligned.sami"],
        ["scc_file_path", "subtitle_test.scc", "subtitle_test_aligned.scc"],
        ["sbv_file_path", "subtitle_test.sbv", "subtitle_test_aligned.sbv"],
        ["ytt_file_path", "subtitle_test.ytt", "subtitle_test_aligned.ytt"],
        ["json_file_path", "subtitle_test.json", "subtitle_test_aligned.json"],
    ])
    def test_shift_subtitle_without_destination_file_path(self, subtitle_file_path, shifted_subtitle_file_name, aligned_file_name):
        shifted_file_path = os.path.join(self.resource_tmp, shifted_subtitle_file_name)
        another_shifted_file_path = os.path.join(self.resource_tmp, aligned_file_name)
        Undertest.shift_subtitle(
            getattr(self, subtitle_file_path), 2, shifted_file_path, suffix="_test"
        )
        Undertest.shift_subtitle(
            shifted_file_path, 2, None, suffix="_aligned"
        )
        with open(shifted_file_path) as original:
            for i, lo in enumerate(original):
                pass
        with open(another_shifted_file_path) as shifted:
            for j, ls in enumerate(shifted):
                pass
        original_line_num = i + 1
        shifted_line_num = j + 1
        self.assertEqual(original_line_num, shifted_line_num)

    @parameterized.expand([
        ["ttml_file_path", "subtitle_test.xml"],
        ["vtt_file_path", "subtitle_test.vtt"],
        ["ass_file_path", "subtitle_test.ass"],
        ["ssa_file_path", "subtitle_test.ssa"],
        ["microdvd_file_path", "subtitle_test.sub"],
        ["mpl2_file_path", "subtitle_test.mpl2.txt"],
        ["tmp_file_path", "subtitle_test.tmp"],
        ["sami_file_path", "subtitle_test.sami"],
        ["scc_file_path", "subtitle_test.scc"],
        ["sbv_file_path", "subtitle_test.sbv"],
        ["ytt_file_path", "subtitle_test.ytt"],
        ["json_file_path", "subtitle_test.json"],
    ])
    def test_export_subtitle(self, subtitle_file_path, target_subtitle_file_name):
        target_file_path = os.path.join(self.resource_tmp, target_subtitle_file_name)
        Undertest.export_subtitle(
            getattr(self, subtitle_file_path),
            Undertest.load(getattr(self, subtitle_file_path)).subs,
            target_file_path,
        )
        with open(getattr(self, subtitle_file_path)) as original:
            for i, lo in enumerate(original):
                pass
        with open(target_file_path) as target:
            for j, lt in enumerate(target):
                pass
        original_line_num = i + 1
        target_line_num = j + 1
        self.assertEqual(original_line_num, target_line_num)

    def test_export_stl_subtitle_as_srt(self):
        target_file_path = os.path.join(self.resource_tmp, "subtitle_test.srt")
        Undertest.export_subtitle(
            self.stl_file_path,
            Undertest.load(self.stl_file_path).subs,
            target_file_path,
        )
        with open(target_file_path) as target:
            for j, lt in enumerate(target):
                pass
        target_line_num = j + 1
        self.assertEqual(32, target_line_num)

    @parameterized.expand([
        ["subtitle_converted.srt", None, 67],
        ["subtitle_converted.ttml", None, 66],
        ["subtitle_converted.vtt", None, 52],
        ["subtitle_converted.ssa", None, 31],
        ["subtitle_converted.ass", None, 31],
        ["subtitle_converted_mpl2.txt", None, 17],
        ["subtitle_converted.tmp", None, 17],
        ["subtitle_converted.smi", None, 168],
        ["subtitle_converted.stl", None, 68],
        ["subtitle_converted.sub", 25.0, 18],
        ["subtitle_converted.scc", None, 44],
        ["subtitle_converted.sbv", None, 50],
        ["subtitle_converted.ytt", None, 18],
        ["subtitle_converted.json", None, 1],
    ])
    def test_save_subtitle(self, target_subtitle_file_name, frame_rate, expected_lines):
        target_file_path = os.path.join(self.resource_tmp, target_subtitle_file_name)
        Undertest.save_subs_as_target_format(
            Undertest.load(self.srt_file_path).subs,
            self.srt_file_path,
            target_file_path,
            frame_rate
        )
        with open(target_file_path) as target:
            for j, lt in enumerate(target):
                pass
        target_lines = j + 1
        self.assertEqual(expected_lines, target_lines)

    def test_remove_sound_effects_with_affixes(self):
        subtitle = Undertest.load(self.srt_file_path)
        new_subs = Undertest.remove_sound_effects_by_affixes(
            subtitle.subs, se_prefix="(", se_suffix=")"
        )
        self.assertEqual(len(subtitle.subs) - 2, len(new_subs))

    def test_remove_sound_effects_with_out_suffix(self):
        subtitle = Undertest.load(self.srt_file_path)
        new_subs = Undertest.remove_sound_effects_by_affixes(
            subtitle.subs, se_prefix="("
        )
        self.assertEqual(len(subtitle.subs) - 2, len(new_subs))

    def test_remove_sound_effects_with_uppercase(self):
        subtitle = Undertest.load(self.srt_file_path)
        new_subs = Undertest.remove_sound_effects_by_case(
            subtitle.subs, se_uppercase=True
        )
        self.assertEqual(len(subtitle.subs) - 2, len(new_subs))

    @parameterized.expand([
        ["srt_file_path"],
        ["ttml_file_path"],
        ["vtt_file_path"],
        ["ass_file_path"],
        ["ssa_file_path"],
        ["microdvd_file_path"],
        ["mpl2_file_path"],
        ["tmp_file_path"],
        ["sami_file_path"],
        ["scc_file_path"],
        ["sbv_file_path"],
        ["ytt_file_path"],
        ["json_file_path"],
    ])
    def test_extract_text(self, subtitle_file_path):
        text = Undertest.extract_text(getattr(self, subtitle_file_path))
        with open(self.subtxt_file_path) as target:
            expected_text = target.read()
        self.assertEqual(expected_text, text)

    def test_extract_text_from_stl(self):
        text = Undertest.extract_text(self.stl_file_path)
        self.assertEqual(194, len(text))

    def test_subtitle_extentions(self):
        self.assertEqual({
            ".srt", ".xml", ".ttml", ".dfxp", ".vtt",
            ".ssa", ".ass", ".sub", ".txt", ".tmp",
            ".smi", ".sami", ".stl", ".scc", ".sbv",
            ".ytt", ".json"}, Undertest.subtitle_extensions())

    def test_throw_exception_on_missing_subtitle(self):
        try:
            unknown_file_path = os.path.join(self.resource_tmp, "subtitle_test.unknown")
            Path(unknown_file_path).touch()
            Undertest.export_subtitle(unknown_file_path, None, "")
        except Exception as e:
            self.assertTrue(isinstance(e, UnsupportedFormatException))
        else:
            self.fail("Should have thrown exception")

    def test_throw_exception_on_loading_unknown_subtitle(self):
        try:
            unknown_file_path = os.path.join(self.resource_tmp, "subtitle_test.unknown")
            Path(unknown_file_path).touch()
            Undertest.shift_subtitle(unknown_file_path, 2, "", "")
        except Exception as e:
            self.assertTrue(isinstance(e, UnsupportedFormatException))
        else:
            self.fail("Should have thrown exception")

    def test_throw_exception_on_shifting_unknown_subtitle(self):
        try:
            unknown_file_path = os.path.join(self.resource_tmp, "subtitle_test.unknown")
            Path(unknown_file_path).touch()
            Undertest.load(unknown_file_path)
        except Exception as e:
            self.assertTrue(isinstance(e, UnsupportedFormatException))
        else:
            self.fail("Should have thrown exception")

    def test_throw_exception_on_exporting_unknown_subtitle(self):
        try:
            unknown_file_path = os.path.join(self.resource_tmp, "subtitle_test.unknown")
            Path(unknown_file_path).touch()
            Undertest.export_subtitle(
                unknown_file_path,
                Undertest.load(self.ttml_file_path).subs,
                "target",
            )
        except Exception as e:
            self.assertTrue(isinstance(e, UnsupportedFormatException))
        else:
            self.fail("Should have thrown exception")

    def test_throw_exception_on_converting_to_unknown_subtitle(self):
        try:
            unknown_file_path = os.path.join(self.resource_tmp, "subtitle_test.unknown")
            Path(unknown_file_path).touch()
            Undertest.save_subs_as_target_format(
                Undertest.load(self.stl_file_path).subs,
                self.srt_file_path,
                unknown_file_path
            )
        except Exception as e:
            self.assertTrue(isinstance(e, UnsupportedFormatException))
        else:
            self.fail("Should have thrown exception")


if __name__ == "__main__":
    unittest.main()
