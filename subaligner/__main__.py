#!/usr/bin/env python
"""
usage: subaligner [-h] -m {single,dual} -v VIDEO_PATH -s SUBTITLE_PATH [-l MAX_LOGLOSS] [-so] [-fos] [-tod TRAINING_OUTPUT_DIRECTORY] [-o OUTPUT] [-d] [-q]

Subaligner command line interface

optional arguments:
  -h, --help                Show this help message and exit
  -l MAX_LOGLOSS, --max_logloss MAX_LOGLOSS
                            Max global log loss for alignment
  -so, --stretch_off        Switch off stretch on non-English speech and subtitles)
  -sil, --stretch_in_lang   Switch on stretch on subtitles with the language
  -fos, --exit_segfail      Exit on any segment alignment failures
  -tod TRAINING_OUTPUT_DIRECTORY, --training_output_directory TRAINING_OUTPUT_DIRECTORY
                            Path to the output directory containing training results
  -o OUTPUT, --output OUTPUT
                            Path to the output subtitle file
  -d, --debug               Print out debugging information
  -q, --quiet               Switch off logging information

required arguments:
  -m {single,dual}, --mode {single,dual}
                            Alignment mode: either single or dual
  -v VIDEO_PATH, --video_path VIDEO_PATH
                            Path to the video file
  -s SUBTITLE_PATH, --subtitle_path SUBTITLE_PATH
                            Path to the subtitle file
"""

import argparse
import sys
import traceback
import os
import tempfile


def main():
    if sys.version_info.major != 3:
        print("Cannot find Python 3")
        sys.exit(20)
    try:
        import subaligner
    except ModuleNotFoundError:
        print("Subaligner is not installed")
        sys.exit(20)

    parser = argparse.ArgumentParser(description="Subaligner command line interface", formatter_class=argparse.RawTextHelpFormatter)
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-m",
        "--mode",
        type=str,
        default="",
        choices=["single", "dual"],
        help="Alignment mode: either single or dual",
        required=True,
    )
    required_args.add_argument(
        "-v",
        "--video_path",
        type=str,
        default="",
        help="Path to the video file",
        required=True,
    )
    required_args.add_argument(
        "-s",
        "--subtitle_path",
        type=str,
        default="",
        help="Path to the subtitle file",
        required=True,
    )
    parser.add_argument(
        "-l",
        "--max_logloss",
        type=float,
        default=float("inf"),
        help="Max global log loss for alignment",
    )
    parser.add_argument(
        "-so",
        "--stretch_off",
        action="store_true",
        help="Switch off stretch on non-English speech and subtitles)",
    )
    from aeneas.language import Language
    parser.add_argument(
        "-sil",
        "--stretch_in_language",
        type=str,
        choices=Language.ALLOWED_VALUES,
        default=Language.ENG,
        help="Stretch the subtitle with the supported ISO 639-2 language code [https://en.wikipedia.org/wiki/List_of_ISO_639-2_codes].\nNB: This will be ignored if either -so or --stretch_off is present",
    )
    parser.add_argument(
        "-fos",
        "--exit_segfail",
        action="store_true",
        help="Exit on any segment alignment failures",
    )
    parser.add_argument(
        "-tod",
        "--training_output_directory",
        type=str,
        default=os.path.abspath(os.path.dirname(subaligner.__file__)),
        help="Path to the output directory containing training results",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Path to the output subtitle file",
    )
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Print out debugging information")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Switch off logging information")
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.mode == "":
        print("--mode was not passed in")
        sys.exit(21)
    if FLAGS.video_path == "":
        print("--video_path was not passed in")
        sys.exit(21)
    if FLAGS.subtitle_path == "":
        print("--subtitle_path was not passed in")
        sys.exit(21)
    if FLAGS.subtitle_path.lower().startswith("http") and FLAGS.output == "":
        print("--output was not passed in for alignment on a remote subtitle file")
        sys.exit(21)

    local_video_path = FLAGS.video_path
    local_subtitle_path = FLAGS.subtitle_path
    exit_segfail = FLAGS.exit_segfail
    stretch = not FLAGS.stretch_off
    stretch_in_lang = FLAGS.stretch_in_language

    from subaligner.logger import Logger
    Logger.VERBOSE = FLAGS.debug
    Logger.QUIET = FLAGS.quiet
    from subaligner.predictor import Predictor
    from subaligner.subtitle import Subtitle
    from subaligner.exception import UnsupportedFormatException
    from subaligner.exception import TerminalException
    from subaligner.utils import Utils

    try:
        if FLAGS.video_path.lower().startswith("http"):
            _, local_video_path = tempfile.mkstemp()
            _, video_file_extension = os.path.splitext(FLAGS.video_path.lower())
            local_video_path = "{}{}".format(local_video_path, video_file_extension)
            Utils.download_file(FLAGS.video_path, local_video_path)

        if FLAGS.subtitle_path.lower().startswith("http"):
            _, local_subtitle_path = tempfile.mkstemp()
            _, subtitle_file_extension = os.path.splitext(FLAGS.subtitle_path.lower())
            local_subtitle_path = "{}{}".format(local_subtitle_path, subtitle_file_extension)
            Utils.download_file(FLAGS.subtitle_path, local_subtitle_path)

        predictor = Predictor()
        if FLAGS.mode == "single":
            aligned_subs, audio_file_path, voice_probabilities, frame_rate = predictor.predict_single_pass(
                video_file_path=local_video_path,
                subtitle_file_path=local_subtitle_path,
                weights_dir=os.path.join(FLAGS.training_output_directory, "models/training/weights")
            )
        else:
            aligned_subs, subs, voice_probabilities, frame_rate = predictor.predict_dual_pass(
                video_file_path=local_video_path,
                subtitle_file_path=local_subtitle_path,
                weights_dir=os.path.join(FLAGS.training_output_directory, "models/training/weights"),
                stretch=stretch,
                stretch_in_lang=stretch_in_lang,
                exit_segfail=exit_segfail,
            )
        aligned_subtitle_path = "_aligned.".join(
            FLAGS.subtitle_path.rsplit(".", 1)) if FLAGS.output == "" else FLAGS.output
        Subtitle.export_subtitle(FLAGS.subtitle_path, aligned_subs, aligned_subtitle_path, frame_rate)
        print("Aligned subtitle saved to: {}".format(aligned_subtitle_path))

        log_loss = predictor.get_log_loss(voice_probabilities, aligned_subs)
        if log_loss is None or log_loss > FLAGS.max_logloss:
            print(
                "Alignment failed with a too high loss value: {}".format(log_loss)
            )
            _remove_tmp_files(FLAGS, local_video_path, local_subtitle_path)
            sys.exit(22)
    except UnsupportedFormatException as e:
        print(
            "{}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
        )
        _remove_tmp_files(FLAGS, local_video_path, local_subtitle_path)
        sys.exit(23)
    except TerminalException as e:
        print(
            "{}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
        )
        _remove_tmp_files(FLAGS, local_video_path, local_subtitle_path)
        sys.exit(24)
    except Exception as e:
        print(
            "{}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
        )
        _remove_tmp_files(FLAGS, local_video_path, local_subtitle_path)
        sys.exit(1)
    else:
        _remove_tmp_files(FLAGS, local_video_path, local_subtitle_path)
        sys.exit(0)


def _remove_tmp_files(flags, local_video_path, local_subtitle_path):
    if flags.video_path.lower().startswith("http") and os.path.exists(local_video_path):
        os.remove(local_video_path)
    if flags.subtitle_path.lower().startswith("http") and os.path.exists(local_subtitle_path):
        os.remove(local_subtitle_path)


if __name__ == "__main__":
    main()
