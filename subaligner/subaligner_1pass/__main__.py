#!/usr/bin/env python
"""
usage: subaligner_1pass [-h] [-v VIDEO_PATH] [-s SUBTITLE_PATH] [-l MAX_LOGLOSS] [-tod TRAINING_OUTPUT_DIRECTORY] [-o OUTPUT] [-t TRANSLATE] [-lgs] [-d] [-q] [-ver]

Run single-stage alignment

optional arguments:
  -h, --help            show this help message and exit
  -l MAX_LOGLOSS, --max_logloss MAX_LOGLOSS
                        Max global log loss for alignment
  -tod TRAINING_OUTPUT_DIRECTORY, --training_output_directory TRAINING_OUTPUT_DIRECTORY
                        Path to the output directory containing training results
  -o OUTPUT, --output OUTPUT
                        Path to the output subtitle file
  -t TRANSLATE, --translate TRANSLATE
                        Source and target ISO 639-3 language codes separated by a comma (e.g., eng,zho)
  -lgs, --languages     Print out language codes used for stretch and translation
  -d, --debug           Print out debugging information
  -q, --quiet           Switch off logging information
  -ver, --version       show program's version number and exit

required arguments:
  -v VIDEO_PATH, --video_path VIDEO_PATH
                        File path or URL to the video file
  -s SUBTITLE_PATH, --subtitle_path SUBTITLE_PATH
                        File path or URL to the subtitle file (Extensions of supported subtitles: .stl, .dfxp, .xml, .vtt, .sbv, .ytt, .scc, .ttml, .smi, .sami, .ssa, .tmp, .txt, .sub, .srt, .ass) or selector for the embedded subtitle (e.g., embedded:page_num=888 or embedded:stream_index=0)
"""

import argparse
import sys
import traceback
import os
import tempfile
import pkg_resources


def main():
    if sys.version_info.major != 3:
        print("ERROR: Cannot find Python 3")
        sys.exit(20)
    try:
        import subaligner
    except ModuleNotFoundError:
        print("ERROR: Subaligner is not installed")
        sys.exit(20)

    from subaligner._version import __version__
    parser = argparse.ArgumentParser(description="Run single-stage alignment (v%s)" % __version__, formatter_class=argparse.RawTextHelpFormatter)
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-v",
        "--video_path",
        type=str,
        default="",
        help="File path or URL to the video file",
    )
    from subaligner.subtitle import Subtitle
    required_args.add_argument(
        "-s",
        "--subtitle_path",
        type=str,
        default="",
        help="File path or URL to the subtitle file (Extensions of supported subtitles: {}) or selector for the embedded subtitle (e.g., embedded:page_num=888 or embedded:stream_index=0)".format(", ".join(Subtitle.subtitle_extensions())),
    )
    parser.add_argument(
        "-l",
        "--max_logloss",
        type=float,
        default=float("inf"),
        help="Max global log loss for alignment",
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
    parser.add_argument(
        "-t",
        "--translate",
        type=str,
        help="Source and target ISO 639-3 language codes separated by a comma (e.g., eng,zho)",
    )
    parser.add_argument("-lgs", "--languages", action="store_true",
                        help="Print out language codes used for stretch and translation")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Print out debugging information")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Switch off logging information")
    parser.add_argument("-ver", "--version", action="version", version=__version__)
    FLAGS, unparsed = parser.parse_known_args()

    from subaligner.utils import Utils
    if FLAGS.languages:
        print("\n".join(Utils.get_language_table()))
        sys.exit(0)
    if FLAGS.video_path == "":
        print("ERROR: --video_path was not passed in")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.subtitle_path == "":
        print("ERROR: --subtitle_path was not passed in")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.subtitle_path.lower().startswith("http") and FLAGS.output == "":
        print("ERROR: --output was not passed in for alignment on a remote subtitle file")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.subtitle_path.lower().startswith("teletext:") and FLAGS.output == "":
        print("ERROR: --output was not passed in for alignment on embedded subtitles")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.translate is not None:
        if "transformers" not in {pkg.key for pkg in pkg_resources.working_set}:
            print('ERROR: Alignment has been configured to perform translation. Please install "subaligner[translation]" and run your command again.')
            sys.exit(21)

    local_video_path = FLAGS.video_path
    local_subtitle_path = FLAGS.subtitle_path

    from subaligner.logger import Logger
    Logger.VERBOSE = FLAGS.debug
    Logger.QUIET = FLAGS.quiet
    from subaligner.predictor import Predictor
    from subaligner.exception import UnsupportedFormatException
    from subaligner.exception import TerminalException

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

        if FLAGS.subtitle_path.lower().startswith("embedded:"):
            _, local_subtitle_path = tempfile.mkstemp()
            _, subtitle_file_extension = os.path.splitext(FLAGS.output)
            local_subtitle_path = "{}{}".format(local_subtitle_path, subtitle_file_extension)
            params = FLAGS.subtitle_path.lower().split(":")[1].split(",")
            if params and "=" in params[0]:
                params = {param.split("=")[0]: param.split("=")[1] for param in params}
                if "page_num" in params:
                    Utils.extract_teletext_as_subtitle(local_video_path, int(params["page_num"]), local_subtitle_path)
                elif "stream_index" in params:
                    Utils.extract_matroska_subtitle(local_video_path, int(params["stream_index"]), local_subtitle_path)
            else:
                print("ERROR: Embedded subtitle selector cannot be empty")
                parser.print_usage()
                sys.exit(21)

        predictor = Predictor()
        subs, audio_file_path, voice_probabilities, frame_rate = predictor.predict_single_pass(
            video_file_path=local_video_path,
            subtitle_file_path=local_subtitle_path,
            weights_dir=os.path.join(FLAGS.training_output_directory, "models", "training", "weights")
        )

        aligned_subtitle_path = "_aligned.".join(
            FLAGS.subtitle_path.rsplit(".", 1)).replace(".stl", ".srt") if FLAGS.output == "" else FLAGS.output

        if FLAGS.translate is not None:
            from subaligner.translator import Translator
            source, target = FLAGS.translate.split(",")
            translator = Translator(source, target)
            subs = translator.translate(subs)
            Subtitle.save_subs_as_target_format(subs, local_subtitle_path, aligned_subtitle_path, frame_rate, "utf-8")
        else:
            Subtitle.save_subs_as_target_format(subs, local_subtitle_path, aligned_subtitle_path, frame_rate)

        log_loss = predictor.get_log_loss(voice_probabilities, subs)
        if log_loss is None or log_loss > FLAGS.max_logloss:
            print(
                "ERROR: Alignment failed with a too high loss value: {}".format(log_loss)
            )
            _remove_tmp_files(FLAGS, local_video_path, local_subtitle_path)
            sys.exit(22)

        print("Aligned subtitle saved to: {}".format(aligned_subtitle_path))
    except UnsupportedFormatException as e:
        print(
            "ERROR: {}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
        )
        traceback.print_tb(e.__traceback__)
        _remove_tmp_files(FLAGS, local_video_path, local_subtitle_path)
        sys.exit(23)
    except TerminalException as e:
        print(
            "ERROR: {}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
        )
        traceback.print_tb(e.__traceback__)
        _remove_tmp_files(FLAGS, local_video_path, local_subtitle_path)
        sys.exit(24)
    except Exception as e:
        print(
            "ERROR: {}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
        )
        traceback.print_tb(e.__traceback__)
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
