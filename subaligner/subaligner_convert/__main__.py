#!/usr/bin/env python
"""
usage: subaligner_convert [-h] -i INPUT_SUBTITLE_PATH -o OUTPUT_SUBTITLE_PATH [-f FRAME_RATE] [-d] [-q] [-ver]

Convert a subtitle from input format to output format

optional arguments:
  -h, --help            show this help message and exit
  -f FRAME_RATE, --frame_rate FRAME_RATE
                        Frame rate used by conversion to formats such as MicroDVD
  -d, --debug           Print out debugging information
  -q, --quiet           Switch off logging information
  -ver, --version       show program's version number and exit

required arguments:
  -i INPUT_SUBTITLE_PATH, --input_subtitle_path INPUT_SUBTITLE_PATH
                        File path or URL to the input subtitle file
  -o OUTPUT_SUBTITLE_PATH, --output_subtitle_path OUTPUT_SUBTITLE_PATH
                        File path to the output subtitle file
"""

import argparse
import os
import sys
import tempfile
import traceback


def main():
    if sys.version_info.major != 3:
        print("Cannot find Python 3")
        sys.exit(20)
    try:
        import subaligner
        del subaligner
    except ModuleNotFoundError:
        print("Subaligner is not installed")
        sys.exit(20)

    from subaligner._version import __version__
    parser = argparse.ArgumentParser(description="Convert a subtitle from input format to output format (v%s)" % __version__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-i",
        "--input_subtitle_path",
        type=str,
        default="",
        help="File path or URL to the input subtitle file",
        required=True,
    )
    required_args.add_argument(
        "-o",
        "--output_subtitle_path",
        type=str,
        default="",
        help="File path to the output subtitle file",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--frame_rate",
        type=float,
        default=None,
        help="Frame rate used by conversion to formats such as MicroDVD",
    )
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Print out debugging information")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Switch off logging information")
    parser.add_argument("-ver", "--version", action="version", version=__version__)
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.input_subtitle_path == "":
        print("--input_subtitle_path was not passed in")
        sys.exit(21)
    if FLAGS.output_subtitle_path == "":
        print("--output_subtitle_path was not passed in")
        sys.exit(21)
    if FLAGS.output_subtitle_path.endswith(".sub") and FLAGS.frame_rate is None:
        print("--frame_rate was not passed in for conversion to MicroDVD")

    local_subtitle_path = FLAGS.input_subtitle_path

    from subaligner.logger import Logger
    Logger.VERBOSE = FLAGS.debug
    Logger.QUIET = FLAGS.quiet
    from subaligner.subtitle import Subtitle
    from subaligner.exception import UnsupportedFormatException, TerminalException
    from subaligner.utils import Utils

    try:
        if FLAGS.input_subtitle_path.lower().startswith("http"):
            _, local_subtitle_path = tempfile.mkstemp()
            _, subtitle_file_extension = os.path.splitext(FLAGS.input_subtitle_path.lower())
            local_subtitle_path = "{}{}".format(local_subtitle_path, subtitle_file_extension)
            Utils.download_file(FLAGS.input_subtitle_path, local_subtitle_path)

        subtitle = Subtitle.load(local_subtitle_path)
        Subtitle.save_subs_as_target_format(subtitle.subs, local_subtitle_path, FLAGS.output_subtitle_path, FLAGS.frame_rate)
        print("Subtitle converted and saved to: {}".format(FLAGS.output_subtitle_path))
    except UnsupportedFormatException as e:
        print(
            "{}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
        )
        _remove_tmp_files(FLAGS, local_subtitle_path)
        sys.exit(23)
    except TerminalException as e:
        print(
            "{}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
        )
        _remove_tmp_files(FLAGS, local_subtitle_path)
        sys.exit(24)
    except Exception as e:
        print(
            "{}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
        )
        _remove_tmp_files(FLAGS, local_subtitle_path)
        sys.exit(1)
    else:
        _remove_tmp_files(FLAGS, local_subtitle_path)
        sys.exit(0)


def _remove_tmp_files(flags, local_subtitle_path):
    if flags.input_subtitle_path.lower().startswith("http") and os.path.exists(local_subtitle_path):
        os.remove(local_subtitle_path)


if __name__ == "__main__":
    main()
