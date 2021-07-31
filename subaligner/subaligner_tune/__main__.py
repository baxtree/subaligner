#!/usr/bin/env python
"""
usage: subaligner_tune [-h] -vd VIDEO_DIRECTORY -sd SUBTITLE_DIRECTORY -tod TRAINING_OUTPUT_DIRECTORY [-ept EPOCHS_PER_TRAIL] [-t TRAILS] [-nt {lstm,bi_lstm,conv_1d}] [-utd] [-d] [-q] [-ver]

Tune hyperparameters before training

optional arguments:
  -h, --help            show this help message and exit
  -ept EPOCHS_PER_TRAIL, --epochs_per_trail EPOCHS_PER_TRAIL
                        Number of training epochs for each trial
  -t TRAILS, --trails TRAILS
                        Number of tuning trials
  -nt {lstm,bi_lstm,conv_1d}, --network_type {lstm,bi_lstm,conv_1d}
                        Network type
  -utd, --use_training_dump
                        Use training dump instead of files in the video or subtitle directory
  -d, --debug           Print out debugging information
  -q, --quiet           Switch off logging information
  -ver, --version       show program's version number and exit

required arguments:
  -vd VIDEO_DIRECTORY, --video_directory VIDEO_DIRECTORY
                        Path to the video directory
  -sd SUBTITLE_DIRECTORY, --subtitle_directory SUBTITLE_DIRECTORY
                        Path to the subtitle directory
  -tod TRAINING_OUTPUT_DIRECTORY, --training_output_directory TRAINING_OUTPUT_DIRECTORY
                        Path to the output directory containing training results
"""

import os
import argparse
import sys
import traceback


def main():
    if sys.version_info.major != 3:
        print("ERROR: Cannot find Python 3")
        sys.exit(20)
    try:
        import subaligner
        del subaligner
    except ModuleNotFoundError:
        print("ERROR: Subaligner is not installed")
        sys.exit(20)

    from subaligner._version import __version__
    parser = argparse.ArgumentParser(description="Tune hyperparameters before training (v%s)" % __version__, formatter_class=argparse.RawTextHelpFormatter)
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-vd",
        "--video_directory",
        type=str,
        default="",
        help="Path to the video directory",
        required=True,
    )
    required_args.add_argument(
        "-sd",
        "--subtitle_directory",
        type=str,
        default="",
        help="Path to the subtitle directory",
        required=True,
    )
    required_args.add_argument(
        "-tod",
        "--training_output_directory",
        type=str,
        default="",
        help="Path to the output directory containing training results",
        required=True,
    )
    parser.add_argument(
        "-ept",
        "--epochs_per_trail",
        type=int,
        default=5,
        help="Number of training epochs for each trial",
    )
    parser.add_argument(
        "-t",
        "--trails",
        type=int,
        default=5,
        help="Number of tuning trials",
    )
    parser.add_argument(
        "-nt",
        "--network_type",
        type=str,
        choices=["lstm", "bi_lstm", "conv_1d"],
        default="lstm",
        help="Network type",
    )
    parser.add_argument("-utd", "--use_training_dump", action="store_true",
                        help="Use training dump instead of files in the video or subtitle directory")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Print out debugging information")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Switch off logging information")
    parser.add_argument("-ver", "--version", action="version", version=__version__)
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.video_directory == "":
        print("ERROR: --video_directory was not passed in")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.subtitle_directory == "":
        print("ERROR: --subtitle_directory was not passed in")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.training_output_directory == "":
        print("ERROR: --training_output_directory was not passed in")
        parser.print_usage()
        sys.exit(21)

    verbose = FLAGS.debug

    try:
        from subaligner.logger import Logger
        Logger.VERBOSE = FLAGS.debug
        Logger.QUIET = FLAGS.quiet
        from subaligner.exception import UnsupportedFormatException
        from subaligner.exception import TerminalException
        from subaligner.hparam_tuner import HyperParameterTuner
        output_dir = os.path.abspath(FLAGS.training_output_directory)
        os.makedirs(FLAGS.training_output_directory, exist_ok=True)
        video_file_paths = [os.path.abspath(os.path.join(FLAGS.video_directory, p)) for p in os.listdir(FLAGS.video_directory) if not p.startswith(".")]
        subtitle_file_paths = [os.path.abspath(os.path.join(FLAGS.subtitle_directory, p)) for p in os.listdir(FLAGS.subtitle_directory) if not p.startswith(".")]
        exported_hyperparam_path = os.path.join(output_dir, "hyperparameters.json")
        if FLAGS.use_training_dump:
            print("Use data dump from previous training and passed-in video and subtitle directories will be ignored")
            video_file_paths = subtitle_file_paths = None

        hparam_tuner = HyperParameterTuner(video_file_paths,
                                           subtitle_file_paths,
                                           output_dir,
                                           num_of_trials=FLAGS.trails,
                                           tuning_epochs=FLAGS.epochs_per_trail,
                                           network_type=FLAGS.network_type)
        hparam_tuner.tune_hyperparameters()
        hparam_tuner.hyperparameters.to_file(exported_hyperparam_path)
        print(hparam_tuner.hyperparameters.to_json())
    except UnsupportedFormatException as e:
        print(
            "ERROR: {}\n{}".format(str(e), traceback.format_stack() if verbose else "")
        )
        sys.exit(23)
    except TerminalException as e:
        print(
            "ERROR: {}\n{}".format(str(e), traceback.format_stack() if verbose else "")
        )
        sys.exit(24)
    except Exception as e:
        print(
            "ERROR: {}\n{}".format(str(e), traceback.format_stack() if verbose else "")
        )
        sys.exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
