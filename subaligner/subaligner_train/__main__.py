#!/usr/bin/env python
"""
usage: subaligner_train [-h] -tod TRAINING_OUTPUT_DIRECTORY [-vd VIDEO_DIRECTORY] [-sd SUBTITLE_DIRECTORY] [-r] [-dde] [-sesm SOUND_EFFECT_START_MARKER] [-seem SOUND_EFFECT_END_MARKER] [-ess EMBEDDED_SUBTITLE_SELECTOR] [-bs BATCH_SIZE] [-do DROPOUT] [-e EPOCHS]
                        [-p PATIENCE] [-fhs FRONT_HIDDEN_SIZE] [-bhs BACK_HIDDEN_SIZE] [-lr LEARNING_RATE] [-nt {lstm,bi_lstm,conv_1d}] [-vs VALIDATION_SPLIT] [-o {adadelta,adagrad,adam,adamax,ftrl,nadam,rmsprop,sgd}] [-utd] [-d] [-q] [-ver]

Train the Subaligner model

Each subtitle file and its companion audiovisual file need to share the same base filename, the part before the extension.

optional arguments:
  -h, --help            show this help message and exit
  -vd VIDEO_DIRECTORY, --video_directory VIDEO_DIRECTORY
                        Path to the video directory
  -sd SUBTITLE_DIRECTORY, --subtitle_directory SUBTITLE_DIRECTORY
                        Path to the subtitle directory
  -r, --resume          Continue with previous training result if present (hyperparameters passed in will be ignored except for --epochs)
  -dde, --display_done_epochs
                        Display the number of completed epochs
  -sesm SOUND_EFFECT_START_MARKER, --sound_effect_start_marker SOUND_EFFECT_START_MARKER
                        Marker indicating the start of the sound effect which will be ignored during training
  -seem SOUND_EFFECT_END_MARKER, --sound_effect_end_marker SOUND_EFFECT_END_MARKER
                        Marker indicating the end of the sound effect which will be ignored during training and used with sound_effect_start_marker
  -ess EMBEDDED_SUBTITLE_SELECTOR, --embedded_subtitle_selector EMBEDDED_SUBTITLE_SELECTOR
                        E.g., "embedded:page_num=888,file_extension=srt" or "embedded:stream_index=0,file_extension=srt" (supported file extensions: ssa, vtt, ass, srt and ttml).
  -utd, --use_training_dump
                        Use training dump instead of files in the video or subtitle directory
  -d, --debug           Print out debugging information
  -q, --quiet           Switch off logging information
  -ver, --version       show program's version number and exit

required arguments:
  -tod TRAINING_OUTPUT_DIRECTORY, --training_output_directory TRAINING_OUTPUT_DIRECTORY
                        Path to the output directory containing training results

optional hyperparameters:
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Number of 32ms samples at each training step
  -do DROPOUT, --dropout DROPOUT
                        Dropout rate between 0 and 1 used at the end of each intermediate hidden layer
  -e EPOCHS, --epochs EPOCHS
                        Total training epochs
  -p PATIENCE, --patience PATIENCE
                        Number of epochs with no improvement after which training will be stopped
  -fhs FRONT_HIDDEN_SIZE, --front_hidden_size FRONT_HIDDEN_SIZE
                        Number of neurons in the front LSTM or Conv1D layer
  -bhs BACK_HIDDEN_SIZE, --back_hidden_size BACK_HIDDEN_SIZE
                        Comma-separated numbers of neurons in the back Dense layers
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate of the optimiser
  -nt {lstm,bi_lstm,conv_1d}, --network_type {lstm,bi_lstm,conv_1d}
                        Network type
  -vs VALIDATION_SPLIT, --validation_split VALIDATION_SPLIT
                        Fraction between 0 and 1 of the training data to be used as validation data
  -o {adadelta,adagrad,adam,adamax,ftrl,nadam,rmsprop,sgd}, --optimizer {adadelta,adagrad,adam,adamax,ftrl,nadam,rmsprop,sgd}
                        TensorFlow optimizer
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
    parser = argparse.ArgumentParser(description="""Train the Subaligner model (%s)\n
Each subtitle file and its companion audiovisual file need to share the same base filename, the part before the extension.""" % __version__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-tod",
        "--training_output_directory",
        type=str,
        default="",
        help="Path to the output directory containing training results",
        required=True,
    )
    parser.add_argument(
        "-vd",
        "--video_directory",
        type=str,
        default="",
        help="Path to the video directory",
    )
    parser.add_argument(
        "-sd",
        "--subtitle_directory",
        type=str,
        default="",
        help="Path to the subtitle directory",
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="Continue with previous training result if present (hyperparameters passed in will be ignored except for --epochs)",
    )
    parser.add_argument(
        "-dde",
        "--display_done_epochs",
        action="store_true",
        help="Display the number of completed epochs",
    )
    parser.add_argument(
        "-sesm",
        "--sound_effect_start_marker",
        type=str,
        default=None,
        help="Marker indicating the start of the sound effect which will be ignored during training",
    )
    parser.add_argument(
        "-seem",
        "--sound_effect_end_marker",
        type=str,
        default=None,
        help="Marker indicating the end of the sound effect which will be ignored during training and used with sound_effect_start_marker",
    )
    parser.add_argument(
        "-ess",
        "--embedded_subtitle_selector",
        type=str,
        default=None,
        help='E.g., "embedded:page_num=888,file_extension=srt" or "embedded:stream_index=0,file_extension=srt" (supported file extensions: ssa, vtt, ass, srt and ttml).'
    )
    hyperparameter_args = parser.add_argument_group("optional hyperparameters")
    hyperparameter_args.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=32,
        help="Number of 32ms samples at each training step",
    )
    hyperparameter_args.add_argument(
        "-do",
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate between 0 and 1 used at the end of each intermediate hidden layer",
    )
    hyperparameter_args.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="Total training epochs",
    )
    hyperparameter_args.add_argument(
        "-p",
        "--patience",
        type=int,
        default=1000000,
        help="Number of epochs with no improvement after which training will be stopped",
    )
    hyperparameter_args.add_argument(
        "-fhs",
        "--front_hidden_size",
        type=int,
        default=64,
        help="Number of neurons in the front LSTM or Conv1D layer",
    )
    hyperparameter_args.add_argument(
        "-bhs",
        "--back_hidden_size",
        type=str,
        default="32,16",
        help="Comma-separated numbers of neurons in the back Dense layers",
    )
    hyperparameter_args.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate of the optimiser",
    )
    hyperparameter_args.add_argument(
        "-nt",
        "--network_type",
        type=str,
        choices=["lstm", "bi_lstm", "conv_1d"],
        default="lstm",
        help="Network type",
    )
    hyperparameter_args.add_argument(
        "-vs",
        "--validation_split",
        type=float,
        default=0.25,
        help="Fraction between 0 and 1 of the training data to be used as validation data",
    )
    hyperparameter_args.add_argument(
        "-o",
        "--optimizer",
        type=str,
        choices=["adadelta", "adagrad", "adam", "adamax", "ftrl", "nadam", "rmsprop", "sgd"],
        default="adam",
        help="TensorFlow optimizer",
    )

    parser.add_argument("-utd", "--use_training_dump", action="store_true",
                        help="Use training dump instead of files in the video or subtitle directory")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Print out debugging information")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Switch off logging information")
    parser.add_argument("-ver", "--version", action="version", version=__version__)
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.training_output_directory == "":
        print("ERROR: --training_output_directory was not passed in")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.sound_effect_end_marker is not None and FLAGS.sound_effect_start_marker is None:
        print("ERROR: --sound_effect_start_marker was not passed in when --sound_effect_end_marker was in use")
        parser.print_usage()
        sys.exit(21)

    verbose = FLAGS.debug

    try:
        from subaligner.logger import Logger
        Logger.VERBOSE = FLAGS.debug
        Logger.QUIET = FLAGS.quiet
        from subaligner.exception import UnsupportedFormatException
        from subaligner.exception import TerminalException
        from subaligner.hyperparameters import Hyperparameters
        from subaligner.embedder import FeatureEmbedder
        from subaligner.trainer import Trainer
        from subaligner.utils import Utils

        output_dir = os.path.abspath(FLAGS.training_output_directory)
        os.makedirs(FLAGS.training_output_directory, exist_ok=True)
        if FLAGS.display_done_epochs:
            output_dir = os.path.abspath(FLAGS.training_output_directory)
            print("Number of epochs done: %s" % str(Trainer.get_done_epochs(os.path.join(output_dir, "training.log"))))
            exit(0)

        model_dir = os.path.abspath(os.path.join(FLAGS.training_output_directory, "models", "training", "model"))
        os.makedirs(model_dir, exist_ok=True)
        weights_dir = os.path.abspath(os.path.join(FLAGS.training_output_directory, "models", "training", "weights"))
        os.makedirs(weights_dir, exist_ok=True)
        config_dir = os.path.abspath(os.path.join(FLAGS.training_output_directory, "models", "training", "config"))
        os.makedirs(config_dir, exist_ok=True)
        video_file_paths = subtitle_file_paths = None
        if not FLAGS.resume:
            if FLAGS.video_directory == "":
                print("ERROR: --video_directory was not passed in")
                parser.print_usage()
                sys.exit(21)
            video_file_paths = [os.path.abspath(os.path.join(path, p)) for path, _, files in
                                os.walk(FLAGS.video_directory) for p in files if not p.startswith(".")]
            video_file_paths = sorted(video_file_paths, key=lambda x: os.path.splitext(os.path.basename(x))[0])

            if FLAGS.embedded_subtitle_selector and FLAGS.embedded_subtitle_selector.lower().startswith("embedded:"):
                if FLAGS.subtitle_directory:
                    os.makedirs(FLAGS.subtitle_directory, exist_ok=True)
                else:
                    subtitle_directory = f"{os.path.dirname(video_file_paths[0])}_subtitles"
                    os.makedirs(subtitle_directory)
                    FLAGS.subtitle_directory = subtitle_directory
                params = FLAGS.embedded_subtitle_selector.lower().split(":")[1].split(",")
                if params and "=" in params[0]:
                    params = {param.split("=")[0]: param.split("=")[1] for param in params}
                    for video_file_path in video_file_paths:
                        basename = os.path.basename(video_file_path)
                        filename, _ = os.path.splitext(basename)
                        if "page_num" in params:
                            file_extension = params.get("file_extension", "txt")
                            subtitle_file_path = os.path.join(FLAGS.subtitle_directory, f"{filename}.{file_extension}")
                            Utils.extract_teletext_as_subtitle(video_file_path, int(params.get("page_num", "888")),
                                                               subtitle_file_path)
                        elif "stream_index" in params:
                            file_extension = params.get("file_extension", "srt")
                            subtitle_file_path = os.path.join(FLAGS.subtitle_directory, f"{filename}.{file_extension}")
                            Utils.extract_matroska_subtitle(video_file_path, int(params.get("stream_index", "0")),
                                                            subtitle_file_path)
                else:
                    print("ERROR: Embedded subtitle selector cannot be empty")
                    parser.print_usage()
                    sys.exit(21)
            else:
                if FLAGS.subtitle_directory == "":
                    print("ERROR: --subtitle_directory was not passed in")
                    parser.print_usage()
                    sys.exit(21)

            subtitle_file_paths = [os.path.abspath(os.path.join(path, p)) for path, _, files in
                                   os.walk(FLAGS.subtitle_directory) for p in files if not p.startswith(".")]
            subtitle_file_paths = sorted(subtitle_file_paths, key=lambda x: os.path.splitext(os.path.basename(x))[0])
        if FLAGS.use_training_dump:
            print("Use data dump from previous training and passed-in video and subtitle directories will be ignored")
            video_file_paths = subtitle_file_paths = None

        hyperparameters = Hyperparameters()
        hyperparameters.batch_size = FLAGS.batch_size
        hyperparameters.dropout = FLAGS.dropout
        hyperparameters.epochs = FLAGS.epochs
        hyperparameters.es_patience = FLAGS.patience
        hyperparameters.front_hidden_size = [FLAGS.front_hidden_size]
        hyperparameters.back_hidden_size = [int(n) for n in FLAGS.back_hidden_size.split(",")]
        hyperparameters.learning_rate = FLAGS.learning_rate
        hyperparameters.network_type = FLAGS.network_type
        hyperparameters.validation_split = FLAGS.validation_split
        hyperparameters.optimizer = FLAGS.optimizer

        trainer = Trainer(FeatureEmbedder())
        trainer.train(video_file_paths,
                      subtitle_file_paths,
                      model_dir,
                      weights_dir,
                      config_dir,
                      output_dir,
                      output_dir,
                      hyperparameters,
                      os.path.join(output_dir, "training.log"),
                      FLAGS.resume,
                      FLAGS.sound_effect_start_marker,
                      FLAGS.sound_effect_end_marker)
    except UnsupportedFormatException as e:
        print(
            "ERROR: {}\n{}".format(str(e), traceback.format_stack() if verbose else "")
        )
        traceback.print_tb(e.__traceback__)
        sys.exit(23)
    except TerminalException as e:
        print(
            "ERROR: {}\n{}".format(str(e), traceback.format_stack() if verbose else "")
        )
        traceback.print_tb(e.__traceback__)
        sys.exit(24)
    except Exception as e:
        print(
            "ERROR: {}\n{}".format(str(e), traceback.format_stack() if verbose else "")
        )
        traceback.print_tb(e.__traceback__)
        sys.exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
