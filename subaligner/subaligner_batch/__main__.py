#!/usr/bin/env python
"""
usage: subaligner_batch [-h] [-m {single,dual}] [-vd VIDEO_DIRECTORY] [-sd SUBTITLE_DIRECTORY] [-l MAX_LOGLOSS] [-so]
                        [-sil {afr,amh,ara,arg,asm,aze,ben,bos,bul,cat,ces,cmn,cym,dan,deu,ell,eng,epo,est,eus,fas,fin,fra,gla,gle,glg,grc,grn,guj,heb,hin,hrv,hun,hye,ina,ind,isl,ita,jbo,jpn,kal,kan,kat,kir,kor,kur,lat,lav,lfn,lit,mal,mar,mkd,mlt,msa,mya,nah,nep,nld,nor,ori,orm,pan,pap,pol,por,ron,rus,sin,slk,slv,spa,sqi,srp,swa,swe,tam,tat,tel,tha,tsn,tur,ukr,urd,vie,yue,zho}]
                        [-fos] [-tod TRAINING_OUTPUT_DIRECTORY] [-od OUTPUT_DIRECTORY] [-t TRANSLATE] [-lgs] [-d] [-q] [-ver]

Batch align multiple subtitle files and audiovisual files (v0.1.4)

Subtitle files and their companion audiovisual files need to be stored in two separate directories.
Each file pair needs to share the same base filename, the part before the extension.

optional arguments:
  -h, --help            show this help message and exit
  -vd VIDEO_DIRECTORY, --video_directory VIDEO_DIRECTORY
                        Path to the video directory
  -sd SUBTITLE_DIRECTORY, --subtitle_directory SUBTITLE_DIRECTORY
                        Path to the subtitle directory
  -l MAX_LOGLOSS, --max_logloss MAX_LOGLOSS
                        Max global log loss for alignment
  -so, --stretch_off    Switch off stretch on non-English speech and subtitles)
  -sil {afr,amh,ara,arg,asm,aze,ben,bos,bul,cat,ces,cmn,cym,dan,deu,ell,eng,epo,est,eus,fas,fin,fra,gla,gle,glg,grc,grn,guj,heb,hin,hrv,hun,hye,ina,ind,isl,ita,jbo,jpn,kal,kan,kat,kir,kor,kur,lat,lav,lfn,lit,mal,mar,mkd,mlt,msa,mya,nah,nep,nld,nor,ori,orm,pan,pap,pol,por,ron,rus,sin,slk,slv,spa,sqi,srp,swa,swe,tam,tat,tel,tha,tsn,tur,ukr,urd,vie,yue,zho}, --stretch_in_language {afr,amh,ara,arg,asm,aze,ben,bos,bul,cat,ces,cmn,cym,dan,deu,ell,eng,epo,est,eus,fas,fin,fra,gla,gle,glg,grc,grn,guj,heb,hin,hrv,hun,hye,ina,ind,isl,ita,jbo,jpn,kal,kan,kat,kir,kor,kur,lat,lav,lfn,lit,mal,mar,mkd,mlt,msa,mya,nah,nep,nld,nor,ori,orm,pan,pap,pol,por,ron,rus,sin,slk,slv,spa,sqi,srp,swa,swe,tam,tat,tel,tha,tsn,tur,ukr,urd,vie,yue,zho}
                        Stretch the subtitle with the supported ISO 639-3 language code [https://en.wikipedia.org/wiki/List_of_ISO_639-3_codes].
                        NB: This will be ignored if either -so or --stretch_off is present
  -fos, --exit_segfail  Exit on any segment alignment failures
  -tod TRAINING_OUTPUT_DIRECTORY, --training_output_directory TRAINING_OUTPUT_DIRECTORY
                        Path to the output directory containing training results
  -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Path to the output subtitle directory
  -t TRANSLATE, --translate TRANSLATE
                        Source and target ISO 639-3 language codes separated by a comma (e.g., eng,zho)
  -lgs, --languages     Print out language codes used for stretch and translation
  -d, --debug           Print out debugging information
  -q, --quiet           Switch off logging information
  -ver, --version       show program's version number and exit

required arguments:
  -m {single,dual}, --mode {single,dual}
                        Alignment mode: either single or dual
"""

import argparse
import sys
import traceback
import os


def main():
    if sys.version_info.major != 3:
        print("Cannot find Python 3")
        sys.exit(20)
    try:
        import subaligner
    except ModuleNotFoundError:
        print("Subaligner is not installed")
        sys.exit(20)

    from subaligner._version import __version__
    parser = argparse.ArgumentParser(description="""Batch align multiple subtitle files and audiovisual files (v%s)\n
Subtitle files and their companion audiovisual files need to be stored in two separate directories.
Each file pair needs to share the same base filename, the part before the extension.""" % __version__, formatter_class=argparse.RawTextHelpFormatter)
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-m",
        "--mode",
        type=str,
        default="",
        choices=["single", "dual"],
        help="Alignment mode: either single or dual",
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
    from subaligner.utils import Utils
    parser.add_argument(
        "-sil",
        "--stretch_in_language",
        type=str,
        choices=Utils.get_stretch_language_codes(),
        default="eng",
        help="Stretch the subtitle with the supported ISO 639-3 language code [https://en.wikipedia.org/wiki/List_of_ISO_639-3_codes].\nNB: This will be ignored if either -so or --stretch_off is present",
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
        "-od",
        "--output_directory",
        type=str,
        default="",
        help="Path to the output subtitle directory",
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

    if FLAGS.languages:
        print("\n".join(Utils.get_language_table()))
        sys.exit(0)
    if FLAGS.mode == "":
        print("--mode was not passed in")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.video_directory == "":
        print("--video_directory was not passed in")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.subtitle_directory == "":
        print("--subtitle_directory was not passed in")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.output_directory == "":
        print("--output_directory was not passed in")
        parser.print_usage()
        sys.exit(21)
    if os.path.abspath(FLAGS.subtitle_directory) == os.path.abspath(FLAGS.output_directory):
        print("The output directory cannot be set to the same as the input subtitle directory")
        parser.print_usage()
        sys.exit(21)

    video_file_paths = [os.path.abspath(os.path.join(path, p)) for path, _, files in
                        os.walk(FLAGS.video_directory) for p in files if not p.startswith(".")]
    subtitle_file_paths = [os.path.abspath(os.path.join(path, p)) for path, _, files in
                           os.walk(FLAGS.subtitle_directory) for p in files if not p.startswith(".")]
    if len(video_file_paths) != len(subtitle_file_paths):
        print("The numbers of input videos and subtitles do not match")
        parser.print_usage()
        sys.exit(21)

    output_dir = os.path.abspath(FLAGS.output_directory)
    os.makedirs(output_dir, exist_ok=True)
    video_file_paths = sorted(video_file_paths, key=lambda x: os.path.splitext(os.path.basename(x))[0])
    subtitle_file_paths = sorted(subtitle_file_paths, key=lambda x: os.path.splitext(os.path.basename(x))[0])
    exit_segfail = FLAGS.exit_segfail
    stretch = not FLAGS.stretch_off
    stretch_in_lang = FLAGS.stretch_in_language

    from subaligner.logger import Logger
    Logger.VERBOSE = FLAGS.debug
    Logger.QUIET = FLAGS.quiet
    from subaligner.predictor import Predictor
    from subaligner.translator import Translator
    from subaligner.subtitle import Subtitle
    from subaligner.exception import UnsupportedFormatException
    from subaligner.exception import TerminalException

    predictor = Predictor()
    failures = []
    for index in range(len(video_file_paths)):
        local_video_path = video_file_paths[index]
        local_subtitle_path = subtitle_file_paths[index]
        try:
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

            parent_dir = os.path.dirname(local_subtitle_path.replace(os.path.abspath(FLAGS.subtitle_directory), output_dir))
            os.makedirs(parent_dir, exist_ok=True)
            aligned_subtitle_path = os.path.abspath(os.path.join(parent_dir,
                                                    ".".join(os.path.basename(local_subtitle_path).rsplit(".", 1)).replace(".stl", ".srt")))

            if FLAGS.translate is not None:
                source, target = FLAGS.translate.split(",")
                translator = Translator(source, target)
                aligned_subs = translator.translate(aligned_subs)
                Subtitle.export_subtitle(local_subtitle_path, aligned_subs, aligned_subtitle_path, frame_rate, "utf-8")
            else:
                Subtitle.export_subtitle(local_subtitle_path, aligned_subs, aligned_subtitle_path, frame_rate)

            log_loss = predictor.get_log_loss(voice_probabilities, aligned_subs)
            if log_loss is None or log_loss > FLAGS.max_logloss:
                print(
                    "Alignment failed with a too high loss value: {} for {} and {}".format(log_loss, local_video_path, local_subtitle_path)
                )
                failures.append((local_video_path, local_subtitle_path))
                continue

            print("Aligned subtitle saved to: {}".format(aligned_subtitle_path))
        except UnsupportedFormatException as e:
            print(
                "{}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
            )
            traceback.print_tb(e.__traceback__)
            failures.append((local_video_path, local_subtitle_path))
            continue
        except TerminalException as e:
            print(
                "{}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
            )
            traceback.print_tb(e.__traceback__)
            failures.append((local_video_path, local_subtitle_path))
            continue
        except Exception as e:
            print(
                "{}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
            )
            traceback.print_tb(e.__traceback__)
            failures.append((local_video_path, local_subtitle_path))
            continue
        else:
            continue

    if len(failures) > 0:
        print("WARNING: The following video and subtitle failed to align with each other:")
        for failure in failures:
            video_file_path, subtitle_file_path = failure
            print("\t{}  {}".format(video_file_path, subtitle_file_path))


if __name__ == "__main__":
    main()
