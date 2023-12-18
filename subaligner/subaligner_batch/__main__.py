#!/usr/bin/env python
"""
usage: subaligner_batch [-h] [-m {single,dual,script,transcribe}] [-sd SUBTITLE_DIRECTORY] [-vd VIDEO_DIRECTORY] [-l MAX_LOGLOSS] [-so]
                        [-sil {afr,amh,ara,arg,asm,aze,ben,bos,bul,cat,ces,cmn,cym,dan,deu,ell,eng,epo,est,eus,fas,fin,fra,gla,gle,glg,grc,grn,guj,heb,hin,hrv,hun,hye,ina,ind,isl,ita,jbo,jpn,kal,kan,kat,kir,kor,kur,lat,lav,lfn,lit,mal,mar,mkd,mlt,msa,mya,nah,nep,nld,nor,ori,orm,pan,pap,pol,por,ron,rus,sin,slk,slv,spa,sqi,srp,swa,swe,tam,tat,tel,tha,tsn,tur,ukr,urd,vie,yue,zho}]
                        [-fos] [-tod TRAINING_OUTPUT_DIRECTORY] [-od OUTPUT_DIRECTORY] [-of {srt,ytt,ttml,txt,smi,xml,ssa,ass,dfxp,sub,scc,tmp,sami,vtt,stl,sbv}] [-t TRANSLATE]
                        [-ml {afr,amh,ara,arg,asm,aze,ben,bos,bul,cat,ces,cmn,cym,dan,deu,ell,eng,epo,est,eus,fas,fin,fra,gla,gle,glg,grc,grn,guj,heb,hin,hrv,hun,hye,ina,ind,isl,ita,jbo,jpn,kal,kan,kat,kir,kor,kur,lat,lav,lfn,lit,mal,mar,mkd,mlt,msa,mya,nah,nep,nld,nor,ori,orm,pan,pap,pol,por,ron,rus,sin,slk,slv,spa,sqi,srp,swa,swe,tam,tat,tel,tha,tsn,tur,ukr,urd,vie,yue,zho}]
                        [-mr {whisper}] [-mf {tiny,tiny.en,small,medium,medium.en,base,base.en,large-v1,large-v2,large-v3,large}] [-lgs] [-d] [-q] [-ver]

Batch align multiple subtitle files and audiovisual files

Subtitle files and their companion audiovisual files need to be stored in two separate directories.
Each file pair needs to share the same base filename, the part before the extension.

optional arguments:
  -h, --help            show this help message and exit
  -sd SUBTITLE_DIRECTORY, --subtitle_directory SUBTITLE_DIRECTORY
                        Path to the subtitle directory
  -vd VIDEO_DIRECTORY, --video_directory VIDEO_DIRECTORY
                        Path to the video directory
  -l MAX_LOGLOSS, --max_logloss MAX_LOGLOSS
                        Max global log loss for alignment
  -so, --stretch_on     Switch on stretch on subtitles)
  -sil {afr,amh,ara,arg,asm,aze,ben,bos,bul,cat,ces,cmn,cym,dan,deu,ell,eng,epo,est,eus,fas,fin,fra,gla,gle,glg,grc,grn,guj,heb,hin,hrv,hun,hye,ina,ind,isl,ita,jbo,jpn,kal,kan,kat,kir,kor,kur,lat,lav,lfn,lit,mal,mar,mkd,mlt,msa,mya,nah,nep,nld,nor,ori,orm,pan,pap,pol,por,ron,rus,sin,slk,slv,spa,sqi,srp,swa,swe,tam,tat,tel,tha,tsn,tur,ukr,urd,vie,yue,zho}, --stretch_in_language {afr,amh,ara,arg,asm,aze,ben,bos,bul,cat,ces,cmn,cym,dan,deu,ell,eng,epo,est,eus,fas,fin,fra,gla,gle,glg,grc,grn,guj,heb,hin,hrv,hun,hye,ina,ind,isl,ita,jbo,jpn,kal,kan,kat,kir,kor,kur,lat,lav,lfn,lit,mal,mar,mkd,mlt,msa,mya,nah,nep,nld,nor,ori,orm,pan,pap,pol,por,ron,rus,sin,slk,slv,spa,sqi,srp,swa,swe,tam,tat,tel,tha,tsn,tur,ukr,urd,vie,yue,zho}
                        Stretch the subtitle with the supported ISO 639-3 language code [https://en.wikipedia.org/wiki/List_of_ISO_639-3_codes].
                        NB: This will be ignored if neither -so nor --stretch_on is present
  -fos, --exit_segfail  Exit on any segment alignment failures
  -tod TRAINING_OUTPUT_DIRECTORY, --training_output_directory TRAINING_OUTPUT_DIRECTORY
                        Path to the output directory containing training results
  -od OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Path to the output subtitle directory
  -of {srt,ytt,ttml,txt,smi,xml,ssa,ass,dfxp,sub,scc,tmp,sami,vtt,stl,sbv}, --output_format {srt,ytt,ttml,txt,smi,xml,ssa,ass,dfxp,sub,scc,tmp,sami,vtt,stl,sbv}
                        File format of the output subtitles
  -t TRANSLATE, --translate TRANSLATE
                        Source and target ISO 639-3 language codes separated by a comma (e.g., eng,zho)
  -ml {afr,amh,ara,arg,asm,aze,ben,bos,bul,cat,ces,cmn,cym,dan,deu,ell,eng,epo,est,eus,fas,fin,fra,gla,gle,glg,grc,grn,guj,heb,hin,hrv,hun,hye,ina,ind,isl,ita,jbo,jpn,kal,kan,kat,kir,kor,kur,lat,lav,lfn,lit,mal,mar,mkd,mlt,msa,mya,nah,nep,nld,nor,ori,orm,pan,pap,pol,por,ron,rus,sin,slk,slv,spa,sqi,srp,swa,swe,tam,tat,tel,tha,tsn,tur,ukr,urd,vie,yue,zho}, --main_language {afr,amh,ara,arg,asm,aze,ben,bos,bul,cat,ces,cmn,cym,dan,deu,ell,eng,epo,est,eus,fas,fin,fra,gla,gle,glg,grc,grn,guj,heb,hin,hrv,hun,hye,ina,ind,isl,ita,jbo,jpn,kal,kan,kat,kir,kor,kur,lat,lav,lfn,lit,mal,mar,mkd,mlt,msa,mya,nah,nep,nld,nor,ori,orm,pan,pap,pol,por,ron,rus,sin,slk,slv,spa,sqi,srp,swa,swe,tam,tat,tel,tha,tsn,tur,ukr,urd,vie,yue,zho}
                        Target video's main language as an ISO 639-3 language code [https://en.wikipedia.org/wiki/List_of_ISO_639-3_codes]
  -mr {whisper}, --transcription_recipe {whisper}
                        LLM recipe used for transcribing video files
  -mf {tiny,tiny.en,small,medium,medium.en,base,base.en,large-v1,large-v2,large-v3,large}, --transcription_flavour {tiny,tiny.en,small,medium,medium.en,base,base.en,large-v1,large-v2,large-v3,large}
                        Flavour variation for a specific LLM recipe supporting transcription
  -lgs, --languages     Print out language codes used for stretch and translation
  -d, --debug           Print out debugging information
  -q, --quiet           Switch off logging information
  -ver, --version       show program's version number and exit

required arguments:
  -m {single,dual,script,transcribe}, --mode {single,dual,script,transcribe}
                        Alignment mode: either single or dual
"""

import argparse
import sys
import traceback
import os
import tempfile


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
    parser = argparse.ArgumentParser(description="""Batch align multiple subtitle files and audiovisual files (v%s)\n
Subtitle files and their companion audiovisual files need to be stored in two separate directories.
Each file pair needs to share the same base filename, the part before the extension.""" % __version__, formatter_class=argparse.RawTextHelpFormatter)
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument(
        "-m",
        "--mode",
        type=str,
        default="",
        choices=["single", "dual", "script", "transcribe"],
        help="Alignment mode: either single or dual",
    )
    parser.add_argument(
        "-sd",
        "--subtitle_directory",
        type=str,
        default="",
        help="Path to the subtitle directory",
    )
    parser.add_argument(
        "-vd",
        "--video_directory",
        type=str,
        default="",
        help="Path to the video directory",
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
        "--stretch_on",
        action="store_true",
        help="Switch on stretch on subtitles)",
    )
    from subaligner.utils import Utils
    from subaligner.subtitle import Subtitle
    parser.add_argument(
        "-sil",
        "--stretch_in_language",
        type=str,
        choices=Utils.get_stretch_language_codes(),
        default="eng",
        help="Stretch the subtitle with the supported ISO 639-3 language code [https://en.wikipedia.org/wiki/List_of_ISO_639-3_codes].\nNB: This will be ignored if neither -so nor --stretch_on is present",
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
        "-of",
        "--output_format",
        type=str,
        choices=list(map(lambda x: x[1:], Subtitle.subtitle_extensions())),
        default="",
        help="File format of the output subtitles"
    )
    parser.add_argument(
        "-t",
        "--translate",
        type=str,
        help="Source and target ISO 639-3 language codes separated by a comma (e.g., eng,zho)",
    )
    parser.add_argument(
        "-ml",
        "--main_language",
        type=str.lower,
        choices=Utils.get_stretch_language_codes(),
        help="Target video's main language as an ISO 639-3 language code [https://en.wikipedia.org/wiki/List_of_ISO_639-3_codes]",
    )
    from subaligner.llm import TranscriptionRecipe
    from subaligner.llm import WhisperFlavour
    parser.add_argument(
        "-mr",
        "--transcription_recipe",
        type=str.lower,
        default=TranscriptionRecipe.WHISPER.value,
        choices=[r.value for r in TranscriptionRecipe],
        help="LLM recipe used for transcribing video files"
    )
    parser.add_argument(
        "-mf",
        "--transcription_flavour",
        type=str.lower,
        default=WhisperFlavour.SMALL.value,
        choices=[wf.value for wf in WhisperFlavour],
        help="Flavour variation for a specific LLM recipe supporting transcription"
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
        print("ERROR: --mode was not passed in")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.video_directory == "":
        print("ERROR: --video_directory was not passed in")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.mode != "transcribe" and FLAGS.subtitle_directory == "":
        print("ERROR: --subtitle_directory was not passed in")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.output_directory == "":
        print("ERROR: --output_directory was not passed in")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.mode != "transcribe" and os.path.abspath(FLAGS.subtitle_directory) == os.path.abspath(FLAGS.output_directory):
        print("ERROR: The output directory cannot be set to the same as the input subtitle directory")
        parser.print_usage()
        sys.exit(21)
    if FLAGS.translate is not None or FLAGS.mode == "transcribe":
        try:
            import transformers
        except ModuleNotFoundError:
            print('ERROR: Alignment has been configured to use language models. Please install "subaligner[llm]" and run your command again.')
            sys.exit(21)
    if FLAGS.stretch_on or FLAGS.mode == "script":
        try:
            import aeneas
        except ModuleNotFoundError:
            print('ERROR: Alignment has been configured to use extra features. Please install "subaligner[stretch]" and run your command again.')
            sys.exit(21)
    if FLAGS.mode == "transcribe":
        if FLAGS.main_language is None:
            print("ERROR: --main_language was not passed in but required by mode 'transcribe'")
            parser.print_usage()
            sys.exit(21)

    video_file_paths = [os.path.abspath(os.path.join(path, p)) for path, _, files in
                        os.walk(FLAGS.video_directory) for p in files if not p.startswith(".")]

    if FLAGS.mode != "transcribe":
        subtitle_file_paths = [os.path.abspath(os.path.join(path, p)) for path, _, files in
                               os.walk(FLAGS.subtitle_directory) for p in files if not p.startswith(".")]
        if len(video_file_paths) != len(subtitle_file_paths):
            print("ERROR: The numbers of input videos and subtitles do not match")
            parser.print_usage()
            sys.exit(21)

    output_dir = os.path.abspath(FLAGS.output_directory)
    os.makedirs(output_dir, exist_ok=True)
    video_file_paths = sorted(video_file_paths, key=lambda x: os.path.splitext(os.path.basename(x))[0])
    if FLAGS.mode != "transcribe":
        subtitle_file_paths = sorted(subtitle_file_paths, key=lambda x: os.path.splitext(os.path.basename(x))[0])
    exit_segfail = FLAGS.exit_segfail
    stretch = FLAGS.stretch_on
    stretch_in_lang = FLAGS.stretch_in_language

    from subaligner.logger import Logger
    Logger.VERBOSE = FLAGS.debug
    Logger.QUIET = FLAGS.quiet
    from subaligner.predictor import Predictor
    from subaligner.subtitle import Subtitle
    from subaligner.exception import UnsupportedFormatException
    from subaligner.exception import TerminalException

    predictor = Predictor()
    failures = []
    for index in range(len(video_file_paths)):
        local_video_path = video_file_paths[index]
        local_subtitle_path = subtitle_file_paths[index] if FLAGS.mode != "transcribe" else "{}.srt".format(tempfile.mkstemp()[1])
        try:
            voice_probabilities = None
            if FLAGS.mode == "single":
                aligned_subs, audio_file_path, voice_probabilities, frame_rate = predictor.predict_single_pass(
                    video_file_path=local_video_path,
                    subtitle_file_path=local_subtitle_path,
                    weights_dir=os.path.join(FLAGS.training_output_directory, "models", "training", "weights")
                )
            elif FLAGS.mode == "dual":
                aligned_subs, subs, voice_probabilities, frame_rate = predictor.predict_dual_pass(
                    video_file_path=local_video_path,
                    subtitle_file_path=local_subtitle_path,
                    weights_dir=os.path.join(FLAGS.training_output_directory, "models", "training", "weights"),
                    stretch=stretch,
                    stretch_in_lang=stretch_in_lang,
                    exit_segfail=exit_segfail,
                )
            elif FLAGS.mode == "script":
                aligned_subs, _, voice_probabilities, frame_rate = predictor.predict_plain_text(
                    video_file_path=local_video_path,
                    subtitle_file_path=local_subtitle_path,
                    stretch_in_lang=stretch_in_lang,
                )
            elif FLAGS.mode == "transcribe":
                from subaligner.transcriber import Transcriber
                transcriber = Transcriber(recipe=FLAGS.transcription_recipe, flavour=FLAGS.transcription_flavour)
                subtitle, frame_rate = transcriber.transcribe(local_video_path, stretch_in_lang)
                aligned_subs = subtitle.subs

            if FLAGS.mode == "transcribe":
                parent_dir = os.path.dirname(video_file_paths[index].replace(os.path.abspath(FLAGS.video_directory), output_dir))
                os.makedirs(parent_dir, exist_ok=True)
                file_parts = os.path.basename(video_file_paths[index]).rsplit(".", 1)
                file_parts[1] = FLAGS.output_format if FLAGS.output_format != "" else "srt"
                aligned_subtitle_path = os.path.abspath(os.path.join(parent_dir, ".".join(file_parts).replace(".stl", ".srt")))
            else:
                parent_dir = os.path.dirname(local_subtitle_path.replace(os.path.abspath(FLAGS.subtitle_directory), output_dir))
                os.makedirs(parent_dir, exist_ok=True)
                file_parts = os.path.basename(local_subtitle_path).rsplit(".", 1)
                file_parts[1] = FLAGS.output_format if FLAGS.output_format != "" else file_parts[1]
                aligned_subtitle_path = os.path.abspath(os.path.join(parent_dir, ".".join(file_parts).replace(".stl", ".srt")))

            if FLAGS.translate is not None:
                from subaligner.translator import Translator
                source, target = FLAGS.translate.split(",")
                translator = Translator(source, target)
                aligned_subs = translator.translate(aligned_subs)
                Subtitle.save_subs_as_target_format(aligned_subs, local_subtitle_path, aligned_subtitle_path, frame_rate, "utf-8")
            elif FLAGS.mode == "transcribe":
                Subtitle.save_subs_as_target_format(aligned_subs, local_subtitle_path, aligned_subtitle_path, frame_rate, "utf-8")
            else:
                Subtitle.save_subs_as_target_format(aligned_subs, local_subtitle_path, aligned_subtitle_path, frame_rate)

            if voice_probabilities is not None:
                log_loss = predictor.get_log_loss(voice_probabilities, aligned_subs)
                if log_loss is None or log_loss > FLAGS.max_logloss:
                    print(
                        "ERROR: Alignment failed with a too high loss value: {} for {} and {}".format(log_loss, local_video_path, local_subtitle_path)
                    )
                    failures.append((local_video_path, local_subtitle_path))
                    continue

            print("Aligned subtitle saved to: {}".format(aligned_subtitle_path))
        except UnsupportedFormatException as e:
            print(
                "ERROR: {}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
            )
            traceback.print_tb(e.__traceback__)
            failures.append((local_video_path, local_subtitle_path))
            continue
        except TerminalException as e:
            print(
                "ERROR: {}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
            )
            traceback.print_tb(e.__traceback__)
            failures.append((local_video_path, local_subtitle_path))
            continue
        except Exception as e:
            print(
                "ERROR: {}\n{}".format(str(e), "".join(traceback.format_stack()) if FLAGS.debug else "")
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
