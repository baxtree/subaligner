# -*- coding: utf-8 -*-

import subprocess
import os
import tempfile
import shutil
from radish import given, when, then, before, after  # type: ignore

PWD = os.path.dirname(os.path.realpath(__file__))
WAIT_TIMEOUT_IN_SECONDS = 300


@given('I have a video file "{file_name:S}"')
def video_file(step, file_name):
    if file_name.lower().startswith("http"):
        step.context.video_file_path = file_name
    else:
        step.context.video_file_path = os.path.join(PWD, "..", "..", "subaligner", "resource", file_name).replace("[]", " ")


@given('I have a subtitle file "{file_name:S}"')
def subtitle_file(step, file_name):
    if file_name.lower().startswith("http"):
        step.context.subtitle_path_or_selector = file_name
    else:
        step.context.subtitle_path_or_selector = os.path.join(PWD, "..", "..", "subaligner", "resource", file_name).replace("[]", " ")


@given('I have a list of subtitle files "{file_names:S}"')
def subtitle_file_list(step, file_names):
    step.context.subtitle_path_or_selector = [os.path.join(PWD, "..", "..", "subaligner", "resource", file_name).replace("[]", " ") for file_name in file_names.split(",")]


@when('I run the alignment with subaligner on all of them')
def run_subaligner_on_multi_subtitles(step):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner"),
        "-m", "single",
        "-v", step.context.video_file_path,
        "-q"] + [["-s", path] for path in step.context.subtitle_path_or_selector], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@then('a list of subtitle files "{file_names:S}" are generated')
def expect_result_list(step, file_names):
    for file_name in file_names.split(","):
        output_file_path = os.path.join(step.context.aligning_output, file_name)
        assert os.path.isfile(output_file_path) is True
    assert step.context.exit_code == 0


@given('I have selector "{selector:S}" for the embedded subtitle')
def subtitle_selector(step, selector):
    step.context.subtitle_path_or_selector = selector


@when("I run the alignment with {aligner:S} on them with {mode:S} stage")
def run_subaligner(step, aligner, mode):
    if mode == "<NULL>":
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-q"], shell=False)
    else:
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-m", mode,
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when("I run the alignment with {aligner:S} on them with {mode:S} stage and a short timeout")
def run_subaligner(step, aligner, mode):
    if mode == "<NULL>":
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-mpt", "0",
            "-q"], shell=False)
    else:
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-m", mode,
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-mpt", "0",
            "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when("I run the manual shift with offset of {offset_seconds:g} in seconds")
def run_subaligner_manual_shift(step, offset_seconds):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner"),
        "-m", "shift",
        "-s", step.context.subtitle_path_or_selector,
        "-os", str(offset_seconds),
        "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when("I run the alignment with {aligner:S} on them with {mode:S} stage and {language_pair:S} for translation")
def run_subaligner_with_translation(step, aligner, mode, language_pair):
    if mode == "<NULL>":
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-t", language_pair,
            "-o", os.path.join(PWD, "..", "..", "subaligner", "resource", "test_aligned.srt"),
            "-q"], shell=False)
    else:
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-m", mode,
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-t", language_pair,
            "-o", os.path.join(PWD, "..", "..", "subaligner", "resource", "test_aligned.srt"),
            "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when('I run the alignment with {aligner:S} on them with {mode:S} stage with {language:S} language, {recipe:S} recipe and {flavour:S} flavour and prompt {prompt:S}')
def run_subaligner_with_transcription(step, aligner, mode, language, recipe, flavour, prompt):
    if prompt == "<NULL>":
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-m", mode,
            "-v", step.context.video_file_path,
            "-ml", language,
            "-mr", recipe,
            "-mf", flavour,
            "-o", os.path.join(PWD, "..", "..", "subaligner", "resource", "test_aligned.srt"),
            "-q"], shell=False)
    else:
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-m", mode,
            "-v", step.context.video_file_path,
            "-ml", language,
            "-mr", recipe,
            "-mf", flavour,
            "-ip", prompt,
            "-o", os.path.join(PWD, "..", "..", "subaligner", "resource", "test_aligned.srt"),
            "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when('I run the alignment with {aligner:S} on them with {mode:S} stage with {language:S} language, {recipe:S} recipe and {flavour:S} flavour')
def run_subaligner_with_transcription(step, aligner, mode, language, recipe, flavour):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", aligner),
        "-m", mode,
        "-v", step.context.video_file_path,
        "-s", step.context.subtitle_path_or_selector,
        "-ml", language,
        "-mr", recipe,
        "-mf", flavour,
        "-o", os.path.join(PWD, "..", "..", "subaligner", "resource", "test_aligned.srt"),
        "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when('I run the alignment with subaligner on them with transcribe stage and timed words')
def run_subaligner_with_transcription(step):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner"),
        "-m", "transcribe",
        "-v", step.context.video_file_path,
        "-s", step.context.subtitle_path_or_selector,
        "-wtc",
        "-ml", "eng",
        "-mr", "whisper",
        "-mf", "tiny",
        "-o", os.path.join(PWD, "..", "..", "subaligner", "resource", "test_aligned.json"),
        "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when('I run the alignment with {aligner:S} on them with {mode:S} stage and output "{file_name:S}"')
def run_subaligner_with_output(step, aligner, mode, file_name):
    if mode == "<NULL>":
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-o", os.path.join(PWD, "..", "..", "subaligner", "resource", file_name),
            "-q"], shell=False)
    else:
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-m", mode,
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-o", os.path.join(PWD, "..", "..", "subaligner", "resource", file_name),
            "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when('I run the alignment with subaligner on them with timed words and output "{file_name:S}"')
def run_subaligner_with_timed_words(step, file_name):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner"),
        "-m", "script",
        "-v", step.context.video_file_path,
        "-s", step.context.subtitle_path_or_selector,
        "-wtc",
        "-o", os.path.join(PWD, "..", "..", "subaligner", "resource", file_name),
        "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when("I run the alignment with {aligner:S} on them with {mode:S} stage and with exit_segfail")
def run_subaligner_with_exit_segfail(step, aligner, mode):
    if mode == "<NULL>":
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-es",
            "-q"], shell=False)
    else:
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-m", mode,
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-es",
            "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when("I run the alignment with {aligner:S} on them with {mode:S} stage and with stretch on")
def run_subaligner_with_stretch(step, aligner, mode):
    if mode == "<NULL>":
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-so",
            "-sil", "eng",
            "-q"], shell=False)
    else:
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-m", mode,
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-so",
            "-sil", "eng",
            "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when("I run the alignment with {aligner:S} on them with {mode:S} stage and a custom model")
def run_subaligner_with_custom_model(step, aligner, mode):
    if mode == "<NULL>":
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-tod", os.path.join(PWD, "..", "..", "..", "subaligner"),
            "-q"], shell=False)
    else:
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", aligner),
            "-m", mode,
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-tod", os.path.join(PWD, "..", "..", "..", "subaligner"),
            "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@then('a new subtitle file "{file_name:S}" is generated')
def expect_result(step, file_name):
    output_file_path = os.path.join(PWD, "..", "..", "subaligner", "resource", file_name.replace("[]", " "))
    assert step.context.exit_code == 0
    assert os.path.isfile(output_file_path) is True


@given('I set the max log loss to "{max:f}"')
def set_max_log_loss(step, max):
    step.context.max_log_loss = max


@when("I run the alignment with {alginer:S} on them with {mode:S} stage and max loss")
def run_subaligner_with_max_loss(step, alginer, mode):
    if mode == "<NULL>":
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", alginer),
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-l", str(step.context.max_log_loss),
            "-q"], shell=False)
    else:
        process = subprocess.Popen([
            os.path.join(PWD, "..", "..", "..", "bin", alginer),
            "-m", mode,
            "-v", step.context.video_file_path,
            "-s", step.context.subtitle_path_or_selector,
            "-l", str(step.context.max_log_loss),
            "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@then('it exits with code "{exit_code:d}"')
def assert_exit_code(step, exit_code):
    assert step.context.exit_code == exit_code


@when("I run the {aligner:S} command with help")
def run_subaligner_with_help(step, aligner):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", aligner),
        "-h"], shell=False, stdout=subprocess.PIPE)
    stdout, _ = process.communicate(timeout=WAIT_TIMEOUT_IN_SECONDS)
    step.context.stdout = stdout.decode("utf-8")


@then("{aligner:S} help information is displayed")
def expect_help_information(step, aligner):
    assert "usage: %s " % aligner in step.context.stdout


@when("I run the {aligner:S} command with languages")
def run_subaligner_with_languages(step, aligner):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", aligner),
        "-lgs"], shell=False, stdout=subprocess.PIPE)
    stdout, _ = process.communicate(timeout=WAIT_TIMEOUT_IN_SECONDS)
    step.context.stdout = stdout.decode("utf-8")


@then("supported language codes are displayed")
def expect_language_codes(step):
    assert "eng  English" in step.context.stdout


@then("the dual-stage help information is displayed")
def expect_dual_stage_help_information(step):
    assert "usage: subaligner_2pass" in step.context.stdout


@given("I have an unsupported subtitle file")
def unsupported_subtitle(step):
    step.context.subtitle_path_or_selector = os.path.join(PWD, "..", "..", "subaligner", "resource", "unsupported")


@given("I have an unsupported video file")
def unsupported_video(step):
    step.context.video_file_path = os.path.join(PWD, "..", "..", "subaligner", "resource", "unsupported").replace("[]", " ")


@given('I have an audiovisual file directory "{av_dir:S}"')
def audiovisual_dir(step, av_dir):
    step.context.av_dir = os.path.join(PWD, "..", "..", "subaligner", "resource", av_dir)


@given('I have a subtitle file directory "{sub_dir:S}"')
def subtitle_dir(step, sub_dir):
    step.context.sub_dir = os.path.join(PWD, "..", "..", "subaligner", "resource", sub_dir)


@given('I want to save the training output in directory "{output_dir:S}"')
def output_dir(step, output_dir):
    step.context.training_output = os.path.join(step.context.temp_dir, output_dir)


@when('I run the subaligner_train against them with the following options')
def train(step):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_train"),
        "-vd", step.context.av_dir,
        "-sd", step.context.sub_dir,
        "-tod", step.context.training_output,
        "-q"] + step.text.split(" "), shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when('I run the subaligner_train with subtitle selector "{subtitle_selector:S}" and the following options')
def train_with_subtitle_selector(step, subtitle_selector):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_train"),
        "-vd", step.context.av_dir,
        "-ess", subtitle_selector,
        "-tod", step.context.training_output,
        "-q"] + step.text.split(" "), shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@then("a model and a training log file are generated")
def model_trained(step):
    output_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(step.context.training_output)) for f in fn]
    assert step.context.exit_code == 0
    assert os.path.join(step.context.training_output, "models", "training", "config", "hyperparameters.json") in output_files
    assert os.path.join(step.context.training_output, "models", "training", "model", "combined.hdf5") in output_files
    assert os.path.join(step.context.training_output, "models", "training", "model", "model.hdf5") in output_files
    assert os.path.join(step.context.training_output, "models", "training", "weights", "weights.hdf5") in output_files
    assert os.path.join(step.context.training_output, "training.log") in output_files
    assert os.path.join(step.context.training_output, "training_dump.hdf5") in output_files


@then("a model and a training log file are not generated")
def model_trained(step):
    output_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(step.context.training_output)) for f in fn]
    assert step.context.exit_code == 21
    assert output_files == []


@then('the embedded subtitles are extracted into "{subtitle_dir:S}"')
def embedded_subtitle_extracted(step, subtitle_dir):
    step.context.ext_dir = os.path.join(PWD, "..", "..", "subaligner", "resource", subtitle_dir)
    assert os.path.isdir(step.context.ext_dir)
    av_files = [file for file in os.listdir(step.context.av_dir) if os.path.isfile(os.path.join(step.context.av_dir, file))]
    subtitle_files = [file for file in os.listdir(step.context.ext_dir) if os.path.isfile(os.path.join(step.context.ext_dir, file))]
    assert len(subtitle_files) == len(av_files)


@then("a hyperparameter file is generated")
def hyperparam_tuned(step):
    output_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(step.context.training_output)) for f in fn]
    assert step.context.exit_code == 0
    assert os.path.join(step.context.training_output, "hyperparameters.json") in output_files


@when("I run the subaligner_tune against them with the following flags")
def tuning_configuration(step):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_tune"),
        "-vd", step.context.av_dir,
        "-sd", step.context.sub_dir,
        "-tod", step.context.training_output,
        "-ept", step.table[0]["epoch_per_trail"],
        "-t", step.table[0]["trails"],
        "-nt", step.table[0]["network_type"],
        "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when("I run the subaligner_train to display the finished epochs")
def run_train_display(step):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_train"),
        "-dde",
        "-tod", step.context.training_output,
        "-q"] + step.text.split(" "), shell=False, stdout=subprocess.PIPE)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)
    step.context.done_epochs = process.stdout.read().decode("utf-8")


@then("it shows the done epochs equal to {done_epochs:S}")
def return_done_epochs(step, done_epochs):
    assert step.context.done_epochs == "Number of epochs done: %s\n" % format(done_epochs)


@when('I run the converter with "{output_subtitle:S}" as the output')
def run_subtitle_converter(step, output_subtitle):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_convert"),
        "-i", step.context.subtitle_path_or_selector,
        "-o", os.path.join(PWD, "..", "..", "subaligner", "resource", output_subtitle),
        "-fr", "25.0",
        "-q"] + step.text.split(" "), shell=False, stdout=subprocess.PIPE)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when('I run the converter with {language_pair:S} for translation and "{output_subtitle:S}" as the output')
def run_subtitle_converter_with_translation(step, language_pair, output_subtitle):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_convert"),
        "-i", step.context.subtitle_path_or_selector,
        "-o", os.path.join(PWD, "..", "..", "subaligner", "resource", output_subtitle),
        "-fr", "25.0",
        "-t", language_pair,
        "-q"] + step.text.split(" "), shell=False, stdout=subprocess.PIPE)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@given('I want to save the alignment output in directory "{aligned_dir:S}"')
def aligned_dir(step, aligned_dir):
    step.context.aligning_output = os.path.join(step.context.temp_dir, aligned_dir)


@given('I want to save the alignment output in directory "{aligned_dir:S}" with format "{format:S}"')
def aligned_dir(step, aligned_dir, format):
    step.context.aligning_output = os.path.join(step.context.temp_dir, aligned_dir)
    step.context.subtitle_format = format


@when('I run the subaligner_batch on them with {mode:S} stage')
def run_subaligner_batch(step, mode):
    cmd = [
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_batch"),
        "-m", mode,
        "-vd", step.context.av_dir,
        "-sd", step.context.sub_dir,
        "-od", step.context.aligning_output,
        "-q"
    ]
    if hasattr(step.context, "subtitle_format"):
        cmd.extend(["-of", step.context.subtitle_format])
    process = subprocess.Popen(cmd, shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@then('a new subtitle file "{file_name:S}" is generated in the above output directory')
def expect_result_in_aligning_output(step, file_name):
    output_file_path = os.path.join(step.context.aligning_output, file_name)
    assert step.context.exit_code == 0
    assert os.path.isfile(output_file_path) is True


@before.each_scenario(on_tags="train or hyperparameter-tuning or batch")
def create_training_output_dir(scenario):
    scenario.context.temp_dir = tempfile.mkdtemp()


@after.each_scenario(on_tags="train or hyperparameter-tuning or batch")
def remove_training_output_dir(scenario):
    if os.path.isdir(scenario.context.temp_dir):
        shutil.rmtree(scenario.context.temp_dir)
    if hasattr(scenario.context, "ext_dir") and os.path.isdir(scenario.context.ext_dir):
        shutil.rmtree(scenario.context.ext_dir)
