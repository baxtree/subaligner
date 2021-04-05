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
        step.context.video_file_path = os.path.join(PWD, "..", "..", "subaligner", "resource", file_name)


@given('I have a subtitle file "{file_name:S}"')
def subtitle_file(step, file_name):
    if file_name.lower().startswith("http"):
        step.context.subtitle_path_or_selector = file_name
    else:
        step.context.subtitle_path_or_selector = os.path.join(PWD, "..", "..", "subaligner", "resource", file_name)


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


@when("I run the alignment with {aligner:S} on them with {mode:S} stage and without stretch")
def run_subaligner_without_stretch(step, aligner, mode):
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
    output_file_path = os.path.join(PWD, "..", "..", "subaligner", "resource", file_name)
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


@then("the dual-stage help information is displayed")
def expect_dual_stage_help_information(step):
    assert "usage: subaligner_2pass" in step.context.stdout


@given("I have an unsupported subtitle file")
def unsupported_subtitle(step):
    step.context.subtitle_path_or_selector = os.path.join(PWD, "..", "..", "subaligner", "resource", "unsupported")


@given("I have an unsupported video file")
def unsupported_video(step):
    step.context.video_file_path = os.path.join(PWD, "..", "..", "subaligner", "resource", "unsupported")


@given('I have an audiovisual file directory "{av_dir:S}"')
def audiovisual_dir(step, av_dir):
    step.context.av_dir = os.path.join(PWD, "..", "..", "subaligner", "resource", av_dir)


@given('I have a subtitle file directory "{sub_dir:S}"')
def subtitle_dir(step, sub_dir):
    step.context.sub_dir = os.path.join(PWD, "..", "..", "subaligner", "resource", sub_dir)


@given('I want to save the output in directory "{output_dir:S}"')
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


@before.each_scenario(on_tags="train or hyperparameter-tuning")
def create_training_output_dir(scenario):
    scenario.context.temp_dir = tempfile.mkdtemp()


@after.each_scenario(on_tags="train or hyperparameter-tuning")
def remove_training_output_dir(scenario):
    if os.path.isdir(scenario.context.temp_dir):
        shutil.rmtree(scenario.context.temp_dir)
