# -*- coding: utf-8 -*-

import subprocess
import os
from radish import given, when, then

PWD = os.path.dirname(os.path.realpath(__file__))
WAIT_TIMEOUT_IN_SECONDS = 120


@given('I have a video file "{file_name:S}"')
def video_file(step, file_name):
    step.context.video_file_path = os.path.join(PWD, "..", "..", "subaligner", "resource", file_name)


@given('I have a subtitle file "{file_name:S}"')
def subtitle_file(step, file_name):
    step.context.subtitle_file_path = os.path.join(PWD, "..", "..", "subaligner", "resource", file_name)


@when("I run the single-stage alignment on them")
def run_subaligner_1pass(step):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_1pass"),
        "-v", step.context.video_file_path,
        "-s", step.context.subtitle_file_path,
        "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when('I run the single-stage alignment on them with output "{file_name:S}"')
def run_subaligner_1pass_with_output(step, file_name):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_1pass"),
        "-v", step.context.video_file_path,
        "-s", step.context.subtitle_file_path,
        "-o", os.path.join(PWD, "..", "..", "subaligner", "resource", file_name),
        "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when("I run the dual-stage alignment on them")
def run_subaligner_2pass(step):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_2pass"),
        "-v", step.context.video_file_path,
        "-s", step.context.subtitle_file_path,
        "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when('I run the dual-stage alignment on them with output "{file_name:S}"')
def run_subaligner_2pass_with_output(step, file_name):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_2pass"),
        "-v", step.context.video_file_path,
        "-s", step.context.subtitle_file_path,
        "-o", os.path.join(PWD, "..", "..", "subaligner", "resource", file_name),
        "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when("I run the dual-stage alignment on them without stretch")
def run_subaligner_2pass_without_stretch(step):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_2pass"),
        "-v", step.context.video_file_path,
        "-s", step.context.subtitle_file_path,
        "-so",
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


@when("I run the single-stage alignment on them with max loss")
def run_subaligner_1pass(step):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_1pass"),
        "-v", step.context.video_file_path,
        "-s", step.context.subtitle_file_path,
        "-l", str(step.context.max_log_loss),
        "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@when("I run the dual-stage alignment on them with max loss")
def run_subaligner_1pass(step):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_2pass"),
        "-v", step.context.video_file_path,
        "-s", step.context.subtitle_file_path,
        "-l", str(step.context.max_log_loss),
        "-q"], shell=False)
    step.context.exit_code = process.wait(timeout=WAIT_TIMEOUT_IN_SECONDS)


@then('it exits with code "{exit_code:d}"')
def assert_exit_code(step, exit_code):
    assert step.context.exit_code == exit_code


@when("I run the single-stage command with help")
def run_subaligner_1pass_with_help(step):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_1pass"),
        "-h"], shell=False, stdout=subprocess.PIPE)
    stdout, _ = process.communicate(timeout=WAIT_TIMEOUT_IN_SECONDS)
    step.context.stdout = stdout.decode("utf-8")


@then("the single-stage help information is displayed")
def expect_help_information(step):
    assert "usage: subaligner_1pass" in step.context.stdout


@when("I run the dual-stage command with help")
def run_subaligner_1pass_with_help(step):
    process = subprocess.Popen([
        os.path.join(PWD, "..", "..", "..", "bin", "subaligner_2pass"),
        "-h"], shell=False, stdout=subprocess.PIPE)
    stdout, _ = process.communicate(timeout=WAIT_TIMEOUT_IN_SECONDS)
    step.context.stdout = stdout.decode("utf-8")


@then("the dual-stage help information is displayed")
def expect_help_information(step):
    assert "usage: subaligner_2pass" in step.context.stdout


@given("I have an unsupported subtitle file")
def unsupported_subtitle(step):
    step.context.subtitle_file_path = os.path.join(PWD, "..", "..", "subaligner", "resource", "unsupported")


@given("I have an unsupported video file")
def unsupported_video(step):
    step.context.video_file_path = os.path.join(PWD, "..", "..", "subaligner", "resource", "unsupported")