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
