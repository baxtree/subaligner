import os
from subaligner.hparam_tuner import HyperParameterTuner

if __name__ == "__main__":
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(examples_dir, "tmp")
    os.makedirs(output_dir, exist_ok=True)
    video_file_path = os.path.join(examples_dir, "..", "tests/subaligner/resource/test.mp4")
    srt_file_path = os.path.join(examples_dir, "..", "tests/subaligner/resource/test.srt")

    hparam_tuner = HyperParameterTuner([video_file_path], [srt_file_path], output_dir)
    hparam_tuner.tune_hyperparameters()

    print(hparam_tuner.hyperparameters.to_json())
