import os
from subaligner.hyperparameters import Hyperparameters
from subaligner.hparam_tuner import HyperParameterTuner
from subaligner.embedder import FeatureEmbedder
from subaligner.trainer import Trainer

if __name__ == "__main__":
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(examples_dir, "tmp")
    os.makedirs(output_dir, exist_ok=True)
    video_file_path = os.path.join(examples_dir, "..", "tests/subaligner/resource/test.mp4")
    srt_file_path = os.path.join(examples_dir, "..", "tests/subaligner/resource/test.srt")

    hyperparameters = Hyperparameters()
    hparam_tuner = HyperParameterTuner([video_file_path], [srt_file_path], output_dir)
    hparam_tuner.tune_hyperparameters()
    tuned_hyperparameters = hparam_tuner.hyperparameters
    tuned_hyperparameters.epochs = 10

    trainer = Trainer(FeatureEmbedder())
    trainer.train([video_file_path],
                  [srt_file_path],
                  output_dir,
                  output_dir,
                  output_dir,
                  output_dir,
                  output_dir,
                  tuned_hyperparameters)
