import os
import cProfile
from subaligner.predictor import Predictor
from subaligner.hyperparameters import Hyperparameters
from subaligner.embedder import FeatureEmbedder
from subaligner.trainer import Trainer


def generate_profiles(file_dir="."):
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(examples_dir, "tmp")
    os.makedirs(output_dir, exist_ok=True)
    video_file_path = os.path.join(examples_dir, "..", "tests/subaligner/resource/test.mp4")
    srt_file_path = os.path.join(examples_dir, "..", "tests/subaligner/resource/test.srt")

    predictor = Predictor()
    cProfile.runctx("predictor.predict_dual_pass(video_file_path, srt_file_path)", None, locals(), os.path.join(file_dir, "predictor.prof"))

    hyperparameters = Hyperparameters()
    hyperparameters.epochs = 10

    trainer = Trainer(FeatureEmbedder())
    cProfile.runctx((
        "trainer.train([video_file_path, video_file_path, video_file_path],"
        "[srt_file_path, srt_file_path, srt_file_path],"
        "output_dir,"
        "output_dir,"
        "output_dir,"
        "output_dir,"
        "output_dir,"
        "hyperparameters)"), None, locals(), os.path.join(file_dir, "trainer.prof"))


def kernprof_target():
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(examples_dir, "tmp")
    os.makedirs(output_dir, exist_ok=True)
    video_file_path = os.path.join(examples_dir, "..", "video.mp4")
    srt_file_path = os.path.join(examples_dir, "..", "video.srt")

    predictor = Predictor()
    predictor.predict_dual_pass(video_file_path, srt_file_path)

    # hyperparameters = Hyperparameters()
    # hyperparameters.epochs = 10
    #
    # trainer = Trainer(FeatureEmbedder())
    # trainer.train([video_file_path, video_file_path, video_file_path],
    #               [srt_file_path, srt_file_path, srt_file_path],
    #               output_dir,
    #               output_dir,
    #               output_dir,
    #               output_dir,
    #               output_dir,
    #               hyperparameters)


if __name__ == "__main__":
    kernprof_target()
