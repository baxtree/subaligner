import os
from subaligner.predictor import Predictor
from subaligner.subtitle import Subtitle

if __name__ == "__main__":
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(examples_dir, "tmp")
    os.makedirs(output_dir, exist_ok=True)
    video_file_path = os.path.join(examples_dir, "..", "tests/subaligner/resource/test.mp4")
    srt_file_path = os.path.join(examples_dir, "..", "tests/subaligner/resource/test.srt")

    predictor = Predictor()
    subs, audio_file_path, voice_probabilities, frame_rate = predictor.predict_single_pass(video_file_path, srt_file_path)
    aligned_subtitle_path = os.path.join(output_dir, "test_aligned_1.srt")
    Subtitle.export_subtitle(srt_file_path, subs, aligned_subtitle_path)
    print("Aligned subtitle saved to: {}".format(aligned_subtitle_path))

    log_loss = predictor.get_log_loss(voice_probabilities, subs)
    print("Alignment finished with overall loss: {}".format(log_loss))

    subs_list, subs, voice_probabilities, frame_rate = predictor.predict_dual_pass(video_file_path, srt_file_path, stretch=False)
    aligned_subtitle_path = os.path.join(output_dir, "test_aligned_2.srt")
    Subtitle.export_subtitle(srt_file_path, subs_list, aligned_subtitle_path)
    print("Aligned subtitle saved to: {}".format(aligned_subtitle_path))

    log_loss = predictor.get_log_loss(voice_probabilities, subs)
    print("Alignment finished with overall loss: {}".format(log_loss))
