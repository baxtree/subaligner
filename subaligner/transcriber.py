import os
import torch
import concurrent.futures
import math
import librosa
import re
import statistics
import gc
import multiprocessing as mp
import soundfile as sf
import numpy as np
from functools import partial
from threading import Lock
from typing import Tuple, Optional, Dict, List, Any, Union, cast
from dataclasses import field
from typing import TypedDict
from pysrt import SubRipTime
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.models.whisper.tokenization_whisper import LANGUAGES
from scipy.ndimage import median_filter
from dataclasses import dataclass
from tqdm import tqdm
from .subtitle import Subtitle
from .media_helper import MediaHelper
from .llm import TranscriptionRecipe, WhisperFlavour
from .logger import Logger
from .utils import Utils
from .exception import NoFrameRateException, TranscriptionException
from typing import List, Tuple


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class _WordTimestamp(TypedDict):
    word: str
    start: float
    end: float
    probability: float


@dataclass
class _WhisperWordTiming:
    word: str
    start: float
    end: float
    probability: float = 0.0
    tokens: List[int] = field(default_factory=list)


class Transcriber(object):
    """Transcribe audiovisual content for subtitle generation.

    Arguments:
        recipe {string} -- the LLM recipe used for transcribing video files (default: "whisper").
        flavour {string} -- the flavour variation for a specific LLM recipe (default: "small").

    Raises:
        NotImplementedError: Thrown when the LLM recipe is unknown.
    """

    def __init__(self, recipe: str = TranscriptionRecipe.WHISPER.value, flavour: str = WhisperFlavour.SMALL.value) -> None:
        if recipe not in [r.value for r in TranscriptionRecipe]:
            raise NotImplementedError(f"Unknown recipe: {recipe}")

        self.__LOGGER = Logger().get_logger(__name__)
        self.__recipe = recipe
        self.__flavour = flavour
        self.__media_helper = MediaHelper()
        self.__lock = Lock()
        self.vad_model: Optional[Any] = None

        if recipe == TranscriptionRecipe.WHISPER.value:
            if flavour not in [f.value for f in WhisperFlavour]:
                raise NotImplementedError(f"Unknown {recipe} flavour: {flavour}")
            device = self.__get_device()
            model_id = f"openai/whisper-{flavour}"

            try:
                self.__LOGGER.info(f"Try to load {model_id} from local cache...")
                self.processor = WhisperProcessor.from_pretrained(
                    model_id, local_files_only=True
                )
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_id, attn_implementation="eager", local_files_only=True
                ).to(device)
                self.__LOGGER.info(f"Successfully loaded {model_id} from cache")
            except (OSError, ValueError) as e:
                self.__LOGGER.info(f"Model not found in cache. Downloading {model_id} from HuggingFace Hub...")
                self.processor = WhisperProcessor.from_pretrained(model_id)
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    model_id, attn_implementation="eager"
                ).to(device)
                self.__LOGGER.info(f"Successfully downloaded and loaded {model_id}")

                self.model.eval()
                self.model.config.use_cache = True
        else:
            raise NotImplementedError(f"The '{recipe}' recipe is recognised but not yet supported")

    def transcribe(self,
                   video_file_path: str,
                   language_code: str,
                   initial_prompt: Optional[str] = None,
                   max_char_length: int = 37,
                   with_word_time_codes: bool = False,
                   sample_rate: int = 16000,
                   is_audio: bool = False,
                   batch_size: int = 2) -> Tuple[Subtitle, Optional[float]]:
        """Transcribe an audiovisual file and generate subtitles.

        Arguments:
            video_file_path {string} -- The input video file path.
            language_code {string} -- An alpha 3 language code derived from ISO 639-3.
            initial_prompt {string} -- Optional Text to provide the transcribing context or specific phrases.
            max_char_length {int} -- Optional Maximum number of characters for each generated subtitle segment (default: 37).
            with_word_time_codes {bool} -- True to output time codes for each word (default: {False})
            sample_rate {int} -- The target sample rate for audio extraction (default: 16000)
            is_audio {bool} -- The flag to indicate if the input file is an audio file (default: False)
            batch_size {int} -- The batch size for processing audio segments (default: 2)

        Returns:
            tuple: Generated subtitle after transcription and the detected frame rate

        Raises:
            TranscriptionException: Thrown when transcription is failed.
            NotImplementedError: Thrown when the LLM recipe is not supported.
        """
        if self.__recipe == "whisper":
            lang = Utils.get_iso_639_alpha_2(language_code)
            if lang not in LANGUAGES:
                raise TranscriptionException(f'"{language_code}" is not supported by {self.__recipe} ({self.__flavour})')
            if not is_audio:
                audio_file_path = self.__media_helper.extract_audio(video_file_path, True, sample_rate)
            else:
                audio_file_path = video_file_path
            try:
                self.__LOGGER.info("Start transcribing the audio...")
                self.__LOGGER.debug("Prompting with: '%s'" % initial_prompt)

                audio, sr = self.__load_audio(audio_file_path, target_sample_rate=sample_rate)
                segments, self.vad_model = Utils.vad_segment(
                    audio, sample_rate=sr, recipe="silero", model_local=self.vad_model
                )
                self.__LOGGER.info("Segments detected with voice activities")

                final_segments = []
                for i, (start, end) in enumerate(tqdm(segments, desc="Transcribing segments"), 1):
                    seg_audio = audio[start:end]
                    candidates = self.__transcribe_segment_with_confidence(
                        self.processor,
                        self.model,
                        seg_audio,
                        sr=sr,
                        num_beams=5,
                        num_return_sequences=1,
                        temperature=0.0,
                        no_repeat_ngram_size=3,
                        repetition_penalty=1.05,
                        device=self.model.device,
                        lang=lang,
                        initial_prompt=initial_prompt,
                    )
                    self.__LOGGER.debug(f"Got transcription candidates for segment {str(start)} => {str(end)}")

                    best = self.__choose_best_candidate_by_acoustic(candidates)

                    if best is None:
                        chosen_text = ""
                        flagged = True
                    elif best.get("no_speech_prob") is not None and best["no_speech_prob"] > 0.5:
                        chosen_text = ""
                        flagged = True
                    else:
                        chosen_text = best["text"]
                        flagged = False

                    final_segments.append({
                        "start": start / sr,
                        "end": end / sr,
                        "text": chosen_text,
                        "avg_logprob": best["avg_logprob"] if best is not None else float("-inf"),
                        "no_speech_prob": best.get("no_speech_prob") if best is not None else None,
                        "flagged": flagged,
                    })
                    self.__LOGGER.debug(f"Selected the best candidate for segment {str(start)} => {str(end)}")

                    del candidates, best, seg_audio
                    gc.collect()

                chunks: List[Dict[str, Any]] = []
                for seg in final_segments:
                    if not chunks:
                        chunks.append(seg.copy())
                    else:
                        prev = chunks[-1]
                        gap = seg["start"] - prev["end"]
                        if gap < 0.5 and not prev["flagged"] and not seg["flagged"]:
                            prev["end"] = seg["end"]
                            prev["text"] = (prev["text"] + " " + seg["text"]).strip()
                            prev["avg_logprob"] = float(np.mean([prev["avg_logprob"], seg["avg_logprob"]]))
                        else:
                            chunks.append(seg.copy())

                self.__LOGGER.info("Extracting word timestamps...")
                word_timestamp_tasks = []
                for seg in chunks:
                    if seg["text"] and seg["text"] not in ("", "[INAUDIBLE]"):
                        seg_start_sample = int(seg["start"] * sr)
                        seg_end_sample = int(seg["end"] * sr)
                        seg_audio = audio[seg_start_sample:seg_end_sample]
                        inputs = self.processor(seg_audio, sampling_rate=sr, return_tensors="pt")
                        mel = inputs.input_features[0]
                        num_frames = mel.shape[-1]
                        word_timestamp_tasks.append((seg, mel, num_frames))

                        del inputs, seg_audio
                        gc.collect()
                    else:
                        seg["words"] = []

                for i in range(0, len(word_timestamp_tasks), batch_size):
                    batch = word_timestamp_tasks[i:i + batch_size]
                    self.__process_word_timestamps_batch(batch, lang, max_char_length)

                self.__LOGGER.info("Finished extracting the timed words")

                srt_str = ""
                srt_idx = 1
                for chunk in chunks:
                    start_time = chunk["start"]
                    end_time = chunk["end"]
                    text = chunk["text"].strip()
                    if text:
                        if with_word_time_codes:
                            for word in chunk["words"]:
                                srt_str += f"{srt_idx}\n" \
                                           f"{Utils.format_timestamp(word['start'])} --> {Utils.format_timestamp(word['end'])}\n" \
                                           f"{word['word'].strip().replace('-->', '->')}\n" \
                                           "\n"
                                srt_idx += 1
                        elif max_char_length and len(text) > max_char_length and chunk.get("words"):
                            segment = {
                                "text": text,
                                "words": chunk["words"]
                            }
                            srt_str, srt_idx = self.__chunk_segment(segment, srt_str, srt_idx, max_char_length, start_time, end_time)
                        else:
                            srt_str += f"{srt_idx}\n" \
                                       f"{Utils.format_timestamp(start_time)} --> {Utils.format_timestamp(end_time)}\n" \
                                       f"{text.replace('-->', '->')}\n" \
                                       "\n"
                            srt_idx += 1
                subtitle = Subtitle.load_subrip_str(srt_str)
                self.__LOGGER.debug("Generated the raw subtitle")

                if is_audio:
                    return subtitle, None

                subtitle, frame_rate = self.__on_frame_timecodes(subtitle, video_file_path)
                return subtitle, frame_rate
            finally:
                if not is_audio and os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
        else:
            raise NotImplementedError(f"{self.__recipe} ({self.__flavour}) is not supported")

    def transcribe_with_subtitle_as_prompts(self,
                                            video_file_path: str,
                                            subtitle_file_path: str,
                                            language_code: str,
                                            max_char_length: int = 37,
                                            use_prior_prompting: bool = False,
                                            with_word_time_codes: bool = False,
                                            sample_rate: int = 16000) -> Tuple[Subtitle, Optional[float]]:
        """Transcribe an audiovisual file and generate subtitles using the original subtitle (with accurate time codes) as prompts.


        Arguments:
            video_file_path {string} -- The input video file path.
            subtitle_file_path {string} -- The input subtitle file path to provide prompts.
            language_code {string} -- An alpha 3 language code derived from ISO 639-3.
            max_char_length {int} -- Optional Maximum number of characters for each generated subtitle segment (default: {37}).
            use_prior_prompting {bool} -- Whether to use the previous subtitle cue as the current prompt.
            with_word_time_codes {bool} -- True to output time codes for each word (default: {False})
            sample_rate {int} -- The target sample rate for audio extraction (default: 16000)

        Returns:
            tuple: Generated subtitle after transcription and the detected frame rate

        Raises:
            TranscriptionException: Thrown when transcription is failed.
            NotImplementedError: Thrown when the LLM recipe is not supported.
        """
        if self.__recipe == "whisper":
            lang = Utils.get_iso_639_alpha_2(language_code)
            if lang not in LANGUAGES:
                raise TranscriptionException(
                    f'"{language_code}" is not supported by {self.__recipe} ({self.__flavour})'
                )
            audio_file_path = self.__media_helper.extract_audio(video_file_path, True, sample_rate)
            subtitle = Subtitle.load(subtitle_file_path)
            segment_paths: List[str] = []
            try:
                srt_str = ""
                srt_idx = 1
                self.__LOGGER.info("Start transcribing the audio...")
                segment_paths = []
                transcribe_args = []
                longest_segment_char_length = 0
                prev_sub_text = ""
                for sub in tqdm(subtitle.subs, desc="Extracting audio segments"):
                    segment_path, _ = self.__media_helper.extract_audio_from_start_to_end(
                        audio_file_path, str(sub.start), str(sub.end)
                    )
                    segment_paths.append(segment_path)
                    if use_prior_prompting:
                        transcribe_args.append((
                            segment_path,
                            sub.text if prev_sub_text == "" else prev_sub_text,
                            self.__lock,
                            self.__LOGGER,
                            max_char_length,
                            False,
                            with_word_time_codes,
                        ))
                        prev_sub_text = sub.text
                    else:
                        transcribe_args.append((
                            segment_path,
                            sub.text,
                            self.__lock,
                            self.__LOGGER,
                            max_char_length,
                            use_prior_prompting,
                            with_word_time_codes,
                        ))
                    if len(sub.text) > longest_segment_char_length:
                        longest_segment_char_length = len(sub.text)
                max_subtitle_char_length = max_char_length or longest_segment_char_length

                max_workers = math.ceil(float(os.getenv("MAX_WORKERS", mp.cpu_count() / 2)))
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(partial(self._whisper_transcribe, lang=lang), transcribe_args))

                for sub, result in zip(subtitle.subs, results):
                    original_start_in_secs = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000.0
                    original_end_in_secs = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000.0
                    if len(result["segments"]) == 0:
                        srt_str += f"{srt_idx}\n" \
                                   f"{Utils.format_timestamp(original_start_in_secs)} --> {Utils.format_timestamp(original_end_in_secs)}\n" \
                                   f"{sub.text.strip().replace('-->', '->')}\n" \
                                   "\n"
                        srt_idx += 1
                    else:
                        for segment in result["segments"]:
                            if segment["end"] <= segment["start"]:
                                continue
                            segment_end = min(original_start_in_secs + segment["end"], original_end_in_secs)
                            if len(segment["text"]) > max_subtitle_char_length:
                                srt_str, srt_idx = self.__chunk_segment(segment,
                                                                        srt_str,
                                                                        srt_idx,
                                                                        max_subtitle_char_length,
                                                                        original_start_in_secs,
                                                                        original_end_in_secs)
                            else:
                                srt_str += f"{srt_idx}\n" \
                                           f"{Utils.format_timestamp(original_start_in_secs + segment['start'])} --> {Utils.format_timestamp(segment_end)}\n" \
                                           f"{segment['text'].strip().replace('-->', '->')}\n" \
                                           "\n"
                                srt_idx += 1
                            if segment_end == original_end_in_secs:
                                break
                self.__LOGGER.info("Finished transcribing the audio")
                subtitle = Subtitle.load_subrip_str(srt_str)
                subtitle, frame_rate = self.__on_frame_timecodes(subtitle, video_file_path)
                self.__LOGGER.debug("Generated the raw subtitle")
                return subtitle, frame_rate
            finally:
                if os.path.exists(audio_file_path):
                    os.remove(audio_file_path)
                for segment_path in segment_paths:
                    if os.path.exists(segment_path):
                        os.remove(segment_path)
        else:
            raise NotImplementedError(f"{self.__recipe} ({self.__flavour}) is not supported")

    @staticmethod
    def __chunk_segment(segment: Dict,
                        srt_str: str,
                        srt_idx: int,
                        max_subtitle_char_length: int,
                        start_offset: float = 0.0,
                        end_ceiling: float = float("inf")) -> Tuple[str, int]:
        if len(segment.get("words", [])) == 0:
            srt_str += f"{srt_idx}\n" \
                       f"{Utils.format_timestamp(start_offset)} --> {Utils.format_timestamp(min(end_ceiling, start_offset + 1.0))}\n" \
                       f"{segment['text'].strip().replace('-->', '->')}\n" \
                       "\n"
            srt_idx += 1
            return srt_str, srt_idx

        chunked_text = ""
        chunk_start_in_secs = None
        chunk_end_in_secs = 0.0
        chunk_char_length = 0

        first_word_start = segment["words"][0]["start"] if segment["words"] else 0.0

        for word in segment["words"]:
            if chunk_start_in_secs is None:
                chunk_start_in_secs = word["start"] - first_word_start

            word_to_add = " " + word["word"] if chunked_text.strip() != "" else word["word"]
            if chunk_char_length + len(word_to_add) > max_subtitle_char_length and chunked_text.strip() != "":
                srt_str += f"{srt_idx}\n" \
                           f"{Utils.format_timestamp(start_offset + (chunk_start_in_secs or 0))} --> {Utils.format_timestamp(min(start_offset + chunk_end_in_secs, end_ceiling))}\n" \
                           f"{chunked_text.strip().replace('-->', '->')}\n" \
                           "\n"
                srt_idx += 1

                chunked_text = word["word"]
                chunk_start_in_secs = word["start"] - first_word_start
                chunk_char_length = len(word["word"])
            else:
                chunked_text += word_to_add
                chunk_char_length += len(word_to_add)

            chunk_end_in_secs = word["end"] - first_word_start

        if chunked_text.strip() != "":
            srt_str += f"{srt_idx}\n" \
                       f"{Utils.format_timestamp(start_offset + (chunk_start_in_secs or 0))} --> {Utils.format_timestamp(min(start_offset + chunk_end_in_secs, end_ceiling))}\n" \
                       f"{chunked_text.strip().replace('-->', '->')}\n" \
                       "\n"
            srt_idx += 1

        return srt_str, srt_idx

    @staticmethod
    def __load_audio(path: str, target_sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
        audio, sr = sf.read(path, dtype="float32")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if sr != target_sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate, res_type="kaiser_best")
            sr = target_sample_rate

        if np.issubdtype(audio.dtype, np.integer):
            maxv = np.iinfo(audio.dtype).max
            audio = audio.astype("float32") / maxv

        audio = audio.astype("float32")

        return audio, sr

    @staticmethod
    def __token_logprobs_from_logits(logits: torch.Tensor, token_ids: List[int]) -> np.ndarray:
        if logits.dim() == 3:
            logits = logits[0]
        logprobs = torch.log_softmax(logits, dim=-1)
        lps = []
        for i, token_id in enumerate(token_ids):
            if i < logprobs.size(0):
                lps.append(float(logprobs[i, token_id].cpu().numpy()))
            else:
                lps.append(float(-1e9))
        return np.array(lps, dtype=float)

    @staticmethod
    def __compute_candidate_acoustic_scores(processor: WhisperProcessor,
                                            model: WhisperForConditionalGeneration,
                                            input_features: torch.Tensor,
                                            sequences: List[List[int]],
                                            device: str = "cpu"):
        results = []
        input_features = input_features.to(device)

        for seq in sequences:
            decoder_input_ids = torch.tensor([seq[:-1]], dtype=torch.long).to(device)
            with torch.no_grad():
                out = model(input_features=input_features, decoder_input_ids=decoder_input_ids, return_dict=True)
                logits = out.logits
            token_ids = seq[1:1 + logits.shape[1]]
            per_step_logprobs = Transcriber.__token_logprobs_from_logits(logits, token_ids)
            avg_logprob = float(per_step_logprobs.mean()) if per_step_logprobs.size > 0 else float("-inf")
            acoustic_total = float(per_step_logprobs.sum()) if per_step_logprobs.size > 0 else float("-1e9")
            text = processor.batch_decode([seq], skip_special_tokens=True)[0].strip()
            results.append({
                "text": text,
                "token_ids": token_ids,
                "token_logprobs": per_step_logprobs,
                "avg_logprob": avg_logprob,
                "acoustic_total": acoustic_total,
            })

            del decoder_input_ids, out, logits
            gc.collect()
        return results

    @staticmethod
    def __transcribe_segment_with_confidence(processor: WhisperProcessor,
                                             model: WhisperForConditionalGeneration,
                                             audio_segment: np.ndarray,
                                             sr: int = 16000,
                                             num_beams: int = 5,
                                             num_return_sequences: int = 1,
                                             temperature: float = 0.0,
                                             no_repeat_ngram_size: int = 3,
                                             repetition_penalty: float = 1.0,
                                             device: str = "cpu",
                                             lang: str = "en",
                                             initial_prompt: Optional[str] = None):
        model.eval()
        model.config.use_cache = True

        inputs = processor(audio_segment, sampling_rate=sr, return_tensors="pt")
        inputs.input_features = inputs.input_features.to(device)
        with torch.no_grad():
            encoder_outputs = model.model.encoder(inputs.input_features)

        attention_mask = torch.ones(encoder_outputs.last_hidden_state.shape[:2], dtype=torch.long, device=device)

        prompt_ids = None
        if initial_prompt:
            prompt_ids = processor.get_prompt_ids(initial_prompt, return_tensors="pt").to(device)

        gen_kwargs = dict(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            return_dict_in_generate=True,
            output_scores=False,
            language=lang,
            task="transcribe",
            do_sample=False,
            early_stopping=True,
            use_cache=True,
        )

        if prompt_ids is not None:
            gen_kwargs["prompt_ids"] = prompt_ids
        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)
        sequences = outputs.sequences.cpu().numpy().tolist()

        candidates = Transcriber.__compute_candidate_acoustic_scores(
            processor, model, inputs.input_features, sequences, device=device
        )
        return candidates

    @staticmethod
    def __choose_best_candidate_by_acoustic(candidates: List[dict]) -> Optional[dict]:
        best = None
        best_score = -1e99
        for c in candidates:
            score = c.get("acoustic_total", -1e9)
            if score > best_score:
                best_score = score
                best = c
        return best

    @staticmethod
    def __get_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def __whisper_merge_punctuations(alignment: List[_WhisperWordTiming], prepended: str, appended: str):
        # This function is borrowed from openai-whisper
        i = len(alignment) - 2
        j = len(alignment) - 1
        while i >= 0:
            previous = alignment[i]
            following = alignment[j]
            if previous.word.startswith(" ") and previous.word.strip() in prepended:
                following.word = previous.word + following.word
                following.tokens = previous.tokens + following.tokens
                previous.word = ""
                previous.tokens = []
            else:
                j = i
            i -= 1

        i = 0
        j = 1
        while j < len(alignment):
            previous = alignment[i]
            following = alignment[j]
            if not previous.word.endswith(" ") and following.word in appended:
                previous.word = previous.word + following.word
                previous.tokens = previous.tokens + following.tokens
                following.word = ""
                following.tokens = []
            else:
                i = j
            j += 1

    @staticmethod
    def __dtw(x: torch.Tensor) -> np.ndarray:
        # This function is borrowed from openai-whisper
        x_np = x.float().cpu().numpy()
        N, M = x_np.shape
        cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
        trace = -np.ones((N + 1, M + 1), dtype=np.float32)

        cost[0, 0] = 0
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                c0 = cost[i - 1, j - 1]
                c1 = cost[i - 1, j]
                c2 = cost[i, j - 1]

                if c0 < c1 and c0 < c2:
                    c, t = c0, 0
                elif c1 < c0 and c1 < c2:
                    c, t = c1, 1
                else:
                    c, t = c2, 2

                cost[i, j] = x_np[i - 1, j - 1] + c
                trace[i, j] = t

        i = trace.shape[0] - 1
        j = trace.shape[1] - 1
        trace[0, :] = 2
        trace[:, 0] = 1

        result = []
        while i > 0 or j > 0:
            result.append((i - 1, j - 1))

            if trace[i, j] == 0:
                i -= 1
                j -= 1
            elif trace[i, j] == 1:
                i -= 1
            elif trace[i, j] == 2:
                j -= 1
            else:
                raise ValueError("Unexpected trace[i, j]")

        result = np.array(result)
        return result[::-1, :].T

    def __on_frame_timecodes(self, subtitle: Subtitle, video_file_path: str) -> Tuple[Subtitle, Optional[float]]:
        frame_rate = None
        try:
            frame_rate = self.__media_helper.get_frame_rate(video_file_path)
            frame_duration = 1.0 / frame_rate
            for sub in subtitle.subs:
                start_seconds = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000.0
                end_seconds = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000.0
                start_frames = int(start_seconds / frame_duration)
                end_frames = int(end_seconds / frame_duration)
                sub.start = SubRipTime(seconds=start_frames * frame_duration)
                sub.end = SubRipTime(seconds=end_frames * frame_duration)
        except NoFrameRateException:
            self.__LOGGER.warning("Cannot detect the frame rate for %s" % video_file_path)
        return subtitle, frame_rate

    def _whisper_transcribe(self, transcribe_args: Tuple, lang: str) -> Dict:
        segment_path, sub_text, lock, logger, max_char_length, use_prior_prompting, with_word_time_codes = transcribe_args
        try:
            with lock:
                subtitle, _ = self.transcribe(
                    video_file_path=segment_path,
                    language_code=lang,
                    initial_prompt=sub_text if sub_text and use_prior_prompting else None,
                    max_char_length=max_char_length,
                    with_word_time_codes=with_word_time_codes,
                    is_audio=True,
                )

                segments = []
                for sub in subtitle.subs:
                    start_seconds = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000.0
                    end_seconds = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000.0
                    segments.append({
                        "start": start_seconds,
                        "end": end_seconds,
                        "text": sub.text.strip()
                    })

                logger.debug("Segment transcribed using transcribe(): %s", {"segments": segments})
                return {"segments": segments}
        except Exception as e:
            logger.warning(f"Error while transcribing segment: {e}")
            return {"segments": []}

    def __process_word_timestamps_batch(self,
                                        batch: List[Tuple[Dict[str, Any], Any, Any]],
                                        lang: str,
                                        max_char_length: int) -> None:
        if not batch:
            return

        batch_texts = []
        batch_mels = []
        batch_frames = []
        batch_segs = []

        for seg, mel, num_frames in batch:
            batch_texts.append(seg["text"])
            batch_mels.append(mel)
            batch_frames.append(num_frames)
            batch_segs.append(seg)

        stacked_mels = torch.stack(batch_mels).to(self.model.device)

        try:
            batch_alignments = self.__find_alignment_hf_batch(
                self.model, self.processor.tokenizer, batch_texts, lang, stacked_mels, batch_frames  # type: ignore
            )

            for i, seg in enumerate(batch_segs):
                if i < len(batch_alignments):
                    words_with_timestamps = self.__get_word_timestamps(
                        batch_alignments[i],
                        seg["text"],
                        seg["start"],
                        seg["end"],
                        max_char_length,
                    )
                    seg["words"] = words_with_timestamps
                else:
                    seg["words"] = []
        except Exception as e:
            self.__LOGGER.warning(f"Batch word timestamp processing failed: {e}")
            for seg in batch_segs:
                seg["words"] = []

        del stacked_mels, batch_mels
        gc.collect()

    def __find_alignment_hf_batch(self,
                                  model: PreTrainedModel,
                                  tokenizer: PreTrainedTokenizer,
                                  batch_texts: List[str],
                                  lang: str,
                                  batch_mels: torch.Tensor,
                                  batch_frames: List[int],
                                  medfilt_width: int = 7,
                                  qk_scale: float = 1.0) -> List[List[_WhisperWordTiming]]:
        device = model.device
        batch_size = len(batch_texts)
        tokens_per_second = 50

        sot_sequence = [
            tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
            tokenizer.convert_tokens_to_ids(f"<|{lang}|>"),
            tokenizer.convert_tokens_to_ids("<|transcribe|>"),
        ]

        batch_encoded = tokenizer(batch_texts, add_special_tokens=False, padding=False)
        eot_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

        batch_text_tokens = []
        for text_tokens in batch_encoded["input_ids"]:
            if eot_id is not None:
                text_tokens = [t for t in text_tokens if t != eot_id]
            batch_text_tokens.append(text_tokens)

        valid_indices: List[int] = [i for i, tokens in enumerate(batch_text_tokens) if tokens]
        if not valid_indices:
            return [[]] * batch_size

        max_text_len = max(len(batch_text_tokens[i]) for i in valid_indices)
        batch_tokens_list = []
        batch_mel_list = []
        valid_frames = []

        for i in valid_indices:
            text_tokens = batch_text_tokens[i]
            tokens = [
                *sot_sequence,
                tokenizer.convert_tokens_to_ids("<|notimestamps|>"),
                *text_tokens,
                tokenizer.convert_tokens_to_ids("<|endoftext|>")
            ]
            batch_tokens_list.append(tokens)
            batch_mel_list.append(batch_mels[i])
            valid_frames.append(batch_frames[i])

        max_token_len = max(len(tokens) for tokens in batch_tokens_list)
        padded_tokens = []
        for tokens in batch_tokens_list:
            padded = tokens + [eot_id] * (max_token_len - len(tokens))
            padded_tokens.append(padded)

        batched_tokens = torch.tensor(padded_tokens, device=device)
        batched_mels = torch.stack(batch_mel_list, dim=0)

        target_layer_idx = 0
        QK_batch: List[Union[torch.Tensor, None]] = [None]
        hooks = []

        def __create_hook():
            def hook(module, inputs, outputs):
                QK_batch[0] = outputs[-1][:, 0:1, :, :]
            return hook

        target_layer = model.model.decoder.layers[target_layer_idx]
        hooks.append(
            target_layer.encoder_attn.register_forward_hook(__create_hook())
        )

        try:
            with torch.no_grad():
                decoder_input_ids = batched_tokens[:, :-1]
                outputs = model(
                    input_features=batched_mels,
                    decoder_input_ids=decoder_input_ids
                )
                logits = outputs.logits

            for hook in hooks:
                hook.remove()

            batch_results: List[List[_WhisperWordTiming]] = [[]] * batch_size
            text_start = len(sot_sequence) + 1

            for batch_idx, orig_idx in enumerate(valid_indices):
                try:
                    text_tokens = batch_text_tokens[orig_idx]
                    num_frames = valid_frames[batch_idx]
                    text_end = text_start + len(text_tokens)

                    sampled_logits = logits[batch_idx, text_start:text_end, :]
                    token_probs = sampled_logits.float().softmax(dim=-1)
                    text_token_probs = token_probs[
                        torch.arange(len(text_tokens)), text_tokens
                    ].tolist()

                    qk_tensor = QK_batch[0]
                    assert qk_tensor is not None, "Attention weights not captured"
                    weights = qk_tensor[batch_idx:batch_idx + 1, 0, :, :num_frames // 2]
                    weights = weights.float()
                    weights = (weights * qk_scale).softmax(dim=-1)

                    if medfilt_width > 1:
                        weights_np = weights[0].cpu().numpy()
                        for k in range(weights_np.shape[0]):
                            weights_np[k] = median_filter(
                                weights_np[k],
                                size=medfilt_width
                            )
                        weights = torch.tensor(weights_np, device=device).unsqueeze(0)

                    matrix = weights[0, text_start:text_end]

                    assert matrix.shape[0] == len(text_tokens), (
                        f"DTW matrix mismatch: {matrix.shape[0]} vs {len(text_tokens)}"
                    )

                    text_indices, time_indices = self.__dtw(-matrix)

                    words, word_tokens = self.__split_to_word_tokens(tokenizer, text_tokens)
                    if len(word_tokens) <= 1:
                        batch_results[orig_idx] = []
                        continue

                    word_boundaries = np.cumsum([0] + [len(t) for t in word_tokens])

                    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
                    jump_times = time_indices[jumps] / tokens_per_second
                    max_jump = len(jump_times) - 1
                    safe_starts = np.clip(word_boundaries[:-1], 0, max_jump)
                    safe_ends = np.clip(word_boundaries[1:], 0, max_jump)
                    start_times = jump_times[safe_starts]
                    end_times = jump_times[safe_ends]

                    word_probabilities = [
                        np.mean(text_token_probs[i:j])
                        for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
                    ]

                    alignment = []
                    for word, tokens_, start, end, probability in zip(
                        words, word_tokens, start_times, end_times, word_probabilities
                    ):
                        alignment.append(_WhisperWordTiming(
                            word=word,
                            tokens=tokens_,
                            start=float(start),
                            end=float(end),
                            probability=float(probability)
                        ))

                    batch_results[orig_idx] = alignment

                except Exception as e:
                    self.__LOGGER.warning(f"Error in DTW alignment for item {orig_idx}: {e}")
                    import traceback
                    self.__LOGGER.warning(traceback.format_exc())
                    batch_results[orig_idx] = []

        finally:
            for hook in hooks:
                hook.remove()

        return batch_results

    def __split_to_word_tokens(self, tokenizer, text_tokens):
        full_text = tokenizer.decode(text_tokens)

        if " " in full_text:
            words = full_text.split()
        else:
            words = list(full_text)

        pieces = [tokenizer.decode([token_id]) for token_id in text_tokens]
        word_tokens = []
        token_idx = 0

        for word in words:
            tokens_for_word = []
            accumulated = ""
            while token_idx < len(pieces) and len(accumulated) < len(word):
                p = pieces[token_idx]
                accumulated += p
                tokens_for_word.append(text_tokens[token_idx])
                token_idx += 1
            if accumulated == word:
                word_tokens.append(tokens_for_word)
            else:
                continue

        if not word_tokens:
            word_tokens = [[] for _ in words]

        return words, word_tokens

    def __get_word_timestamps(self,
                              alignment: List[_WhisperWordTiming],
                              text: str,
                              seg_start: float,
                              seg_end: float,
                              max_char_length: int,
                              min_word_dur: float = 0.02,
                              prepend_punctuations: str = "\"'“¿([{-",
                              append_punctuations: str = "\"'.。,，!！?？:：”)]}、") -> List[_WordTimestamp]:
        if not alignment:
            self.__LOGGER.debug("Empty alignment received, falling back to proportional split.")
            return self.__fallback_proportional_split(
                text, seg_start, seg_end, max_char_length, min_word_dur
            )

        self.__whisper_merge_punctuations(alignment, prepend_punctuations, append_punctuations)

        toks = []
        for wt in alignment:
            if isinstance(wt, dict):
                word = wt.get("word", "")
                raw_s = float(wt.get("start", seg_start))
                raw_e = float(wt.get("end", seg_start))
                probability = float(wt.get("probability", 1.0))
            else:
                word = wt.word
                raw_s = float(wt.start)
                raw_e = float(wt.end)
                probability = float(getattr(wt, "probability", 1.0))

            if not word:
                continue

            if np.isnan(raw_s) or np.isnan(raw_e):
                raw_s = raw_e = seg_start
            if raw_e < raw_s:
                raw_e = raw_s

            toks.append({
                "word": word,
                "raw_start": raw_s,
                "raw_end": raw_e,
                "probability": probability,
            })

        if not toks:
            return self.__fallback_proportional_split(
                text, seg_start, seg_end, max_char_length, min_word_dur
            )

        word_durations = np.array([max(0.0, t["raw_end"] - t["raw_start"]) for t in toks])
        median_duration = float(np.median(word_durations)) if word_durations.size > 0 else 0.0
        median_duration = min(0.7, median_duration)
        max_duration = max(0.0, median_duration * 2.0)
        sent_end_punc = set(list(".。!！?？"))

        for idx, t in enumerate(toks):
            dur = t["raw_end"] - t["raw_start"]
            if max_duration > 0 and dur > max_duration:
                last_char = t["word"].strip()[-1] if t["word"].strip() else ""
                if last_char in sent_end_punc:
                    t["raw_end"] = t["raw_start"] + max_duration
                elif idx > 0:
                    prev_last = toks[idx - 1]["word"].strip()[-1] if toks[idx - 1]["word"].strip() else ""
                    if prev_last in sent_end_punc:
                        t["raw_start"] = t["raw_end"] - max_duration

        for t in toks:
            cur_dur = t["raw_end"] - t["raw_start"]
            if cur_dur < min_word_dur:
                mid = (t["raw_start"] + t["raw_end"]) / 2.0
                half = min_word_dur / 2.0
                t["raw_start"] = max(seg_start, mid - half)
                t["raw_end"] = min(seg_end, mid + half)

        n = len(toks)
        eps_zero = 1e-8
        zero_mask = [(toks[i]["raw_end"] - toks[i]["raw_start"]) <= eps_zero for i in range(n)]

        i = 0
        while i < n:
            if not zero_mask[i]:
                i += 1
                continue
            j = i
            while j + 1 < n and zero_mask[j + 1]:
                j += 1
            run_len = j - i + 1

            left_anchor = toks[i - 1]["raw_end"] if i - 1 >= 0 else seg_start
            right_anchor = toks[j + 1]["raw_start"] if (j + 1) < n else seg_end

            if right_anchor < left_anchor:
                right_anchor = left_anchor

            span = right_anchor - left_anchor

            if span <= eps_zero:
                total_needed = run_len * min_word_dur
                if left_anchor + total_needed <= seg_end + 1e-8:
                    dur_each = min_word_dur
                else:
                    dur_each = span / run_len if span > eps_zero else 0.0
            else:
                dur_each = span / run_len

            cur = left_anchor
            for k in range(i, j + 1):
                s = max(seg_start, min(seg_end, cur))
                e = max(seg_start, min(seg_end, cur + dur_each))
                if e < s:
                    e = s
                toks[k]["raw_start"] = s
                toks[k]["raw_end"] = e
                cur = e

            i = j + 1

        for idx in range(n):
            if idx == 0:
                toks[idx]["raw_start"] = max(seg_start, min(seg_end, toks[idx]["raw_start"]))
            else:
                toks[idx]["raw_start"] = max(toks[idx]["raw_start"], toks[idx - 1]["raw_end"])
                toks[idx]["raw_start"] = max(seg_start, min(seg_end, toks[idx]["raw_start"]))

            toks[idx]["raw_end"] = max(toks[idx]["raw_end"], toks[idx]["raw_start"] + min_word_dur)
            toks[idx]["raw_end"] = min(seg_end, toks[idx]["raw_end"])

        for idx in range(n - 2, -1, -1):
            if toks[idx]["raw_end"] > toks[idx + 1]["raw_start"]:
                new_end = min(toks[idx]["raw_end"], toks[idx + 1]["raw_start"])
                if new_end - toks[idx]["raw_start"] < min_word_dur:
                    toks[idx + 1]["raw_start"] = toks[idx]["raw_start"] + min_word_dur
                    toks[idx + 1]["raw_start"] = min(toks[idx + 1]["raw_start"], seg_end)
                    new_end = toks[idx + 1]["raw_start"]
                toks[idx]["raw_end"] = new_end

        toks[0]["raw_start"] = seg_start
        toks[-1]["raw_end"] = seg_end

        durations_after = np.array([max(0.0, t["raw_end"] - t["raw_start"]) for t in toks])
        tiny_count = int((durations_after <= min_word_dur + 1e-9).sum())
        if n > 0 and (tiny_count / n) > 0.30:
            self.__LOGGER.debug(
                f"Alignment produced {tiny_count}/{n} tokens <= min_word_dur; falling back to proportional split"
            )
            return self.__fallback_proportional_split(
                text, seg_start, seg_end, max_char_length, min_word_dur
            )

        out: List[_WordTimestamp] = []
        for t in toks:
            s = float(max(seg_start, min(seg_end, t["raw_start"])))
            e = float(max(seg_start, min(seg_end, t["raw_end"])))
            if e <= s:
                e = min(seg_end, s + min_word_dur)
            out.append({
                "word": t["word"],
                "start": round(s, 3),
                "end": round(e, 3),
                "probability": float(t["probability"]),
            })

        for k in range(1, len(out)):
            if out[k]["start"] < out[k - 1]["end"]:
                out[k]["start"] = out[k - 1]["end"]
                if out[k]["end"] <= out[k]["start"]:
                    out[k]["end"] = round(min(seg_end, out[k]["start"] + min_word_dur), 3)

        if out:
            out[-1]["end"] = round(seg_end, 3)

        return cast(List[_WordTimestamp], out)

    def __fallback_proportional_split(self,
                                      text: str,
                                      seg_start: float,
                                      seg_end: float,
                                      max_char_length: int,
                                      min_word_dur: float = 0.02) -> List[_WordTimestamp]:
        try:
            full_text = text or ""
            words = full_text.split()
            if not words:
                words = re.findall(r"\S+", full_text.strip())
        except Exception as e:
            self.__LOGGER.warning(f"Text splitting failed: {e}")
            words = re.findall(r"\S+", (text or "").strip())

        if not words:
            return []

        processed_words = []
        for w in words:
            if len(w) <= max_char_length:
                processed_words.append(w)
            else:
                for i in range(0, len(w), max_char_length):
                    subword = w[i:i + max_char_length]
                    processed_words.append(subword)
        words = processed_words

        duration = max(seg_end - seg_start, 0.0)

        if duration <= 0:
            out_list = []
            cur_t = seg_start
            for w in words:
                out_list.append({
                    "word": w,
                    "start": round(cur_t, 3),
                    "end": round(cur_t + min_word_dur, 3),
                    "probability": 1.0
                })
                cur_t += min_word_dur
            if out_list:
                out_list[-1]["end"] = round(seg_end, 3)
            return cast(List[_WordTimestamp], out_list)

        lengths = [max(len(w), 1) for w in words]
        total_len = sum(lengths) if sum(lengths) > 0 else len(words)
        raw_durs = [(length / total_len) * duration for length in lengths]
        raw_durs = [max(d, min_word_dur) for d in raw_durs]

        if len(raw_durs) != len(words):
            self.__LOGGER.warning(f"Duration list length mismatch: {len(raw_durs)} vs {len(words)}")
            raw_durs = raw_durs[:len(words)] if len(raw_durs) > len(words) else raw_durs + [min_word_dur] * (len(words) - len(raw_durs))

        med = statistics.median([d for d in raw_durs if d > 0]) if raw_durs else 0.0
        med = min(0.7, float(med))
        max_duration = med * 2.0
        if max_duration > 0:
            raw_durs = [min(d, max_duration) for d in raw_durs]

        s = sum(raw_durs)
        if s > 0:
            scale = duration / s
            raw_durs = [d * scale for d in raw_durs]
        else:
            raw_durs = [duration / len(words)] * len(words)

        if len(raw_durs) != len(words):
            raw_durs = raw_durs[:len(words)] if len(raw_durs) > len(words) else raw_durs + [min_word_dur] * (len(words) - len(raw_durs))

        out: List[_WordTimestamp] = []
        cur_t = seg_start
        for i, (w, d) in enumerate(zip(words, raw_durs)):
            w_start = cur_t
            w_end = cur_t + d
            out.append({
                "word": w,
                "start": round(w_start, 3),
                "end": round(w_end, 3),
                "probability": 1.0
            })
            cur_t = w_end

        if len(out) != len(words):
            self.__LOGGER.warning(f"Output mismatch: got {len(out)} words but expected {len(words)}")
            for i in range(len(out), len(words)):
                out.append({
                    "word": words[i],
                    "start": round(cur_t, 3),
                    "end": round(cur_t + min_word_dur, 3),
                    "probability": 1.0
                })
                cur_t += min_word_dur

        for i in range(1, len(out)):
            if out[i]["start"] < out[i - 1]["end"]:
                out[i]["start"] = out[i - 1]["end"]
                if out[i]["end"] <= out[i]["start"]:
                    out[i]["end"] = out[i]["start"] + min_word_dur

        if out:
            out[-1]["end"] = round(seg_end, 3)
            if out[-1]["end"] - out[-1]["start"] < min_word_dur:
                out[-1]["start"] = round(max(seg_start, out[-1]["end"] - min_word_dur), 3)

        return cast(List[_WordTimestamp], out)
