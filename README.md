<div align="center">
  <img src="https://raw.githubusercontent.com/baxtree/subaligner/master/figures/subaligner.png" alt="subaligner" width="300" />
</div>

[![Build Status](https://github.com/baxtree/subaligner/actions/workflows/ci-pipeline.yml/badge.svg?branch=master)](https://github.com/baxtree/subaligner/actions/workflows/ci-pipeline.yml?query=branch%3Amaster) ![Codecov](https://img.shields.io/codecov/c/github/baxtree/subaligner)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/) [![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/) [![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/) [![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Documentation Status](https://readthedocs.org/projects/subaligner/badge/?version=latest)](https://subaligner.readthedocs.io/en/latest/?badge=latest)
[![GitHub license](https://img.shields.io/github/license/baxtree/subaligner)](https://github.com/baxtree/subaligner/blob/master/LICENSE)
[![PyPI](https://badge.fury.io/py/subaligner.svg)](https://badge.fury.io/py/subaligner)
[![Docker Pulls](https://img.shields.io/docker/pulls/baxtree/subaligner)](https://hub.docker.com/r/baxtree/subaligner)
[![Citation](https://zenodo.org/badge/228440472.svg)](https://doi.org/10.5281/zenodo.5603083)

## Supported Formats
Subtitle: SubRip, TTML, WebVTT, (Advanced) SubStation Alpha, MicroDVD, MPL2, TMP, EBU STL, SAMI, SCC and SBV.

Video/Audio: MP4, WebM, Ogg, 3GP, FLV, MOV, Matroska, MPEG TS, WAV, MP3, AAC, FLAC, etc.

:information_source: <small style="line-height: 1.2;">Subaligner relies on file extensions as default hints to process a wide range of audiovisual or subtitle formats. It is recommended to use extensions widely acceppted by the community to ensure compatibility.</small>

## Dependant package
Required by the basic installation: [FFmpeg](https://www.ffmpeg.org/)
<details>
<summary>Install FFmpeg</summary>
<pre><code>$ apt-get install ffmpeg
$ brew install ffmpeg
</code></pre>
</details>

## Basic Installation
<details>
<summary>Install from PyPI</summary>
<pre><code>$ pip install -U pip && pip install -U setuptools wheel
$ pip install subaligner
</code></pre>
</details>
<details>
<summary>Install from source</summary>
<pre><code>$ git clone git@github.com:baxtree/subaligner.git && cd subaligner
$ pip install -U pip && pip install -U setuptools
$ pip install .
</code></pre>
</details>
:information_source: <small style="line-height: 1.2;">It is highly recommended creating a virtual environment prior to installation.</small>

## Installation with Optional Packages Supporting Additional Features
<details>
<summary>Install dependencies for enabling translation and transcription</summary>
<pre><code>$ pip install 'subaligner[llm]'
</code></pre>
</details>

<details>
<summary>Install dependencies for enabling forced alignment</summary>
<pre><code>$ pip install 'setuptools<65.0.0'
$ pip install 'subaligner[stretch]'
</code></pre>
</details>

<details>
<summary>Install dependencies for setting up the development environment</summary>
<pre><code>$ pip install 'setuptools<65.0.0'
$ pip install 'subaligner[dev]'
</code></pre>
</details>


<details>
<summary>Install all extra dependencies</summary>
<pre><code>$ pip install 'setuptools<65.0.0'
$ pip install 'subaligner[harmony]'
</code></pre>
</details>

Note that `subaligner[stretch]`, `subaligner[dev]` and `subaligner[harmony]` require [eSpeak](https://espeak.sourceforge.net/) to be pre-installed:
<details>
<summary>Install eSpeak</summary>
<pre><code>$ apt-get install espeak libespeak1 libespeak-dev espeak-data
$ brew install espeak
</code></pre>
</details>

## Container Support
If you prefer using a containerised environment over installing everything locally:
<details>
<summary>Run subaligner with a container</summary>
<pre><code>$ docker run -v `pwd`:`pwd` -w `pwd` -it baxtree/subaligner bash
</code></pre>
</details>

For Windows users, you can use Windows Subsystem for Linux ([WSL](https://learn.microsoft.com/en-us/windows/wsl/install)) to install Subaligner.
Alternatively, you can use [Docker Desktop](https://docs.docker.com/docker-for-windows/install/) to pull and run the image.
Assuming your media assets are stored under `d:\media`, open built-in command prompt, PowerShell, or Windows Terminal:
<details>
<summary>Run the subaligner container on Windows</summary>
<pre><code>docker pull baxtree/subaligner
docker run -v "/d/media":/media -w "/media" -it baxtree/subaligner bash
</code></pre>
</details>

## Usage
<details>
<summary>Single-stage alignment (high-level shift with lower latency)</summary>
<pre><code>$ subaligner -m single -v video.mp4 -s subtitle.srt
$ subaligner -m single -v https://example.com/video.mp4 -s https://example.com/subtitle.srt -o subtitle_aligned.srt
</code></pre>
</details>

<details>
<summary>Dual-stage alignment (low-level shift with higher latency)</summary>
<pre><code>$ subaligner -m dual -v video.mp4 -s subtitle.srt
$ subaligner -m dual -v https://example.com/video.mp4 -s https://example.com/subtitle.srt -o subtitle_aligned.srt
</code></pre>
</details>

<details>
<summary>Generate subtitles by transcribing audiovisual files</summary>
<pre><code>$ subaligner -m transcribe -v video.mp4 -ml eng -mr whisper -mf small -o subtitle_aligned.srt
$ subaligner -m transcribe -v video.mp4 -ml zho -mr whisper -mf medium -o subtitle_aligned.srt
</code></pre>
</details>

<details>
<summary>Pass in a global prompt for the entire audio transcription</summary>
<pre><code>$ subaligner -m transcribe -v video.mp4 -ml eng -mr whisper -mf turbo -ip "your initial prompt" -o subtitle_aligned.srt 
</code></pre>
</details>

<details>
<summary>Use the full subtitle content as a prompt</summary>
<pre><code>$ subaligner -m transcribe -v video.mp4 -s subtitle.srt -ml eng -mr whisper -mf turbo -o subtitle_aligned.srt
</code></pre>
</details>

<details>
<summary>Use the previous subtitle segment as the prompt when transcribing the following segment</summary>
<pre><code>$ subaligner -m transcribe -v video.mp4 -s subtitle.srt --use_prior_prompting -ml eng -mr whisper -mf turbo -o subtitle_aligned.srt
</code></pre>
</details>

(For details on the prompt crafting for transcription, please refer to [Whisper prompting guide](https://cookbook.openai.com/examples/whisper_prompting_guide).)

<details>
<summary>Alignment on segmented plain texts (double newlines as the delimiter)</summary>
<pre><code>$ subaligner -m script -v video.mp4 -s subtitle.txt -o subtitle_aligned.srt
$ subaligner -m script -v https://example.com/video.mp4 -s https://example.com/subtitle.txt -o subtitle_aligned.srt
</code></pre>
</details>

<details>
<summary>Generate JSON raw subtitle with per-word timings</summary>
<pre><code>$ subaligner -m transcribe -v video.mp4 -ml eng -mr whisper -mf turbo -ip "your initial prompt" --word_time_codes -o raw_subtitle.json
$ subaligner -m script -v video.mp4 -s subtitle.txt --word_time_codes -o raw_subtitle.json
</code></pre>
</details>


<details>
<summary>Alignment on multiple subtitles against the single media file</summary>
<pre><code>$ subaligner -m script -v video.mp4 -s subtitle_lang_1.txt -s subtitle_lang_2.txt
$ subaligner -m script -v video.mp4 -s subtitle_lang_1.txt subtitle_lang_2.txt
</code></pre>
</details>

<details>
<summary>Alignment on embedded subtitles</summary>
<pre><code>$ subaligner -m single -v video.mkv -s embedded:stream_index=0 -o subtitle_aligned.srt
$ subaligner -m dual -v video.mkv -s embedded:stream_index=0 -o subtitle_aligned.srt
</code></pre>
</details>

<details>
<summary>Translative alignment with the ISO 639-3 language code pair (src,tgt)</summary>
<pre><code>$ subaligner --languages
$ subaligner -m single -v video.mp4 -s subtitle.srt -t src,tgt
$ subaligner -m dual -v video.mp4 -s subtitle.srt -t src,tgt
$ subaligner -m script -v video.mp4 -s subtitle.txt -o subtitle_aligned.srt -t src,tgt
$ subaligner -m dual -v video.mp4 -s subtitle.srt -tr helsinki-nlp -o subtitle_aligned.srt -t src,tgt
$ subaligner -m dual -v video.mp4 -s subtitle.srt -tr facebook-mbart -tf large -o subtitle_aligned.srt -t src,tgt
$ subaligner -m dual -v video.mp4 -s subtitle.srt -tr facebook-m2m100 -tf small -o subtitle_aligned.srt -t src,tgt
$ subaligner -m dual -v video.mp4 -s subtitle.srt -tr whisper -tf small -o subtitle_aligned.srt -t src,eng
</code></pre>
</details>

<details>
<summary>Transcribe audiovisual files and generate translated subtitles</summary>
<pre><code>$ subaligner -m transcribe -v video.mp4 -ml src -mr whisper -mf small -tr helsinki-nlp -o subtitle_aligned.srt -t src,tgt
</code></pre>
</details>


<details>
<summary>Shift subtitle manually by offset in seconds</summary>
<pre><code>$ subaligner -m shift --subtitle_path subtitle.srt -os 5.5
$ subaligner -m shift --subtitle_path subtitle.srt -os -5.5 -o subtitle_shifted.srt
</code></pre>
</details>

<details>
<summary>Run batch alignment against directories</summary>
<pre><code>$ subaligner_batch -m single -vd videos/ -sd subtitles/ -od aligned_subtitles/
$ subaligner_batch -m dual -vd videos/ -sd subtitles/ -od aligned_subtitles/
$ subaligner_batch -m dual -vd videos/ -sd subtitles/ -od aligned_subtitles/ -of ttml
</code></pre>
</details>

<details>
<summary>Run alignments with pipx</summary>
<pre><code>$ pipx run subaligner -m single -v video.mp4 -s subtitle.srt
$ pipx run subaligner -m dual -v video.mp4 -s subtitle.srt
</code></pre>
</details>

<details>
<summary>Run the module as a script</summary>
<pre><code>$ python -m subaligner -m single -v video.mp4 -s subtitle.srt
$ python -m subaligner -m dual -v video.mp4 -s subtitle.srt
</code></pre>
</details>

<details>
<summary>Run alignments with the docker image</summary>
<pre><code>$ docker pull baxtree/subaligner
$ docker run -v `pwd`:`pwd` -w `pwd` -it baxtree/subaligner subaligner -m single -v video.mp4 -s subtitle.srt
$ docker run -v `pwd`:`pwd` -w `pwd` -it baxtree/subaligner subaligner -m dual -v video.mp4 -s subtitle.srt
$ docker run -it baxtree/subaligner subaligner -m single -v https://example.com/video.mp4 -s https://example.com/subtitle.srt -o subtitle_aligned.srt
$ docker run -it baxtree/subaligner subaligner -m dual -v https://example.com/video.mp4 -s https://example.com/subtitle.srt -o subtitle_aligned.srt
</code></pre>
</details>

![](figures/screencast.gif)

The aligned subtitle will be saved at `subtitle_aligned.srt`. To obtain the subtitle in raw JSON format for downstream
processing, replace the output file extension with `.json`. For details on CLIs, run `subaligner -h` or `subaligner_batch -h`,
`subaligner_convert -h`, `subaligner_train -h` and `subaligner_tune -h` for additional utilities. `subaligner_1pass` and `subaligner_2pass` are shortcuts for running `subaligner` with `-m single` and `-m dual` options, respectively.

## Advanced Usage
You can train a new model with your own audiovisual files and subtitle files,
<details>
<summary>Train a custom model</summary>
<pre><code>$ subaligner_train -vd VIDEO_DIRECTORY -sd SUBTITLE_DIRECTORY -tod TRAINING_OUTPUT_DIRECTORY
</code></pre>
</details>

Then you can apply it to your subtitle synchronisation with the aforementioned commands. For more details on how to train and tune your own model, please refer to [Subaligner Docs](https://subaligner.readthedocs.io/en/latest/advanced_usage.html).

For larger media files taking longer to process, you can reconfigure various timeouts using the following:
<details>
<summary>Options for tuning timeouts</summary>
<pre><code>-mpt [Maximum waiting time in seconds when processing media files]
-sat [Maximum waiting time in seconds when aligning each segment]
-fet [Maximum waiting time in seconds when embedding features for training]
</code></pre>
</details>

## Anatomy
Subtitles can be out of sync with their companion audiovisual media files for a variety of causes including latency introduced by Speech-To-Text on live streams or calibration and rectification involving human intervention during post-production.

A model has been trained with synchronised video and subtitle pairs and later used for predicating shifting offsets and directions under the guidance of a dual-stage aligning approach. 

First Stage (Global Alignment):
![](figures/1st_stage.png)

Second Stage (Parallelised Individual Alignment):
![](figures/2nd_stage.png)

## Acknowledgement
This tool wouldn't be possible without the following packages:
[librosa](https://librosa.github.io/librosa/)
[tensorflow](https://www.tensorflow.org/)
[scikit-learn](https://scikit-learn.org)
[pycaption](https://pycaption.readthedocs.io)
[pysrt](https://github.com/byroot/pysrt)
[pysubs2](https://github.com/tkarabela/pysubs2)
[aeneas](https://www.readbeyond.it/aeneas/)
[transformers](https://huggingface.co/transformers/)
[openai-whisper](https://github.com/openai/whisper).

Thanks to Alan Robinson and Nigel Megitt for their invaluable feedback.
