<div align="center">
  <img src="./figures/subaligner.png" alt="subaligner" width="300" />
</div>

[![Build Status](https://github.com/baxtree/subaligner/actions/workflows/ci-pipeline.yml/badge.svg?branch=master)](https://github.com/baxtree/subaligner/actions/workflows/ci-pipeline.yml?query=branch%3Amaster) ![Codecov](https://img.shields.io/codecov/c/github/baxtree/subaligner)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/) [![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) [![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Documentation Status](https://readthedocs.org/projects/subaligner/badge/?version=latest)](https://subaligner.readthedocs.io/en/latest/?badge=latest)
[![GitHub license](https://img.shields.io/github/license/baxtree/subaligner)](https://github.com/baxtree/subaligner/blob/master/LICENSE)
[![PyPI](https://badge.fury.io/py/subaligner.svg)](https://badge.fury.io/py/subaligner)
[![Docker Hub](https://img.shields.io/docker/cloud/automated/baxtree/subaligner)](https://hub.docker.com/r/baxtree/subaligner)
[![Docker Hub](https://zenodo.org/badge/doi/10.5281/zenodo.1234.svg)](https://doi.org/10.5281/zenodo.1234)

## Supported Formats
Subtitle: SubRip, TTML, WebVTT, (Advanced) SubStation Alpha, MicroDVD, MPL2, TMP, EBU STL, SAMI, SCC and SBV.

Video/Audio: MP4, WebM, Ogg, 3GP, FLV, MOV, Matroska, MPEG TS, WAV, MP3, AAC, FLAC, etc.

## Dependencies
Required by basic: [FFmpeg](https://www.ffmpeg.org/)
```
$ apt-get install ffmpeg
```
or
```
$ brew install ffmpeg
```

## Basic Installation
```
$ pip install -U pip
$ pip install subaligner
```

## Installation with Optional Packages Supporting Additional Features
```
# Install dependencies for enabling translation

$ pip install 'subaligner[translation]'
```
```
# Install dependencies for enabling forced alignment

$ pip install 'subaligner[stretch]'
```
```
# Install dependencies for setting up the development environment

$ pip install 'subaligner[dev]'
```
Note that both `subaligner[stretch]` and `subaligner[dev]` require additional dependencies to be pre-installed:
```
$ apt-get install espeak libespeak1 libespeak-dev espeak-data
```
or
```
$ brew install espeak
```
To install all supported features:
```
$ pip install 'subaligner[harmony]'
```

## Alternative Installations
```
# Install via pipx
$ pip install -U pip pipx
$ pipx install subaligner
```
or
```
# Install from GitHub via Pipenv
$ pipenv install subaligner
$ pipenv install 'subaligner[stretch]'
$ pipenv install 'subaligner[dev]'
```
or
```
# Install from source

$ git clone git@github.com:baxtree/subaligner.git
$ cd subaligner
$ python setup.py install
```
or
```
# Use dockerised installation

$ docker run -v `pwd`:`pwd` -w `pwd` -it baxtree/subaligner bash
```
For users on Windows 10: [Docker Desktop](https://docs.docker.com/docker-for-windows/install/) is the only option at present.
Assuming your media assets are stored under `d:\media`, open built-in command prompt, PowerShell, or Windows Terminal and run:
```
docker pull baxtree/subaligner
docker run -v "/d/media":/media -w "/media" -it baxtree/subaligner bash
```

## Usage
```
# Single-stage alignment (high-level shift with lower latency)

$ subaligner_1pass -v video.mp4 -s subtitle.srt
$ subaligner_1pass -v https://example.com/video.mp4 -s https://example.com/subtitle.srt -o subtitle_aligned.srt
```
```
# Dual-stage alignment (low-level shift with higher latency)

$ subaligner_2pass -v video.mp4 -s subtitle.srt
$ subaligner_2pass -v https://example.com/video.mp4 -s https://example.com/subtitle.srt -o subtitle_aligned.srt
```
or
```
# Pass in single-stage or dual-stage as the alignment mode

$ subaligner -m single -v video.mp4 -s subtitle.srt
$ subaligner -m dual -v video.mp4 -s subtitle.srt
$ subaligner -m single -v https://example.com/video.mp4 -s https://example.com/subtitle.srt -o subtitle_aligned.srt
$ subaligner -m dual -v https://example.com/video.mp4 -s https://example.com/subtitle.srt -o subtitle_aligned.srt
```
```
# Alignment on segmented plain texts (double newlines as the delimiter)

$ subaligner -m script -v test.mp4 -s subtitle.txt -o subtitle_aligned.srt
$ subaligner -m script -v https://example.com/video.mp4 -s https://example.com/subtitle.txt -o subtitle_aligned.srt
```
```
# Translative alignment with the ISO 639-3 language code pair (src,tgt)

$ subaligner_1pass --languages
$ subaligner_1pass -v video.mp4 -s subtitle.srt -t src,tgt
$ subaligner_2pass --languages
$ subaligner_2pass -v video.mp4 -s subtitle.srt -t src,tgt
$ subaligner --languages
$ subaligner -m single -v video.mp4 -s subtitle.srt -t src,tgt
$ subaligner -m dual -v video.mp4 -s subtitle.srt -t src,tgt
$ subaligner -m script -v test.mp4 -s subtitle.txt -o subtitle_aligned.srt -t src,tgt
```
```
# Run batch alignment against directories

$ subaligner_batch -m single -vd videos/ -sd subtitles/ -od aligned_subtitles/
$ subaligner_batch -m dual -vd videos/ -sd subtitles/ -od aligned_subtitles/
$ subaligner_batch -m dual -vd videos/ -sd subtitles/ -od aligned_subtitles/ -of ttml
```
```
# Run alignments with pipx

$ pipx run subaligner -m single -v video.mp4 -s subtitle.srt
$ pipx run subaligner -m dual -v video.mp4 -s subtitle.srt
```
```
# Run the module as a script
$ python -m subaligner -m single -v video.mp4 -s subtitle.srt
$ python -m subaligner -m dual -v video.mp4 -s subtitle.srt
$ python -m subaligner.subaligner_1pass -v video.mp4 -s subtitle.srt
$ python -m subaligner.subaligner_2pass -v video.mp4 -s subtitle.srt
```
```
# Run alignments with the docker image

$ docker pull baxtree/subaligner
$ docker run -v `pwd`:`pwd` -w `pwd` -it baxtree/subaligner subaligner_1pass -v video.mp4 -s subtitle.srt
$ docker run -v `pwd`:`pwd` -w `pwd` -it baxtree/subaligner subaligner_2pass -v video.mp4 -s subtitle.srt
$ docker run -it baxtree/subaligner subaligner_1pass -v https://example.com/video.mp4 -s https://example.com/subtitle.srt -o subtitle_aligned.srt
$ docker run -it baxtree/subaligner subaligner_2pass -v https://example.com/video.mp4 -s https://example.com/subtitle.srt -o subtitle_aligned.srt
```
The aligned subtitle will be saved at `subtitle_aligned.srt`. For details on CLI, run `subaligner_1pass -h`, `subaligner_2pass -h` or `subaligner -h`.
Additional utilities can be used after consulting `subaligner_batch -h`, `subaligner_convert -h`, `subaligner_train -h` and `subaligner_tune -h`.

![](figures/screencast.gif)

## Advanced Usage
You can train a new model with your own audiovisual files and subtitle files:
```
$ subaligner_train -vd VIDEO_DIRECTORY -sd SUBTITLE_DIRECTORY -tod TRAINING_OUTPUT_DIRECTORY
```
Then you can apply it to your subtitle synchronisation with the aforementioned commands. For more details on how to train and tune your own model, please refer to [Subaligner Docs](https://subaligner.readthedocs.io/en/latest/advanced_usage.html).

## Anatomy
Subtitles can be out of sync with their companion audiovisual media files for a variety of causes including latency introduced by Speech-To-Text on live streams or calibration and rectification involving human intervention during post-production.

A model has been trained with synchronised video and subtitle pairs and later used for predicating shifting offsets and directions under the guidance of a dual-stage aligning approach. 

First Stage (Global Alignment):
![](figures/1st_stage.png)

Second Stage (Parallelised Individual Alignment):
![](figures/2nd_stage.png)

## Acknowledgement
Thanks to Alan Robinson and Nigel Megitt for their invaluable feedback.

This tool wouldn't be possible without the following packages:
[librosa](https://librosa.github.io/librosa/)
[tensorflow](https://www.tensorflow.org/)
[scikit-learn](https://scikit-learn.org)
[pycaption](https://pycaption.readthedocs.io)
[pysrt](https://github.com/byroot/pysrt)
[pysubs2](https://github.com/tkarabela/pysubs2)
[aeneas](https://www.readbeyond.it/aeneas/)
[transformers](https://huggingface.co/transformers/).