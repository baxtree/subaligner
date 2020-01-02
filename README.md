[![Build Status](https://travis-ci.com/baxtree/subaligner.svg?branch=master)](https://travis-ci.com/baxtree/subaligner) ![Codecov](https://img.shields.io/codecov/c/github/baxtree/subaligner)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/) [![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)](https://www.python.org/downloads/release/python-350/) [![Python 3.4](https://img.shields.io/badge/python-3.4-blue.svg)](https://www.python.org/downloads/release/python-340/) 
[![Documentation Status](https://readthedocs.org/projects/subaligner/badge/?version=latest)](https://subaligner.readthedocs.io/en/latest/?badge=latest)
[![GitHub license](https://img.shields.io/github/license/baxtree/subaligner)](https://github.com/baxtree/subaligner/blob/master/LICENSE)

## Dependencies
[FFmpeg](https://www.ffmpeg.org/) and [eSpeak](http://espeak.sourceforge.net/index.html)
```
apt-get install ffmpeg espeak libespeak1 libespeak-dev espeak-data
```
or
```
brew install ffmpeg espeak
```

## Installation
```
make install && source .venv/bin/activate
```

## Usage
```
# Single-stage alignment (high-level shift with lower latency)

    (.venv) $ subaligner_1pass -v video.mp4 -s subtitle.srt
```

```
# Dual-stage alignment (low-level shift with higher latency)

(.venv) $ subaligner_2pass -v video.mp4 -s subtitle.srt
```

The aligned subtitle will be saved at `subtitle_aligned.srt`. For details on CLI, run `subaligner_1pass --help` or `subaligner_2pass --help`.

## Supported Formats
Subtitle: SubRip and TTML

Video: MP4, WebM, Ogg, 3GP, FLV and MOV 

## Anatomy
Subtitles can be out of sync with their companion audiovisual media files for a variety of causes including latency introduced by Speech-To-Text on live streams or calibration and rectification involving human intervention during post-production.

A model has been trained with synchronised video and subtitle pairs and later used for predicating shifting offsets and directions under the guidance of a two-stage aligning approach. 

First Stage (Global Alignment):
![](figures/1st_stage.png)

Second Stage (Parallelised Individual Alignment):
![](figures/2nd_stage.png)


