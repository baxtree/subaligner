########################
Usage
########################

Subaligner provides two ways of aligning subtitles: single-stage alignment and dual-stage alignment. The former way has
lower latency and shifts all subtitle segments globally. The latter way has higher latency and shifts the
segments individually with an option of stretching each segment.

Make sure you have got the virtual environment activated upfront.

**Single-stage alignment**::

    (.venv) $ subaligner_1pass -v video.mp4 -s subtitle.srt

**Dual-stage alignment**::

    (.venv) $ subaligner_2pass -v video.mp4 -s subtitle.srt

Currently the stretching is experimental and only works for speech and subtitles in English.

**Use flag "-so" to switch off stretching when aligning subtitles not in English**::

    (.venv) $ subaligner_2pass -v video.mp4 -s subtitle.srt -so

