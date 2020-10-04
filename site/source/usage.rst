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


**Pass in single-stage or dual-stage as the alignment mode**::

    (.venv) $ subaligner -m single -v video.mp4 -s subtitle.srt
    (.venv) $ subaligner -m dual -v video.mp4 -s subtitle.srt


**Run alignments with the docker image**::

    $ docker pull baxtree/subaligner
    $ docker run -v `pwd`:`pwd` -w `pwd` -it baxtree/subaligner subaligner_1pass -v video.mp4 -s subtitle.srt
    $ docker run -v `pwd`:`pwd` -w `pwd` -it baxtree/subaligner subaligner_2pass -v video.mp4 -s subtitle.srt

**Run alignments with pipx**::

    $ pipx run subaligner -m single -v video.mp4 -s subtitle.srt
    $ pipx run subaligner -m dual -v video.mp4 -s subtitle.srt

**Run the module as a script**::

    $ python -m subaligner -m single -v video.mp4 -s subtitle.srt
    $ python -m subaligner -m dual -v video.mp4 -s subtitle.srt
    $ python -m subaligner.subaligner_1pass -v video.mp4 -s subtitle.srt
    $ python -m subaligner.subaligner_2pass -v video.mp4 -s subtitle.srt

Currently the stretching is experimental and only works for speech and subtitles in English.

**Use flag "-so" to switch off stretching when aligning subtitles not in English**::

    (.venv) $ subaligner_2pass -v video.mp4 -s subtitle.srt -so
    or
    (.venv) $ subaligner -m dual -v video.mp4 -s subtitle.srt -so

**Use flag "-o" to save the aligned subtitle to a specific location**::

    (.venv) $ subaligner_2pass -v video.mp4 -s subtitle.srt -o /path/to/the/output/subtitle.srt
    or
    (.venv) $ subaligner -m dual -v video.mp4 -s subtitle.srt -o /path/to/the/output/subtitle.srt
