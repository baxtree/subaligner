########################
Usage
########################

Subaligner provides two ways of aligning subtitles: single-stage alignment and dual-stage alignment. The former way has
lower latency and shifts all subtitle segments globally. The latter way has higher latency and shifts the
segments individually with an option of stretching each segment. Multilingual translation on subtitles can be achieved
together with the alignment in one go or separately (see in :ref:`Advanced Usage`).

Make sure you have got the virtual environment activated upfront.

**Single-stage alignment (high-level shift with lower latency)**::

    (.venv) $ subaligner_1pass -v video.mp4 -s subtitle.srt
    (.venv) $ subaligner_1pass -v https://example.org/video.mp4 -s https://example.org/subtitle.srt -o subtitle_aligned.srt

**Dual-stage alignment (low-level shift with higher latency)**::

    (.venv) $ subaligner_2pass -v video.mp4 -s subtitle.srt
    (.venv) $ subaligner_2pass -v https://example.org/video.mp4 -s https://example.org/subtitle.srt -o subtitle_aligned.srt

**Pass in single-stage or dual-stage as the alignment mode**::

    (.venv) $ subaligner -m single -v video.mp4 -s subtitle.srt
    (.venv) $ subaligner -m single -v https://example.org/video.mp4 -s https://example.org/subtitle.srt -o subtitle_aligned.srt
    (.venv) $ subaligner -m dual -v video.mp4 -s subtitle.srt
    (.venv) $ subaligner -m dual -v https://example.org/video.mp4 -s https://example.org/subtitle.srt -o subtitle_aligned.srt

**Translative alignment with the ISO 639-3 language code pair (src,tgt)**::

    (.venv) $ subaligner_1pass --languages
    (.venv) $ subaligner_1pass -v video.mp4 -s subtitle.srt -t src,tgt
    (.venv) $ subaligner_2pass --languages
    (.venv) $ subaligner_2pass -v video.mp4 -s subtitle.srt -t src,tgt
    (.venv) $ subaligner --languages
    (.venv) $ subaligner -m single -v video.mp4 -s subtitle.srt -t src,tgt
    (.venv) $ subaligner -m dual -v video.mp4 -s subtitle.srt -t src,tgt

**Run batch alignment against directories**::

    (.venv) $ subaligner_batch -m single -vd /videos -sd /subtitles -od /aligned_subtitles
    (.venv) $ subaligner_batch -m dual -vd /videos -sd /subtitles -od /aligned_subtitles

**Run alignments with the docker image**::

    $ docker pull baxtree/subaligner
    $ docker run -v `pwd`:`pwd` -w `pwd` -it baxtree/subaligner subaligner_1pass -v video.mp4 -s subtitle.srt
    $ docker run -v `pwd`:`pwd` -w `pwd` -it baxtree/subaligner subaligner_2pass -v video.mp4 -s subtitle.srt
    $ docker run -it baxtree/subaligner subaligner_1pass -v https://example.com/video.mp4 -s https://example.com/subtitle.srt -o subtitle_aligned.srt
    $ docker run -it baxtree/subaligner subaligner_2pass -v https://example.com/video.mp4 -s https://example.com/subtitle.srt -o subtitle_aligned.srt

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

**On Windows**::

    docker run -v "/d/media":/media -w "/media" -it baxtree/subaligner COMMAND

The aforementioned commands can be run with `Docker Desktop <https://docs.docker.com/docker-for-windows/install/>`_ on Windows 10.

**Re-configure FFmpeg/Libav path**::

    (.venv) $ export FFMPEG_PATH=/path/to/ffmpeg
    (.venv) $ subaligner -m dual -v video.mp4 -s subtitle.srt
    or
    (.venv) $ FFMPEG_PATH=/path/to/ffmpeg subaligner -m dual -v video.mp4 -s subtitle.srt
    or when using `Libav<https://libav.org/>`_
    (.venv) $ FFMPEG_PATH=/path/to/avconv subaligner -m dual -v video.mp4 -s subtitle.srt

The lower case "ffmpeg_path" is also supported.