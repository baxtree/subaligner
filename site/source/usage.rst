########################
Usage
########################

Subaligner provides two ways of aligning subtitles: single-stage alignment and dual-stage alignment. The former way has
lower latency and shifts all subtitle segments globally. The latter way has higher latency and shifts the
segments individually with an option of stretching each segment. Multilingual translation on subtitles can be achieved
together with the alignment in one go or separately (see in :ref:`Advanced Usage`).

With no subtitles in your hand beforehand, Subligner's transcribe mode utilises Large Language Models (LLMs) to transcribe
audiovisual content and generates subtitles in various formats which suit your needs.

Make sure you have got the virtual environment activated upfront.

**Single-stage alignment (high-level shift with lower latency)**::

    (.venv) $ subaligner -m single -v video.mp4 -s subtitle.srt
    (.venv) $ subaligner -m single -v https://example.org/video.mp4 -s https://example.org/subtitle.srt -o subtitle_aligned.srt

**Dual-stage alignment (low-level shift with higher latency)**::

    (.venv) $ subaligner -m dual -v video.mp4 -s subtitle.srt
    (.venv) $ subaligner -m dual -v https://example.org/video.mp4 -s https://example.org/subtitle.srt -o subtitle_aligned.srt

**Generate subtitles by transcribing audiovisual files**::

    (.venv) $ subaligner -m transcribe -v video.mp4 -ml eng -mr whisper -mf small -o subtitle_aligned.srt
    (.venv) $ subaligner -m transcribe -v video.mp4 -ml zho -mr whisper -mf medium -o subtitle_aligned.srt

**Alignment on segmented plain texts (double newlines as the delimiter)**::

    (.venv) $ subaligner -m script -v test.mp4 -s subtitle.txt -o subtitle_aligned.srt
    (.venv) $ subaligner -m script -v https://example.com/video.mp4 -s https://example.com/subtitle.txt -o subtitle_aligned.srt

**Alignment on multiple subtitles against the single media file**::

    (.venv) $ subaligner -m script -v test.mp4 -s subtitle_lang_1.txt -s subtitle_lang_2.txt
    (.venv) $ subaligner -m script -v test.mp4 -s subtitle_lang_1.txt subtitle_lang_2.txt


**Alignment on embedded subtitles**::

    (.venv) $ subaligner -m single -v video.mkv -s embedded:stream_index=0 -o subtitle_aligned.srt
    (.venv) $ subaligner -m dual -v video.mkv -s embedded:stream_index=0 -o subtitle_aligned.srt

**Translative alignment with the ISO 639-3 language code pair (src,tgt)**::

    (.venv) $ subaligner --languages
    (.venv) $ subaligner -m single -v video.mp4 -s subtitle.srt -t src,tgt
    (.venv) $ subaligner -m dual -v video.mp4 -s subtitle.srt -t src,tgt
    (.venv) $ subaligner -m script -v test.mp4 -s subtitle.txt -o subtitle_aligned.srt -t src,tgt
    (.venv) $ subaligner -m dual -v video.mp4 -tr helsinki-nlp -o subtitle_aligned.srt -t src,tgt
    (.venv) $ subaligner -m dual -v video.mp4 -tr facebook-mbart -tf large -o subtitle_aligned.srt -t src,tgt
    (.venv) $ subaligner -m dual -v video.mp4 -tr whisper -tf small -o subtitle_aligned.srt -t src,eng

**Transcribe audiovisual files and generate translated subtitles**::

    (.venv) $ subaligner -m transcribe -v video.mp4 -ml src -mr whisper -mf small -tr helsinki-nlp -o subtitle_aligned.srt -t src,tgt

**Shift subtitle manually by offset in seconds**::

    (.venv) $ subaligner -m shift --subtitle_path subtitle.srt -os 5.5
    (.venv) $ subaligner -m shift --subtitle_path subtitle.srt -os -5.5 -o subtitle_shifted.srt

**Run batch alignment against directories**::

    (.venv) $ subaligner_batch -m single -vd videos/ -sd subtitles/ -od aligned_subtitles/
    (.venv) $ subaligner_batch -m dual -vd videos/ -sd subtitles/ -od aligned_subtitles/
    (.venv) $ subaligner_batch -m dual -vd videos/ -sd subtitles/ -od aligned_subtitles/ -of ttml

**Run alignments with the docker image**::

    $ docker pull baxtree/subaligner
    $ docker run -v `pwd`:`pwd` -w `pwd` -it baxtree/subaligner subaligner -m single -v video.mp4 -s subtitle.srt
    $ docker run -v `pwd`:`pwd` -w `pwd` -it baxtree/subaligner subaligner -m dual -v video.mp4 -s subtitle.srt
    $ docker run -it baxtree/subaligner subaligner -m single -v https://example.com/video.mp4 -s https://example.com/subtitle.srt -o subtitle_aligned.srt
    $ docker run -it baxtree/subaligner subaligner -m dual -v https://example.com/video.mp4 -s https://example.com/subtitle.srt -o subtitle_aligned.srt

**Run alignments with pipx**::

    $ pipx run subaligner -m single -v video.mp4 -s subtitle.srt
    $ pipx run subaligner -m dual -v video.mp4 -s subtitle.srt

**Run the module as a script**::

    $ python -m subaligner -m single -v video.mp4 -s subtitle.srt
    $ python -m subaligner -m dual -v video.mp4 -s subtitle.srt

Currently the stretching is experimental and make sure subaligner[stretch] is installed before switching it on with `-so`
or `--stretch_on` as shown below.

**Switch on stretching when aligning subtitles**::

    (.venv) $ subaligner -m dual -v video.mp4 -s subtitle.srt -so

**Save the aligned subtitle to a specific location**::

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