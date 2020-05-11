########################
Installation
########################

**Install necessary dependencies**::

    $ apt-get install ffmpeg
    $ apt-get install espeak libespeak1 libespeak-dev espeak-data

**or**::

    $ brew install ffmpeg espeak

**Install Subaligner via PyPI (pre-emptive NumPy)**::
    $ pip install numpy
    $ pip install subaligner

You can also download the latest
release on `GitHub <https://github.com/baxtree/subaligner>`_ and follow the steps down below
to create a virtual environment and set up all the dependencies:

**Install Subaligner from source**::

    $ git clone git@github.com:baxtree/subaligner.git
    $ cd subaligner
    $ make install && source .venv/bin/activate

**subaligner_1pass and subaligner_2pass should be on your PATH now**::

    (.venv) $ subaligner_1pass --help
    (.venv) $ subaligner_2pass --help

*How to install `HomeBrew <https://brew.sh/>`_