########################
Installation
########################

**Install necessary dependencies**::

    $ apt-get install ffmpeg

**or**::

    $ brew install ffmpeg

§ You may also need to install `HomeBrew <https://brew.sh/>`_.

**Install Subaligner via PyPI (pre-emptive NumPy)**::

    $ pip install -U pip && pip install -U setuptools
    $ pip install subaligner

**Install dependencies for enabling translation**::

    $ pip install 'subaligner[llm]'

**Pre-install additional dependencies before installing subaligner[stretch] or subaligner[dev]**::

    $ apt-get install espeak libespeak1 libespeak-dev espeak-data

**or**::

    $ brew install espeak

**Install dependencies for enabling forced alignment**::

    $ pip install 'subaligner[stretch]'

**Install dependencies for setting up the development environment**::

    $ pip install 'subaligner[dev]'

**Install all supported features**::

    $ pip install 'subaligner[harmony]'

**Install Subaligner via pipx**::

    $ pipx install subaligner
    $ pipx install 'subaligner[stretch]'
    $ pipx install 'subaligner[dev]'

**Install from GitHub via Pipenv**::

    $ pipenv install subaligner
    $ pipenv install 'subaligner[stretch]'
    $ pipenv install 'subaligner[dev]'

**Container Support**::

    $ docker run -v `pwd`:`pwd` -w `pwd` -it baxtree/subaligner bash

Users may prefer using a containerised environment over installing everything locally. The following builds are available on dockerhub for several Linux distributions: CentOS 7 (latest and VERSION.el7), CentOS 8 (VERSION.el8), Ubuntu 18 (VERSION.u18), Ubuntu 20 (VERSION.u20), Debian 10 (VERSION.deb10), Fedora 31 (VERSION.fed31) and ArchLinux (VERSION.arch).

You can also download the latest
release on `GitHub <https://github.com/baxtree/subaligner>`_ and follow the steps down below
to create a virtual environment and set up all the dependencies:

**Install Subaligner from source**::

    $ git clone git@github.com:baxtree/subaligner.git && cd subaligner
    $ pip install -U pip && pip install -U setuptools
    $ python setup.py install

**Subaligner CLI should be on your PATH now**::

    (.venv) $ subaligner --help
    (.venv) $ subaligner_1pass --help # shortcut for "subaligner -m single"
    (.venv) $ subaligner_2pass --help # shortcut for "subaligner -m dual"
    (.venv) $ subaligner_batch --help
    (.venv) $ subaligner_convert --help
    (.venv) $ subaligner_train --help
    (.venv) $ subaligner_tune --help

**On Windows with Docker Desktop**::

    docker pull baxtree/subaligner
    docker run -v "/d/media":/media -w "/media" -it baxtree/subaligner bash

Assuming that your media assets are stored under "d:\\media", open built-in command prompt, PowerShell, or Windows Terminal and run the above.
`Docker Desktop <https://docs.docker.com/docker-for-windows/install/>`_ is the only option at present for Windows users. Nonetheless, it is recommended to use Windows Subsystem for Linux (`WSL <https://learn.microsoft.com/en-us/windows/wsl/install>`_) to install Subaligner.
