########################
Installation
########################

**Install necessary dependencies**::

    $ apt-get install ffmpeg
    $ apt-get install espeak libespeak1 libespeak-dev espeak-data

**or**::

    $ brew install ffmpeg espeak

§ You may also need to install `HomeBrew <https://brew.sh/>`_.

**Install Subaligner via PyPI (pre-emptive NumPy)**::

    $ pip install -U pip
    $ pip install numpy
    $ pip install subaligner

**Install Subaligner via pipx**::

    $ pipx install subaligner

**Install from GitHub via Pipenv**::

    ...
    [packages]
    numpy = "*"
    subaligner = {git = "ssh://git@github.com/baxtree/subaligner.git", ref = "<TAG>"}
    ...

**Use dockerised installation**::

    $ docker run -v `pwd`:`pwd` -w `pwd` -it baxtree/subaligner bash

The following builds are available on dockerhub for several Linux distributions: CentOS 7 (latest and VERSION.el7), CentOS 8 (VERSION.el8), Ubuntu 18 (VERSION.u18), Ubuntu 20 (VERSION.u20), Debian 10 (VERSION.deb10), Fedora 31 (VERSION.fed31) and ArchLinux (VERSION.arch).

You can also download the latest
release on `GitHub <https://github.com/baxtree/subaligner>`_ and follow the steps down below
to create a virtual environment and set up all the dependencies:

**Install Subaligner from source**::

    $ git clone git@github.com:baxtree/subaligner.git
    $ cd subaligner
    $ make install && source .venv/bin/activate

**Subaligner CLI should be on your PATH now**::

    (.venv) $ subaligner --help
    (.venv) $ subaligner_1pass --help
    (.venv) $ subaligner_2pass --help
    (.venv) $ subaligner_train --help
    (.venv) $ subaligner_tune --help

**On Windows**::

    docker pull baxtree/subaligner
    docker run -v "/d/media":/media -w "/media" -it baxtree/subaligner bash

Assume your media assets are stored under "d:\\media". Open built-in command prompt, PowerShell, or Windows Terminal and run the above.
`Docker Desktop <https://docs.docker.com/docker-for-windows/install/>`_ is the only option at the present for Windows users.
