.. subaligner documentation master file, created by
   sphinx-quickstart on Wed Dec 25 23:33:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Subaligner's documentation!
======================================

Given an out-of-sync subtitle file along with a piece of audiovisual content carrying speeches described by it,
Subaligner provides a one-stop solution on automatic subtitle synchronisation with a pretrained deep neural network and forced
alignments. In essence, aligning subtitles is a two-stage process with a Bidirectional Long Short-Term Memory network trained
upfront and Subaligner helps subtitlers not only in preprocessing raw subtitle materials (outcome from stenographers or
STT workflow, etc.) but also in gaining quality control over their work within subtitle post-production. This solution
also tolerates errors occurred in live subtitles which sometimes do not completely or correctly represent what people
actually spoke in the companion audiovisual content.

The source code can be found on GitHub: `subaligner <https://github.com/baxtree/subaligner>`_.

.. toctree::
   :maxdepth: 2

   installation
   usage
   anatomy
   test
   acknowledgement
   license
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
