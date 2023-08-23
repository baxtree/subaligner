.. subaligner documentation master file, created by
   sphinx-quickstart on Wed Dec 25 23:33:25 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Subaligner's documentation!
======================================

.. image:: ../../figures/subaligner.png
   :width: 200
   :align: center
   :alt: Subaligner

Given an out-of-sync subtitle file along with a piece of audiovisual content carrying speeches described by it,
Subaligner provides a one-stop solution on automatic subtitle synchronisation and translation with pretrained deep neural networks
, forced alignments and transformers. In essence, aligning subtitles is a dual-stage process with a Bidirectional Long Short-Term Memory network trained
upfront. Subaligner helps subtitlers not only in preprocessing raw subtitle materials (outcome from stenographers or
STT workflow, etc.) but also in gaining quality control over their work within subtitle post-production. This tool
also tolerates errors that occurred in live subtitles which sometimes do not completely or correctly represent what people
actually spoke in the companion audiovisual content.

Subligner has been shipped with a command-line interface which helps users to conduct various tasks around subtitle
synchronisation and multilingual translation without writing any code. Application programming interfaces are also provided
to developers wanting to perform those tasks programmatically. Moreover, with existing audiovisual and in-sync subtitle files at
hand, advanced users can train their own synchronisers with a single command and zero setup. A handful of subtitle formats are supported
and can be converted from one to another either during synchronisation and translation or on on-demand.

Even without any subtitles available beforehand, Subaligner provides transcription by utilising SOTA Large Language
Models (LLMs). This pipeline, combined with translation, can generate near ready-to-use subtitles of increasingly higher quality in
various languages and formats which cater to your preferences, thanks to those models continually advancing over time.

Subligner supports the following subtitle formats: SubRip, TTML, WebVTT, (Advanced) SubStation Alpha, MicroDVD, MPL2, TMP,
EBU STL, SAMI, SCC and SBV. The source code can be found on GitHub: `subaligner <https://github.com/baxtree/subaligner>`_.

.. toctree::
   :maxdepth: 2

   installation
   usage
   advanced_usage
   anatomy
   test
   acknowledgement
   license
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
