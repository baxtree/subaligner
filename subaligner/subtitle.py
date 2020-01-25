import pysrt
import tempfile
import os
import re
import xml.etree.ElementTree as ElementTree
from pysrt import SubRipFile
from copy import deepcopy
from .utils import Utils
from .exception import UnsupportedFormatException


class Subtitle(object):
    """Load a subtitle file into internal data structure
    """

    __secret = object()

    ElementTree.register_namespace("", "http://www.w3.org/ns/ttml")
    ElementTree.register_namespace("tts", "http://www.w3.org/ns/ttml#styling")
    ElementTree.register_namespace("ttm", "http://www.w3.org/ns/ttml#metadata")
    ElementTree.register_namespace("ttp", "http://www.w3.org/ns/ttml#parameter")
    ElementTree.register_namespace("ebuttm", "urn:ebu:tt:metadata")
    ElementTree.register_namespace("ebutts", "urn:ebu:tt:style")
    TT_NS = {"tt": "http://www.w3.org/ns/ttml"}

    SUBRIP_EXTENTIONS = [".srt"]
    TTML_EXTENSIONS = [".xml", ".ttml", ".dfxp"]
    WEBVTT_EXTENSIONS = [".vtt"]

    def __init__(self, secret, subtitle_file_path, subtitle_format):
        """Subtitle object initialiser.

        Arguments:
            secret {object} -- A hash only known by factory methods.
            subtitle_file_path {string} -- The path to the subtitle file.
            format {string} -- Supported subtitle formats: subrip and ttml.

        Raises:
            NotImplementedError --  Thrown when any subtitle attributes are modified.
        """

        assert (
            secret == Subtitle.__secret
        ), "Only factory methods are supported when creating instances"

        self.__subtitle_file_path = subtitle_file_path

        if subtitle_format == "subrip":
            self.__subs = pysrt.open(subtitle_file_path, encoding="utf-8")
        elif subtitle_format == "ttml":
            self.__subs = self.__convert_ttml_to_subs(subtitle_file_path)
        elif subtitle_format == "webvtt":
            self.__subs = self.__convert_vtt_to_subs(subtitle_file_path)
        else:
            raise UnsupportedFormatException(
                "Unknown subtitle format for file: {}".format(subtitle_file_path)
            )

        # freeze the object after creation
        def __setattr__(self, *args):
            raise NotImplementedError("Cannot modify the immutable object")

        def __delattr__(self, *args):
            raise NotImplementedError("Cannot modify the immutable object")

    @classmethod
    def load_subrip(cls, subtitle_file_path):
        """Load a SubRip subtitle file.

        Arguments:
            subtitle_file_path {string} -- The path to the subtitle file.

        Returns:
            Subtitle -- Subtitle object.
        """

        return cls(cls.__secret, subtitle_file_path, "subrip")

    @classmethod
    def load_ttml(cls, subtitle_file_path):
        """Load a TTML subtitle file.

        Arguments:
            subtitle_file_path {string} -- The path to the subtitle file.

        Returns:
            Subtitle -- Subtitle object.
        """

        return cls(cls.__secret, subtitle_file_path, "ttml")

    @classmethod
    def load_webvtt(cls, subtitle_file_path):
        """Load a WebVTT subtitle file.

        Arguments:
            subtitle_file_path {string} -- The path to the subtitle file.

        Returns:
            Subtitle -- Subtitle object.
        """

        return cls(cls.__secret, subtitle_file_path, "webvtt")

    @classmethod
    def load(cls, subtitle_file_path):
        """Load a SubRip or TTML subtitle file based on the file extension.

        Arguments:
            subtitle_file_path {string} -- The path to the subtitle file.

        Returns:
            Subtitle -- Subtitle object.
        """

        filename, file_extension = os.path.splitext(subtitle_file_path.lower())
        if file_extension in cls.SUBRIP_EXTENTIONS:
            return cls(cls.__secret, subtitle_file_path, "subrip")
        elif file_extension in cls.TTML_EXTENSIONS:
            return cls(cls.__secret, subtitle_file_path, "ttml")
        elif file_extension in cls.WEBVTT_EXTENSIONS:
            return cls(cls.__secret, subtitle_file_path, "webvtt")
        else:
            return cls(cls.__secret, subtitle_file_path, "unknown")

    @classmethod
    def shift_subtitle(
        cls,
        subtitle_file_path,
        seconds,
        shifted_subtitle_file_path=None,
        suffix="_shifted",
    ):
        """Shift subtitle cues based on the input seconds.

        Arguments:
            subtitle_file_path {string} -- The path to the subtitle file.
            seconds {float} -- The number of seconds by which the cues are shifted.

        Keyword Arguments:
            shifted_subtitle_file_path {string} -- The path to the shifted subtitle file (default: {None}).
            suffix {string} -- The suffix used as part of the aligned subtitle file name.

        Returns:
            string -- The path to the shifted subtitle file.
        """
        filename, file_extension = os.path.splitext(subtitle_file_path)
        if file_extension.lower() in cls.SUBRIP_EXTENTIONS:
            subs = cls(cls.__secret, subtitle_file_path, "subrip").subs
            subs.shift(seconds=seconds)
            if shifted_subtitle_file_path is None:
                shifted_subtitle_file_path = subtitle_file_path.replace(
                    file_extension, "{}{}".format(suffix, file_extension)
                )
            Subtitle.export_subtitle(subtitle_file_path, subs, shifted_subtitle_file_path)
            return shifted_subtitle_file_path
        elif file_extension.lower() in cls.TTML_EXTENSIONS:
            subs = cls(cls.__secret, subtitle_file_path, "ttml").subs
            subs.shift(seconds=seconds)
            tree = ElementTree.parse(subtitle_file_path)
            tt = tree.getroot()
            cues = (
                tt.find("tt:body", Subtitle.TT_NS)
                .find("tt:div", Subtitle.TT_NS)
                .findall("tt:p", Subtitle.TT_NS)
            )
            for index, cue in enumerate(cues):
                cue.attrib["begin"] = subs[index].start
                cue.attrib["end"] = subs[index].end
            if shifted_subtitle_file_path is None:
                shifted_subtitle_file_path = subtitle_file_path.replace(
                    file_extension, "{}{}".format(suffix, file_extension)
                )
            tree.write(shifted_subtitle_file_path, encoding="utf8")
            return shifted_subtitle_file_path
        elif file_extension.lower() in cls.WEBVTT_EXTENSIONS:
            subs = cls(cls.__secret, subtitle_file_path, "webvtt").subs
            subs.shift(seconds=seconds)
            if shifted_subtitle_file_path is None:
                shifted_subtitle_file_path = subtitle_file_path.replace(
                    file_extension, "{}{}".format(suffix, file_extension)
                )
            Subtitle.export_subtitle(subtitle_file_path, subs, shifted_subtitle_file_path)
            return shifted_subtitle_file_path
        else:
            raise UnsupportedFormatException(
                "Unknown subtitle format for file: {}".format(subtitle_file_path)
            )

    @staticmethod
    def export_subtitle(source_file_path, subs, target_file_path):
        """Export subtitle in the format determined by the file extension.

        Arguments:
            source_file_path {string} -- The path to the original subtitle file.
            subs {list} -- A list of SubRipItems.
            target_file_path {string} -- The path to the exported subtitle file.
        """

        filename, file_extension = os.path.splitext(source_file_path.lower())
        if file_extension in Subtitle.SUBRIP_EXTENTIONS:
            SubRipFile(subs).save(target_file_path, encoding="utf8")
        elif file_extension in Subtitle.TTML_EXTENSIONS:
            tree = ElementTree.parse(source_file_path)
            tt = tree.getroot()
            cues = (
                tt.find("tt:body", Subtitle.TT_NS)
                .find("tt:div", Subtitle.TT_NS)
                .findall("tt:p", Subtitle.TT_NS)
            )
            for index, cue in enumerate(cues):
                cue.attrib["begin"] = str(subs[index].start).replace(",", ".")
                cue.attrib["end"] = str(subs[index].end).replace(",", ".")

            # Change single quotes in the XML header to double quotes
            with open(target_file_path, "w") as target:
                normalised = (
                    ElementTree.tostring(tt, encoding="utf8", method="xml")
                    .decode("utf-8")
                    .replace(
                        "<?xml version='1.0' encoding='utf8'?>",
                        '<?xml version="1.0" encoding="utf8"?>',
                    )
                )
                target.write(normalised)
        elif file_extension in Subtitle.WEBVTT_EXTENSIONS:
            try:
                _, path = tempfile.mkstemp()
                SubRipFile(subs).save(path, encoding="utf8")
                Utils.srt2vtt(path, target_file_path)
            finally:
                os.remove(path)
        else:
            raise UnsupportedFormatException(
                "Unknown subtitle format for file: {}".format(source_file_path)
            )

    @staticmethod
    def remove_sound_effects_by_case(subs, se_uppercase=True):
        """Remove subtitles of sound effects based on case

        Arguments:
            subs {list} -- A list of SubRipItems.
            se_uppercase {bool} -- True when the sound effect is in uppercase or False when in lowercase (default: {True}).

        Returns:
            {list} -- A list of SubRipItems.
        """
        new_subs = deepcopy(subs)
        for sub in subs:
            if se_uppercase is not None:
                if se_uppercase and sub.text.isupper():
                    new_subs.remove(sub)
                elif not se_uppercase and sub.text.islower():
                    new_subs.remove(sub)
        return new_subs

    @staticmethod
    def remove_sound_effects_by_affixes(subs, se_prefix, se_suffix=None):
        """Remove subtitles of sound effects based on prefix or prefix and suffix

        Arguments:
            subs {list} -- A list of SubRipItems.
            se_prefix {string} -- A prefix indicating the start of the sound effect.
            se_suffix {string} -- A suffix indicating the end of the sound effect (default: {None}).

        Returns:
            {list} -- A list of SubRipItems.
        """
        new_subs = deepcopy(subs)
        for sub in subs:
            if se_suffix is not None:
                match = re.search(
                    "^{0}[^{0}{1}]+{1}$".format(
                        re.escape(se_prefix), re.escape(se_suffix)
                    ),
                    sub.text,
                )
            else:
                match = re.search("^{0}[^{0}]+$".format(re.escape(se_prefix)), sub.text)
            if match:
                new_subs.remove(sub)
        return new_subs

    @staticmethod
    def extract_text(subtitle_file_path, delimiter=" "):
        """Extract plain texts from a subtitle file.

        Arguments:
            subtitle_file_path {string} -- The path to the subtitle file.

        Returns:
            {string} -- The plain text of subtitle.
        """

        subs = Subtitle.load(subtitle_file_path).subs
        texts = [sub.text for sub in subs]
        return delimiter.join(texts)

    @property
    def subtitle_file_path(self):
        return self.__subtitle_file_path

    @property
    def subs(self):
        return self.__subs

    @staticmethod
    def __convert_ttml_to_subs(ttml_file_path):
        """Convert a subtitle file from the TTML format to the SubRip format

        Arguments:
            ttml_file_path {string} -- The path to the TTML subtitle file.

        Returns:
            {list} -- A list of SubRipItems.
        """

        _, path = tempfile.mkstemp()
        Utils.ttml2srt(ttml_file_path, path)

        try:
            subs = pysrt.open(path, encoding="utf-8")
        finally:
            os.remove(path)
        return subs

    @staticmethod
    def __convert_vtt_to_subs(vtt_file_path):
        """Convert a subtitle file from the WebVTT format to the SubRip format

        Arguments:
            vtt_file_path {string} -- The path to the WebVTT subtitle file.

        Returns:
            {list} -- A list of SubRipItems.
        """

        _, path = tempfile.mkstemp()
        Utils.vtt2srt(vtt_file_path, path)

        try:
            subs = pysrt.open(path, encoding="utf-8")
        finally:
            os.remove(path)
        return subs
