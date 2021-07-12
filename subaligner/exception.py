class UnsupportedFormatException(Exception):
    """ An exception raised due to unsupported formats."""


class TerminalException(Exception):
    """ An exception raised due to unrecoverable failures."""


class NoFrameRateException(Exception):
    """ An exception raised due to frame rate not found."""
