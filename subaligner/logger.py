import logging
import multiprocessing
from typing import Dict
from absl import logging as absl_logging
from .singleton import Singleton
absl_logging._warn_preinit_stderr = 0


class Logger(metaclass=Singleton):
    """Common logging."""

    VERBOSE = False
    QUIET = True

    def __init__(self, output_log: str = "output.log") -> None:
        self.__loggers: Dict[str, logging.Logger] = {}
        self.__output_log = output_log

    def get_logger(self, name: str) -> logging.Logger:

        if self.__loggers.get(name) is not None:
            return self.__loggers.get(name)  # type: ignore
        else:
            if Logger.VERBOSE:
                logger = multiprocessing.get_logger()
                logger.setLevel(logging.DEBUG)
            else:
                logger = logging.getLogger(name)
                logger.setLevel(logging.INFO)
            if Logger.QUIET:
                logger.setLevel(logging.ERROR)
            formatter = logging.Formatter(
                "%(logger_name)s - %(levelname)s - %(threadName)-9s - %(message)s"
            )

            file_handler = logging.FileHandler(self.__output_log, "w+")
            logger.propagate = False
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            if not logger.handlers:
                logger.addHandler(file_handler)
                logger.addHandler(console_handler)

            logger.addFilter(_Filter(name))

            self.__loggers[name] = logger
            return logger


class _Filter(logging.Filter):

    def __init__(self, logger_name):
        self.logger_name = logger_name

    def filter(self, record: logging.LogRecord) -> bool:
        record.logger_name = self.logger_name   # type: ignore
        return True
