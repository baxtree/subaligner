import logging
from absl import logging as absl_logging
from .singleton import Singleton
absl_logging._warn_preinit_stderr = 0


class Logger(Singleton):
    """Common logging."""

    VERBOSE = True

    def __init__(self, output_log="output.log"):
        self.__loggers = {}
        self.__output_log = output_log

    def get_logger(self, name):

        if self.__loggers.get(name):
            return self.__loggers.get(name)
        else:
            logger = logging.getLogger(name)
            logger.setLevel(logging.DEBUG if Logger.VERBOSE else logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            file_handler = logging.FileHandler(self.__output_log, "w+")
            logger.propagate = False
            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            if not logger.handlers:
                logger.addHandler(file_handler)
                logger.addHandler(console_handler)

            self.__loggers[name] = logger
            return logger
