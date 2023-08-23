import os
import warnings
import logging
import multiprocessing as mp
from ._version import __version__

__all__ = ["__version__"]

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

mp.set_start_method("spawn", force=True)
os.environ["KMP_WARNINGS"] = "0"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
logging.getLogger("tensorflow").disabled = True
