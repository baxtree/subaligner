import os
import warnings
import multiprocessing as mp
from ._version import __version__

__all__ = ["__version__"]

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

mp.set_start_method("spawn", force=True)
os.environ["KMP_WARNINGS"] = "0"
