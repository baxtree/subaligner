import os
import warnings
import logging
import multiprocessing as mp
import tensorflow as tf
from ._version import __version__

__all__ = ["__version__"]

warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

mp.set_start_method("spawn", force=True)
os.environ["KMP_WARNINGS"] = "0"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
logging.getLogger("tensorflow").disabled = True

if tf.__version__ >= "2.16.0":
    os.environ["TF_USE_LEGACY_KERAS"] = "1"
    try:
        import tf_keras
        import tensorflow._api.v2.compat.v2.__internal__ as tf_internal
    except ImportError:
        raise RuntimeError(f"Tensorflow {tf.__version__} was installed but not supported by subaligner.")
    if hasattr(tf_internal, 'register_call_context_function'):
        tf_internal.register_load_context_function = tf_internal.register_call_context_function
