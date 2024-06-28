# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys


sys.path.insert(0, os.path.abspath("../.."))
with open(os.path.join(os.getcwd(), "..", "..", "subaligner", "_version.py")) as f:
    exec(f.read())

# -- Project information -----------------------------------------------------

project = "subaligner"
copyright = "2019-present, Xi Bai"
author = "Xi Bai"
master_doc = "index"

# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "**lib**",
    "**subaligner_1pass**",
    "**subaligner_2pass**",
    "**subaligner_batch**",
    "**subaligner_convert**",
    "**subaligner_train**",
    "**subaligner_tune**",
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

autodoc_mock_imports = [
    "absl",
    "absl-py",
    "aeneas",
    "h5py",
    "hyperopt",
    "librosa",
    "numpy",
    "psutil",
    "pycaption",
    "pysrt",
    "sklearn",
    "tensorflow",
    "pysubs2",
    "chardet",
    "captionstransformer",
    "bs4",
    "transformers",
    "pycountry",
    "tqdm",
    "whisper"
]

def setup(app):
    if os.getenv("READTHEDOCS", False):
        def run_apidoc(_):
            from sphinx.ext.apidoc import main as apidoc_main
            cur_dir = os.path.abspath(os.path.dirname(__file__))
            included_module = "../../subaligner"
            excluded_module = "../../subaligner/models"
            apidoc_main(["-e", "-o", cur_dir, included_module, excluded_module, "--force"])
        app.connect("builder-inited", run_apidoc)
    else:
        pass
