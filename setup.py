#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from platform import architecture, machine
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel

with open(os.path.join(os.getcwd(), "subaligner", "_version.py")) as f:
    exec(f.read())

with open("README.md") as readme_file:
    readme = readme_file.read()

if sys.platform == "darwin" and machine() == "arm64":
    with open("requirements-arm64.txt") as requirements_file:
        requirements = requirements_file.read().splitlines()[::-1]
else:
    with open("requirements.txt") as requirements_file:
        requirements = requirements_file.read().splitlines()[::-1]

with open("requirements-stretch.txt") as stretch_requirements_file:
    stretch_requirements = stretch_requirements_file.read().splitlines()[::-1]

with open("requirements-site.txt") as docs_requirements_file:
    docs_requirements = docs_requirements_file.read().splitlines()[::-1]

with open("requirements-llm.txt") as llm_requirements_file:
    llm_requirements = llm_requirements_file.read().splitlines()[::-1]

with open("requirements-dev.txt") as dev_requirements_file:
    dev_requirements = dev_requirements_file.read().splitlines()[::-1]

EXTRA_DEPENDENCIES = {
    "harmony": stretch_requirements + llm_requirements,
    "dev": dev_requirements + stretch_requirements + llm_requirements + docs_requirements,
    "docs": docs_requirements,
    "stretch": stretch_requirements,
    "translation": llm_requirements,    # for backward compatibility and now deprecated with "llm"
    "llm": llm_requirements,
}

architecture = architecture()[0] if sys.platform == "win32" else machine()


class bdist_wheel_local(bdist_wheel):

    def get_tag(self):
        python = f"py{sys.version_info.major}{sys.version_info.minor}"
        if sys.platform == "darwin" and architecture == "arm64":
            os_arch = "macosx_11_0_arm64"
        else:
            os_arch = "any"
        return python, "none", os_arch


setup(name="subaligner",
      version=__version__,
      author="Xi Bai",
      author_email="xi.bai.ed@gmail.com",
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          "Intended Audience :: Developers",
          "Topic :: Utilities",
      ],
      license="MIT",
      url="https://github.com/baxtree/subaligner",
      description="Automatically synchronize and translate subtitles, or create new ones by transcribing, using pre-trained DNNs, Forced Alignments and Transformers.",
      long_description=readme + "\n\n",
      long_description_content_type="text/markdown",
      project_urls={
          "Documentation": "https://subaligner.readthedocs.io/en/latest/",
          "Source": "https://github.com/baxtree/subaligner",
      },
      python_requires=">=3.8,<3.12",
      wheel=True,
      package_dir={"subaligner": "subaligner"},
      packages=[
          "subaligner",
          "subaligner.lib",
          "subaligner.subaligner_1pass",
          "subaligner.subaligner_2pass",
          "subaligner.subaligner_batch",
          "subaligner.subaligner_convert",
          "subaligner.subaligner_train",
          "subaligner.subaligner_tune",
          "subaligner.models.training.model",
          "subaligner.models.training.weights",
          "subaligner.models.training.config",
      ],
      package_data={
          "subaligner.models.training.model": ["model.hdf5"],
          "subaligner.models.training.weights": ["weights.hdf5"],
          "subaligner.models.training.config": ["hyperparameters.json"],
      },
      install_requires=requirements,
      test_suite="tests.subaligner",
      extras_require=EXTRA_DEPENDENCIES,
      scripts=[
          "bin/subaligner",
          "bin/subaligner_1pass",
          "bin/subaligner_2pass",
          "bin/subaligner_batch",
          "bin/subaligner_convert",
          "bin/subaligner_train",
          "bin/subaligner_tune",
      ],
      entry_points={
          "console_scripts": [
              "subaligner=subaligner.__main__:main",
              "subaligner_1pass=subaligner.subaligner_1pass.__main__:main",
              "subaligner_2pass=subaligner.subaligner_2pass.__main__:main",
              "subaligner_batch=subaligner.subaligner_batch.__main__:main",
              "subaligner_convert=subaligner.subaligner_convert.__main__:main",
              "subaligner_train=subaligner.subaligner_train.__main__:main",
              "subaligner_tune=subaligner.subaligner_tune.__main__:main",
          ]
      },
      cmdclass={"bdist_wheel": bdist_wheel_local})
