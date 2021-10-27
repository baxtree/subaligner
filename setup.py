#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import setup

with open(os.path.join(os.getcwd(), "subaligner", "_version.py")) as f:
    exec(f.read())

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as requirements_file:
    requirements = requirements_file.read().splitlines()[::-1]

with open("requirements-stretch.txt") as stretch_requirements_file:
    stretch_requirements = stretch_requirements_file.read().splitlines()[::-1]

with open("requirements-site.txt") as docs_requirements_file:
    docs_requirements = docs_requirements_file.read().splitlines()[::-1]

with open("requirements-translation.txt") as translate_requirements_file:
    translate_requirements = translate_requirements_file.read().splitlines()[::-1]

with open("requirements-dev.txt") as dev_requirements_file:
    dev_requirements = dev_requirements_file.read().splitlines()[::-1]

EXTRA_DEPENDENCIES = {
    "harmony": stretch_requirements + translate_requirements,
    "dev": dev_requirements + stretch_requirements + translate_requirements + docs_requirements,
    "docs": docs_requirements,
    "stretch": stretch_requirements,
    "translation": translate_requirements,
}

setup(name="subaligner",
      version=__version__,
      author="Xi Bai",
      author_email="xi.bai.ed@gmail.com",
      classifiers=[
          "License :: OSI Approved :: MIT License",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "Intended Audience :: Developers",
      ],
      license="MIT",
      url="https://subaligner.readthedocs.io/en/latest/",
      description="Automatically synchronize and translate subtitles with pretrained deep neural networks, forced alignments and transformers.",
      long_description=readme + "\n\n",
      long_description_content_type='text/markdown',
      python_requires=">=3.6",
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
          ]})
