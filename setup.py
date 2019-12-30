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

with open("requirements-dev.txt") as requirements_dev_file:
    requirements_dev = requirements_dev_file.read().splitlines()[::-1]

setup(name="subaligner",
      version=__version__,
      author="Xi Bai",
      author_email="xi.bai.ed@gmail.com",
      classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
      ],
      license="MIT",
      url="git@github.com:baxtree/subaligner.git",
      description="Automatically aligns an out-of-sync subtitle file to its companion video/audio using Deep Neural Network and Forced Alignment.",
      long_description=readme + "\n\n",
      python_requires=">=3.4",
      package_dir={"subaligner": "subaligner"},
      packages=[
          "subaligner",
          "subaligner.models.training.model",
          "subaligner.models.training.weights",
      ],
      package_data={
          "subaligner.models.training.model": ["model.hdf5"],
          "subaligner.models.training.weights": ["weights.hdf5"]
      },
      install_requires=requirements,
      test_suite="tests.subaligner",
      tests_require=requirements_dev,
      setup_requires=["numpy>=1.14.1,<1.18.0"],
      scripts=["bin/subaligner_1pass", "bin/subaligner_2pass"],
)