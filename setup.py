#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import setup

with open(os.path.join(os.getcwd(), "subaligner", "_version.py")) as f:
    exec(f.read())

with open("README.md") as readme_file:
    readme = readme_file.read()

if "STRETCH_OFF" not in os.environ:
    with open("requirements.txt") as requirements_file:
        requirements = requirements_file.read().splitlines()[::-1]
else:
    with open("requirements-app.txt") as requirements_file:
        requirements = requirements_file.read().splitlines()[::-1]

setup(name="subaligner",
      version=__version__,
      author="Xi Bai",
      author_email="xi.bai.ed@gmail.com",
      classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
      ],
      license="MIT",
      url="https://subaligner.readthedocs.io/en/latest/",
      description="Automatically synchronise subtitles to companion audiovisual content with Deep Neural Network and Forced Alignment.",
      long_description=readme + "\n\n",
      long_description_content_type='text/markdown',
      python_requires=">=3.6",
      package_dir={"subaligner": "subaligner"},
      packages=[
          "subaligner",
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
      setup_requires=["numpy>=1.14.1,<1.18.0"],
      scripts=["bin/subaligner_1pass", "bin/subaligner_2pass", "bin/subaligner"],
)