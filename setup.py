#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from setuptools import setup, find_packages

with open(os.path.join("subaligner", "_version.py")) as f:
    exec(f.read())

with open("README.md") as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().splitlines()

with open('requirements-dev.txt') as requirements_dev_file:
    requirements_dev = requirements_dev_file.read().splitlines()

setup(name="subaligner",
      version=__version__,
      author="Xi Bai",
      author_email="xi.bai.ed@gmail.com",
      classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
      ],
      license="MIT",
      url="git@bitbucket.org:baxtree/subaligner-cli.git",
      description="Automatically align an out-of-sync subtitle to its video with DNN",
      long_description=readme + '\n\n',
      python_requires='>=3.4',
      package_dir={"subaligner": "subaligner"},
      packages=find_packages("subaligner"),
      package_data={
          "subaligner.models.training.model": ["model.hdf5"],
          "subaligner.models.training.weights": ["weights.hdf5"]
      },
      setup_requires=["numpy"],
      install_requires=requirements,
      test_suite="tests.subaligner",
      tests_require=requirements_dev
)