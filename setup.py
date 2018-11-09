"""The setup.py file for auto-ml package."""

import os
from setuptools import setup


setup(
    name="auto-ml",
    version="0.0.1",
    author="Jorge Gomez Robles; Firoz Ansari",
    author_email="j.gomez.robles@student.tue.nl",
    description=("A package to solve classification and regression problems \
using AutoML approaches."),
    keywords="automl machine learning",
    url="https://gomerudo.github.io/auto-ml/",
    packages=['automl'],
)
