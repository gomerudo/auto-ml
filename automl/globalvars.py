"""Here we expose shared variables to use inside the package."""
import os
import sys

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..')
ROOT_DIR = os.path.abspath(ROOT_DIR)
sys.path.append(ROOT_DIR)
"""The root directory of the project"""

LOGGER_IDENTIFIER = "automl-logger"
"""Identifier for the logging messages coming from any model in the package."""
