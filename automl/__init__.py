"""Automated Machine Learning tool for Python.

This package has been developed by students of the Eindhoven University of
Technology (TU/e) for Achmea's internal use. It is intended to assist the Data
Scientists to solve classification (and possibly regression) problems by
automatically finding pipelines that include pre-processing, feature
engineering and classification/regression models to solve a dataset.
"""

import logging
from .globalvars import LOGGER_IDENTIFIER


def automl_log(message=None, level=logging.DEBUG):
    """Print a generic AutoML log messages, based on the level.

    Args:
        level (str): The log level as in python's `logging` package.

    """
    if message is None:
        return  # Do nothing

    log_message = "{logid} : {message}".format(logid=LOGGER_IDENTIFIER,
                                               message=message)
    if level == logging.DEBUG or level == 'DEBUG':
        logging.debug(log_message)
        return

    if level == logging.INFO or level == 'INFO':
        logging.info(log_message)
        return

    if level == logging.WARNING or level == 'WARNING':
        logging.warning(log_message)
        return

    if level == logging.ERROR or level == 'ERROR':
        logging.error(log_message)
        return

    if level == logging.CRITICAL or level == 'CRITICAL':
        logging.critical(log_message)
        return

    raise ValueError("Invalid log level")
