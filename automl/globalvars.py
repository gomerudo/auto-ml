"""Here we expose shared variables to use inside the package."""

LOGGER_IDENTIFIER = "automl"
"""Identifier for the logging messages coming from any model in the package."""

# TODO: Move these to the relevant package
CONFIGURATIONS_CSV_NAME = "configurations.csv"
"""Name of the auto-sklearn's file storing the discoverd configurations for a
given metric."""

ALGORUNS_CSV_NAME = "algorithm_runs.arff"
"""Name of the auto-sklearn's file storing the relations dataset-configuration
for a given metric."""
