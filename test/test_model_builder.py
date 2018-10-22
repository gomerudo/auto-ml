"""Module to test the model builder."""

import unittest
import os.path
from automl.metalearning.db.configurations_parsing import ConfigurationBuilder
from automl.metalearning.db.load_db import ConfigurationsFile


class TestMetaModelBuilder(unittest.TestCase):
    """Test the SearchSpaceBuilder features."""

    def test_build_suggestion(self):
        """Test the building of a ML Suggestion from the Meta-Knowledge."""
        automl_path = os.path.dirname(os.path.dirname(__file__))
        automl_path = os.path.join(automl_path,
                                   "automl",
                                   "metalearning",
                                   "db",
                                   "files",
                                   "accuracy_multiclass.classification_dense",
                                   "configurations.csv")

        c_f = ConfigurationsFile(automl_path)
        mmb = ConfigurationBuilder(c_f.get_configuration(32))
        ml_suggestion = mmb.build_configuration()

        print("Summary: ")
        print("Imputations: ", ml_suggestion.imputations)
        print("Encoders: ", ml_suggestion.encoders)
        print("Classifiers: ", ml_suggestion.classifiers)
        print("Rescalers: ", ml_suggestion.rescalers)
        print("Preprocessors: ", ml_suggestion.preprocessors)
