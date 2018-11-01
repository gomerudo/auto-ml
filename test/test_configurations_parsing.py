"""Module to test the model builder."""

import unittest
import os.path
from automl.metalearning.database.configurations_parsing \
    import ConfigurationBuilder
from automl.metalearning.database.configurations_parsing import MLSuggestion
from automl.metalearning.database.load_db import ConfigurationsFile


class TestConfigurationBuilder(unittest.TestCase):
    """Test that building a configuration gives the good result."""

    def test_build_configuration_type(self):
        """Test the result is of the type MLSuggestion and attributes 'set'."""
        automl_path = os.path.dirname(os.path.dirname(__file__))
        automl_path = os.path.join(automl_path,
                                   "automl",
                                   "metalearning",
                                   "database",
                                   "files",
                                   "accuracy_multiclass.classification_dense",
                                   "configurations.csv")

        c_f = ConfigurationsFile(automl_path)
        mmb = ConfigurationBuilder(c_f.get_configuration(32))

        ml_suggestion = mmb.build_configuration()

        # Main type
        self.assertTrue(isinstance(ml_suggestion, MLSuggestion))

        # Set types
        self.assertTrue(isinstance(ml_suggestion.encoders, set))
        self.assertTrue(isinstance(ml_suggestion.classifiers, set))
        self.assertTrue(isinstance(ml_suggestion.imputations, set))
        self.assertTrue(isinstance(ml_suggestion.preprocessors, set))
        self.assertTrue(isinstance(ml_suggestion.rescalers, set))

    def test_build_configuration_behaviour(self):
        """Test the result is as expected, for a given metric."""
        automl_path = os.path.dirname(os.path.dirname(__file__))
        automl_path = os.path.join(automl_path,
                                   "automl",
                                   "metalearning",
                                   "database",
                                   "files",
                                   "accuracy_multiclass.classification_dense",
                                   "configurations.csv")

        expected_encoders = set(["sklearn.preprocessing.OneHotEncoder"])
        expected_classifier = \
            set(["sklearn.ensemble.GradientBoostingClassifier"])
        expected_imputation = set(["sklearn.impute.SimpleImputer"])
        expected_processor = set()
        expected_escaler = set(["sklearn.preprocessing.StandardScaler"])

        c_f = ConfigurationsFile(automl_path)
        mmb = ConfigurationBuilder(c_f.get_configuration(32))

        ml_suggestion = mmb.build_configuration()

        self.assertTrue(ml_suggestion.encoders == expected_encoders)
        self.assertTrue(ml_suggestion.classifiers == expected_classifier)
        self.assertTrue(ml_suggestion.imputations == expected_imputation)
        self.assertTrue(ml_suggestion.preprocessors == expected_processor)
        self.assertTrue(ml_suggestion.rescalers == expected_escaler)

    # TODO: Test MLSuggestion operators

    # TODO: Test mix_suggestions

if __name__ == '__main__':
    unittest.main()
