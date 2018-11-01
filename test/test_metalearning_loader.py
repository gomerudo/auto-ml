"""Module to test methods related to loading the Meta Knowledge."""

import unittest
import os.path
import math
import pandas as pd
from automl.metalearning.database.load_db import MetaKnowledge
from automl.metalearning.database.load_db import LandmarkModelParser
from automl.metalearning.database.load_db import ConfigurationsFile
from automl.metalearning.database.load_db import AlgorithmRunsFile
from automl.metalearning.database.load_db import MKDatabaseClient
from automl.metalearning.database.configurations_parsing import MLSuggestion
from automl.datahandler.dataloader import DataLoader


class TestMetaKnowledge(unittest.TestCase):
    """Check that the loading of the MetaKnowledge is correct."""

    def test_load_db_init(self):
        """Test initialization of the LoadMetaDB object.

        By default, it needs to assign None references to the costs/features
        objects.
        """
        lmdb = MetaKnowledge()

        # We assert everything is None
        self.assertIsNone(lmdb.costs,
                          "List for costs should be empty while initializing")
        self.assertIsNone(lmdb.features,
                          "List for features should be empty while \
                          initializing")

    def test_shape_of_files(self):
        """Test both feature values and costs are sorted and share shapes."""
        lmdb = MetaKnowledge().load_datasets_info()

        # Check they have the same length
        self.assertTrue(lmdb.costs.shape[0] == lmdb.features.shape[0])

        # Check they are correctly sorted
        features_ids = list(lmdb.features.values_by_attribute('instance_id'))
        features_ids.reverse()
        a = features_ids.pop()  # pylint: disable=C0103
        while features_ids:
            b = features_ids.pop()  # pylint: disable=C0103
            self.assertTrue(a < b,
                            "The features list is not sorted: Relation \
                            {a} < {b} not satisfied".format(a=a, b=b))
            # update
            a = b  # pylint: disable=C0103

        costs_ids = list(lmdb.costs.values_by_attribute('instance_id'))
        costs_ids.reverse()
        a = costs_ids.pop()  # pylint: disable=C0103
        while costs_ids:
            b = costs_ids.pop()  # pylint: disable=C0103
            self.assertTrue(a < b,
                            "The costs list is not sorted: Relation a > b \
                            not satisfied")
            # update
            a = b  # pylint: disable=C0103

        # Finally, test that they are candidates for elementwise multiplication
        self.assertTrue(lmdb.costs.shape == lmdb.features.shape)

    def test_feature_files_structure(self):
        """Test that feature_values/feature_costs files use the same subjects.

        For a proper usage of these files, we only accept the case when both
        use exactly the same IDs - i.e., if we create two lists with each's IDs
        then: A - B = B - A = empty set.
        """
        lmdb = MetaKnowledge().load_datasets_info()

        # Check they are not None after initialization
        self.assertIsNotNone(lmdb.features)
        self.assertIsNotNone(lmdb.costs)

        # And then check the IDs ...

        # ... by creating a new list with the IDs of one list ...
        check_list = list(lmdb.costs.values_by_attribute('instance_id'))
        # for element in lmdb.costs.values_by_attribute('instance_id'):
        #     check_list.append(element)

        # ... and then for the other's IDs, remove the elements in check_list
        for element in lmdb.features.values_by_attribute('instance_id'):
            # If one element is not there. Then B - A != 0
            self.assertTrue(element in check_list,
                            "Element {id} is not in both costs and features \
                            list".format(id=element))
            check_list.remove(element)

        # If we come here, then B - A = 0. Then the only check left is
        # if A - B = 0.
        self.assertListEqual(check_list, [],
                             "Features/costs lists do not match IDs")

    def test_weighted_simple_vectors_agree(self):
        """Check that what is obtained from the multiplication is correct.

        In order to compute the weighted matrix, a multiplication is done. We
        want to test that the resulting multiplied matrix is indeed what it is
        expected.
        """
        lmdb = MetaKnowledge().load_datasets_info()
        w_ids, weighted = lmdb.weighted_matrix()
        s_ids, simple = lmdb.simple_matrix()

        costs = lmdb.costs.sort_attributes().as_numpy_array()

        # Test shapes
        self.assertTrue(weighted.shape == simple.shape)
        self.assertTrue(
            costs.shape[0] == simple.shape[0] and
            costs.shape[1] - 1 == simple.shape[1]
        )

        # Assert ids are indeed correct
        costs_ids = lmdb.costs.values_by_attribute('instance_id')
        for orig, weigth, sim in zip(costs_ids, w_ids, s_ids):
            self.assertTrue(orig == weigth and weigth == sim)

        # Assert the weighted is indeed correct. We test one by one cause it is
        # indeed a test :)
        for row_w, row_s, row_c in zip(weighted, simple, costs):
            for w, s, c in zip(row_w, row_s, row_c):  # pylint: disable=C0103
                if math.isnan(c) or math.isnan(s):
                    self.assertTrue(math.isnan(w))
                else:
                    expected = round(w, 5)
                    obtained = round(s*c, 5)
                    self.assertTrue(expected == obtained,
                                    "{} != {}".format(expected, obtained))


class TestLandmarkModelParser(unittest.TestCase):
    """Test the parsing of the landmark models by auto-sklearn."""

    def test_metrics_available(self):
        """Test that the metrics available are loaded correctly."""
        # Check not empty
        self.assertTrue(LandmarkModelParser.metrics_available())

    def test_non_supported_metric(self):
        """Test that method error raises if no valid metric is passed."""
        with self.assertRaises(ValueError):
            LandmarkModelParser.models_by_metric('achmea_metric')

    def test_model_by_metric_error(self):
        """Test the behaviour of the model_by_metric method for ValueError."""
        dataset = DataLoader.get_openml_dataset(openml_id=46, problem_type=0)
        instances_ids = [1]
        metric = 'accuracy'

        with self.assertRaises(ValueError):
            LandmarkModelParser.models_by_metric(instances_ids,
                                                 dataset,
                                                 metric)

    def test_model_by_metric_all_instances_all_metrics(self):
        """Test the that models by metric works for every record in the DB."""
        dataset = DataLoader.get_openml_dataset(openml_id=46, problem_type=0)

        metrics = [
            'accuracy',
            'average_precision',
            'balanced_accuracy',
            'f1',
            'f1_macro',
            'f1_micro',
            'f1_multiclass',
            'f1_weighted',
            'log_loss',
            'pac_score',
            'precision',
            'precision_macro',
            'precision_micro',
            'precision_multiclass',
            'precision_weighted',
            'recall',
            'recall_macro',
            'recall_micro',
            'recall_weighted',
            'roc_auc'
        ]

        instances_ids, _ = \
            MetaKnowledge().load_datasets_info().weighted_matrix()

        for metric in metrics:
            for instance in instances_ids:
                # Do one by one for testing
                try:
                    models = LandmarkModelParser.models_by_metric([instance],
                                                                  dataset,
                                                                  metric)
                    self.assertTrue(models)
                    # Assert type of list
                    self.assertTrue(isinstance(models, list))

                    # Assert type of each of the list's elements.
                    for model in models:
                        self.assertTrue(isinstance(model, MLSuggestion))
                except ValueError:
                    pass

    def test_model_by_metric_return_type(self):
        """Test the behaviour of the model_by_metric method."""
        dataset = DataLoader.get_openml_dataset(openml_id=46, problem_type=0)
        instances_ids = [251, 75115]
        metric = 'accuracy'

        models = LandmarkModelParser.models_by_metric(instances_ids,
                                                      dataset,
                                                      metric)

        # Assert type of list
        self.assertTrue(isinstance(models, list))

        # Assert type of each of the list's elements.
        for model in models:
            self.assertTrue(isinstance(model, MLSuggestion))


class TestConfigurationsFile(unittest.TestCase):
    """Test the configurations file parsing.

    The ConfigurationsFile is intended to retrieve information from a given
    configurations.csv file. In this file we test these methods.
    """

    def test_get_configuration_wrong_file(self):
        """Test that when a inexistent file is passed, Value Error raises."""
        automl_path = os.path.dirname(os.path.dirname(__file__))
        automl_path = os.path.join(automl_path,
                                   "automl",
                                   "metalearning",
                                   "database",
                                   "",
                                   "accuracy_multiclass.classification_dense",
                                   "configurations.csv")
        with self.assertRaises(ValueError):
            ConfigurationsFile(automl_path)

    def test_get_configuration_wrong_id(self):
        """Test that when a inexistent id is passed, Value Error raises."""
        automl_path = os.path.dirname(os.path.dirname(__file__))
        automl_path = os.path.join(automl_path,
                                   "automl",
                                   "metalearning",
                                   "database",
                                   "files",
                                   "accuracy_multiclass.classification_dense",
                                   "configurations.csv")
        configs_file = ConfigurationsFile(automl_path)

        with self.assertRaises(ValueError):
            configs_file.get_configuration(100000)

    def test_get_configuration(self):
        """Test that a valid series is retrieved with a valid id and file."""
        automl_path = os.path.dirname(os.path.dirname(__file__))
        automl_path = os.path.join(automl_path,
                                   "automl",
                                   "metalearning",
                                   "database",
                                   "files",
                                   "accuracy_multiclass.classification_dense",
                                   "configurations.csv")

        cf_object = ConfigurationsFile(automl_path)
        config = cf_object.get_configuration(algorithm_id=32)

        # Assert the type
        self.assertTrue(isinstance(config, pd.Series))

        # Assert is non empty
        self.assertFalse(config.empty)


class TestAlgorithmRunsFile(unittest.TestCase):
    """Test the AlgorithmRuns class.

    The AlgorithmRuns class exposes methods to retrieve values from a given
    algorithm_runs.arff file. We test these methods.
    """

    def test_get_associated_configuration_id_error(self):
        """The configuration retrieved."""
        automl_path = os.path.dirname(os.path.dirname(__file__))
        automl_path = os.path.join(automl_path,
                                   "automl",
                                   "metalearning",
                                   "database",
                                   "files",
                                   "accuracy_multiclass.classification_dense",
                                   "algorithm_runs.arff")

        arf = AlgorithmRunsFile(automl_path)
        with self.assertRaises(ValueError):
            arf.get_associated_configuration_id(instance_id=21170)

    def test_get_associated_configuration_file_error(self):
        """The configuration retrieved."""
        automl_path = os.path.dirname(os.path.dirname(__file__))
        automl_path = os.path.join(automl_path,
                                   "automl",
                                   "metalearning",
                                   "database",
                                   "",
                                   "accuracy_multiclass.classification_dense",
                                   "algorithm_runs.arff")

        with self.assertRaises(ValueError):
            AlgorithmRunsFile(automl_path)

    def test_get_associated_configuration(self):
        """The configuration retrieved."""
        automl_path = os.path.dirname(os.path.dirname(__file__))
        automl_path = os.path.join(automl_path,
                                   "automl",
                                   "metalearning",
                                   "database",
                                   "files",
                                   "accuracy_multiclass.classification_dense",
                                   "algorithm_runs.arff")

        arf = AlgorithmRunsFile(automl_path)
        res = arf.get_associated_configuration_id(instance_id=2117)
        self.assertTrue(isinstance(res, int))


class TestMKDatabaseClient(unittest.TestCase):
    """Test the database client.

    The database client is the main point of interaction for the Data Scientist
    since provides the principal methods to retrieve information. In this class
    we make sure the methos retrieve the necessary types and handle exceptions
    correctly.
    """

    def test_load(self):
        """Test the loading retrieves a valid instance."""
        client = MKDatabaseClient()
        self.assertTrue(isinstance(client, MKDatabaseClient))

    def test_reload(self):
        """Test the loading retrieves a valid instance."""
        client = MKDatabaseClient().reload()
        self.assertTrue(isinstance(client, MKDatabaseClient))

    def test_value_k(self):
        """Test that if a k > maximum_allowd is passed, error raises."""
        dataset = DataLoader.get_openml_dataset(openml_id=46, problem_type=0)
        max_k = len(MKDatabaseClient().metaknwoledge.weighted_matrix()[0])

        # Lower values -> do not fail
        for k in range(1, max_k+1):
            try:
                _, _ = MKDatabaseClient().nearest_datasets(dataset=dataset,
                                                           k=k)
                flag = False
            except Exception:  # pylint: disable=W0703
                flag = True
            finally:
                self.assertFalse(flag)

        # Greater than max -> fails
        with self.assertRaises(ValueError):
            MKDatabaseClient().nearest_datasets(dataset=dataset, k=max_k+1)

        # 0 -> fails
        with self.assertRaises(ValueError):
            MKDatabaseClient().nearest_datasets(dataset=dataset, k=0)

    def test_value_distance_metric(self):
        """Test that a different distance metric evaluates correctly.

        For this, we test with cosine similarity.
        """
        dataset = DataLoader.get_openml_dataset(openml_id=46, problem_type=0)
        dists, _ = MKDatabaseClient().nearest_datasets(
            dataset=dataset,
            distance_metric='cosine'
        )

        for distance in dists[0]:
            self.assertTrue(0 <= distance <= 1)  # it can be by chance but
            # running out of time and I know for 46 the default metric gives
            # higher than one :) Sorry ...

    def test_suggestions_type(self):
        """Test that returned value from meta_suggestions() is correct.

        We know that the internal methods have already been tested, so testing
        the type is the only thing left.
        """
        dataset = DataLoader.get_openml_dataset(openml_id=46, problem_type=0)
        _, idx = MKDatabaseClient().nearest_datasets(dataset=dataset)

        res = MKDatabaseClient().meta_suggestions(dataset, idx[0])

        self.assertTrue(isinstance(res, MLSuggestion))

if __name__ == '__main__':
    unittest.main()
