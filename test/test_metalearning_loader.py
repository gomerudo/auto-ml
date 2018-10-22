"""Module to test methods related to loading the MetaKnowledge."""

import unittest
import os.path
from automl.metalearning.db.load_db import MetaKnowledge, LandmarkModelParser,\
    ConfigurationsFile, AlgorithmRunsFile
from automl.datahandler.dataloader import DataLoader


class TestMetaKnowledge(unittest.TestCase):
    """Check that the loading of the MetaKnowledge is correct."""

    # def test_fake2(self):
    #     lmdb = MetaKnowledge().load_datasets_info()
    #     print(lmdb.costs.attribute_names())

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

    def test_intersection_of_files(self):
        """Test that both feature values and costs have the same id's."""
        lmdb = MetaKnowledge().load_datasets_info()

        # Check they have the same length
        self.assertTrue(lmdb.costs.shape()[0] == lmdb.features.shape()[0])

        # Check they are correctly sorted
        features_ids = list(lmdb.features[:, 0])
        features_ids.reverse()
        a = features_ids.pop()  # pylint: disable=C0103
        while features_ids:
            b = features_ids.pop()  # pylint: disable=C0103
            self.assertTrue(a < b,
                            "The features list is not sorted: Relation \
                            {a} < {b} not satisfied".format(a=a, b=b))
            # update
            a = b  # pylint: disable=C0103

        costs_ids = list(lmdb.costs[:, 0])
        costs_ids.reverse()
        a = costs_ids.pop()  # pylint: disable=C0103
        while costs_ids:
            b = costs_ids.pop()  # pylint: disable=C0103
            self.assertTrue(a < b,
                            "The features list is not sorted: Relation a > b \
                            not satisfied")
            # update
            a = b  # pylint: disable=C0103

        # Finally, test that they are candidates for elementwise multiplication
        # TODO: finish test by checking the n*m <-> m*p relation.
        print(lmdb.costs.shape)
        print(lmdb.features.shape)

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
        check_list = []
        for element in lmdb.costs:
            check_list.append(element[0])

        # ... and then for the other's IDs, remove the elements in check_list
        for element in lmdb.features:
            current_id = element[0]
            print("Checking {}".format(current_id))
            # If one element is not there. Then B - A != 0
            self.assertTrue(current_id in check_list,
                            "Element {id} is not in both costs and features \
                            list".format(id=current_id))
            check_list.remove(current_id)

        # If we come here, then B - A = 0. Then the only check left is
        # if A - B = 0.
        self.assertListEqual(check_list, [],
                             "Features/costs lists do not match IDs")

    # def test_weighted_vectors(self):
    #     lmdb = LoadMetaDB()
    #     lmdb.loadDatasetsInfo()

    #     lmdb.weightedVectors()


class TestLandmarkModelParser(unittest.TestCase):
    """Test the parsing of the landmark models by auto-sklearn."""

    def test_metrics_available(self):
        """Test that the metrics available are loaded correctly."""
        assert LandmarkModelParser.metrics_available()

    def test_non_supported_metric(self):
        """Test that method error raises if no valid metric is passed."""
        with self.assertRaises(ValueError):
            LandmarkModelParser.model_by_metric('achmea_metric')

    def test_model_by_metric(self):
        """Test the behaviour of the model_by_metric method."""
        dataset = DataLoader.get_openml_dataset(openml_id=46, problem_type=0)
        print(LandmarkModelParser.model_by_metric([1, 2], dataset,
                                                  'accuracy'))

    # def test_fakemethod(self):
    #     """Fake test."""
    #     LandmarkModelParser.fake_method()


class TestConfigurationsFile(unittest.TestCase):

    def test_get_configuration(self):
        automl_path = os.path.dirname(os.path.dirname(__file__))
        automl_path = os.path.join(automl_path,
                                   "automl",
                                   "metalearning",
                                   "db",
                                   "files",
                                   "accuracy_multiclass.classification_dense",
                                   "configurations.csv")

        cf = ConfigurationsFile(automl_path)
        cf.get_configuration(32)
        

class TestAlgorithmRunsFile(unittest.TestCase):

    def test_get_configuration(self):
        automl_path = os.path.dirname(os.path.dirname(__file__))
        automl_path = os.path.join(automl_path,
                                   "automl",
                                   "metalearning",
                                   "db",
                                   "files",
                                   "accuracy_multiclass.classification_dense",
                                   "algorithm_runs.arff")

        arf = AlgorithmRunsFile(automl_path)
        res = arf.get_associated_configuration(2117)
        print("Associated algorithm is:", res)

if __name__ == '__main__':
    unittest.main()


