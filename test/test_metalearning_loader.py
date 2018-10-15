"""Module to test methods related to loading the MetaKnowledge."""

import unittest
from automl.metalearning.db.load_db import MetaKnowledge


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

if __name__ == '__main__':
    unittest.main()
