"""TODO.

Some test
"""

import unittest
# from automl.datahandler.dataloader import DataLoader
from automl.metalearning.db.load_db import LoadMetaDB
import automl.errors.customerrors 

class Test_TestMetalearningLoader(unittest.TestCase):  
    """TODO.

    Add more
    """

    def test_fake2(self):
        lmdb = LoadMetaDB().load_datasets_info()
        print(lmdb.costs.attribute_names())

    def test_load_db_init(self):
        """Test initialization of the LoadMetaDB object.

        By default, it needs to assign None references to the costs/features 
        objects.
        """

        lmdb = LoadMetaDB()

        # We assert everything is None
        self.assertIsNone(lmdb.costs, 
            "List for costs should be empty while initializing")
        self.assertIsNone(lmdb.features, 
            "List for features should be empty while initializing")

    def test_arrays(self):
        lmdb = LoadMetaDB().load_datasets_info()
        
        # Check they have the same length
        self.assertTrue(lmdb.costs.shape()[0] == lmdb.features.shape()[0])

        # Check they are correctly sorted
        featuresIDs = list(lmdb.features[:, 0])
        featuresIDs.reverse()
        a = featuresIDs.pop()
        while featuresIDs:
            b = featuresIDs.pop()
            self.assertTrue(a < b, 
                "The features list is not sorted: Relation \
{a} < {b} not satisfied".format(a = a, b = b))
            # update
            a = b

        costsIds = list(lmdb.costs[:, 0])
        costsIds.reverse()
        a = costsIds.pop()
        while costsIds:
            b = costsIds.pop()
            self.assertTrue(a < b, 
                "The features list is not sorted: Relation a > b not satisfied")
            # update
            a = b
        
        # Finally, test that they are candidates for element wise multiplication
        print(lmdb.costs.shape)
        print(lmdb.features.shape)


    def test_feature_files_structure(self):
        """Test that feature_values and feature_costs files are using the same
        observations.

        For a proper usage of these files, we only accept the case when both
        use exactly the same IDs - i.e., if we create two lists with each's IDs
        then: A - B = B - A = empty set.
        """

        lmdb = LoadMetaDB().load_datasets_info()

        # Check they are not None after initialization
        self.assertIsNotNone(lmdb.features)
        self.assertIsNotNone(lmdb.costs)

        # And then check the IDs ...
        
        # ... by creating a new list with the IDs of one list ...
        checkList = []
        for element in lmdb.costs:
            checkList.append(element[0])
        
        # ... and then for the other's IDs, remove the elements in checkList
        for element in lmdb.features:
            currentId = element[0]
            print("Checking {}".format(currentId))
            # If one element is not there. Then B - A != 0
            self.assertTrue(currentId in checkList, 
                "Element {id} is not in both costs and features list".format(
                    id = currentId
                ))
            checkList.remove(currentId)

        # If we come here, then B - A = 0. Then the only check left is
        # if A - B = 0. 
        self.assertListEqual(checkList, [], 
            "Features/costs lists do not match IDs")
        
    # def test_weighted_vectors(self):
    #     lmdb = LoadMetaDB()
    #     lmdb.loadDatasetsInfo()

    #     lmdb.weightedVectors()

if __name__ == '__main__':
    unittest.main()
