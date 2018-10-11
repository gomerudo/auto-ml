import unittest

from automl.datahandler.dataloader import Dataset, DataLoader
from automl.metalearning.metafeatures.metafeatures_manager import DatasetMetaFeatures

class Test_TestDatasetMetaFeatures(unittest.TestCase):  

    def test_metafeatures_computation(self):
        dataset = DataLoader.get_openml_dataset(openml_id = 46, problem_type = 0)
        dataset_mf = DatasetMetaFeatures(dataset)
        dataset_mf.get_metafeatures()
        print(dataset_mf.get_metafeatures())
        # print(dataset_mf.metafeatures_as_pandas_df())
        # print(dataset_mf.metafeatures_as_numpy_array())

    if __name__ == '__main__':
        unittest.main()
