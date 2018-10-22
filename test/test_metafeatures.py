"""Module to test the computation of the metafeatures."""

import unittest

from automl.datahandler.dataloader import DataLoader
from automl.metalearning.metafeatures.metafeatures_interaction \
    import MetaFeaturesManager


class TestMetaFeaturesManager(unittest.TestCase):
    """Test the MetaFeaturesManager class.

    We verify that the computation is performed correctly.
    """

    def test_metafeatures_computation(self):
        """Test that the metafeatures are computed."""
        dataset = DataLoader.get_openml_dataset(openml_id=46, problem_type=0)
        dataset_mf = MetaFeaturesManager(dataset)
        dataset_mf.metafeatures_as_dict()
        print(dataset_mf.metafeatures_as_dict())

        # TODO: finish the test
        # print(dataset_mf.metafeatures_as_pandas_df())
        # print(dataset_mf.metafeatures_as_numpy_array())

    if __name__ == '__main__':
        unittest.main()
