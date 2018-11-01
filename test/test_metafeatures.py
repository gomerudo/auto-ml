"""Module to test the computation of the metafeatures."""

import unittest

import pandas as pd
import numpy as np
from automl.datahandler.dataloader import DataLoader
import automl.metalearning.metafeatures.metafeatures_interaction as mfi
from automl.metalearning.metafeatures.metafeatures_interaction \
    import MetaFeaturesManager


class TestMetaFeaturesManager(unittest.TestCase):
    """Test the MetaFeaturesManager class.

    We verify that the computation is performed correctly.
    """

    def setUp(self):
        """Initialize global values for the test."""
        self.dataset = \
            DataLoader.get_openml_dataset(openml_id=46, problem_type=0)
        self.expected_metafeatures = [
            mfi.STATS_CLASS_ENTROPY,
            mfi.STATS_CLASS_PROB_MAX,
            mfi.STATS_CLASS_PROB_MEAN,
            mfi.STATS_CLASS_PROB_MIN,
            mfi.STATS_CLASS_PROB_STD,
            mfi.STATS_DS_RATIO,
            mfi.STATS_INV_DS_RATIO,
            mfi.STATS_KURT_MAX,
            mfi.STATS_KURT_MEAN,
            mfi.STATS_KURT_MIN,
            mfi.STATS_KURT_STD,
            mfi.STATS_LANDMARK_1NN,
            mfi.STATS_LANDMARK_DNL,
            mfi.STATS_LANDMARK_DT,
            mfi.STATS_LANDMARK_LDA,
            mfi.STATS_LANDMARK_NB,
            mfi.STATS_LANDMARK_RNL,
            mfi.STATS_LOG_DS_RATIO,
            mfi.STATS_LOG_INV_DS_RATIO,
            mfi.STATS_LOG_N_FEAT,
            mfi.STATS_LOG_N_INST,
            mfi.STATS_N_CAT_FEAT,
            mfi.STATS_N_CLASS,
            mfi.STATS_N_FEAT,
            mfi.STATS_N_FEAT_NA,
            mfi.STATS_N_INST,
            mfi.STATS_N_INST_NA,
            mfi.STATS_N_NA,
            mfi.STATS_N_NF,
            mfi.STATS_PCA_F95V,
            mfi.STATS_PCA_KURT_1PC,
            mfi.STATS_PCA_SKEW_1PC,
            mfi.STATS_PERC_FEAT_NA,
            mfi.STATS_PERC_INST_NA,
            mfi.STATS_PERC_NA,
            mfi.STATS_RATIO_NOM_NUM,
            mfi.STATS_RATIO_NUM_NOM,
            mfi.STATS_SKEW_MAX,
            mfi.STATS_SKEW_MEAN,
            mfi.STATS_SKEW_MIN,
            mfi.STATS_SKEW_STD,
            mfi.STATS_SYM_MAX,
            mfi.STATS_SYM_MEAN,
            mfi.STATS_SYM_MIN,
            mfi.STATS_SYM_STD,
            mfi.STATS_SYM_SUM,
        ]

    def test_metafeatures_computation_dict(self):
        """Test that the metafeatures are computed."""
        dataset_mf = MetaFeaturesManager(self.dataset)
        returned_value = dataset_mf.metafeatures_as_dict()

        # Assert type is right
        self.assertTrue(isinstance(returned_value, dict))

        # Assert values are right
        to_evaluate = self.expected_metafeatures.copy()
        evaluated = list(returned_value.keys())
        while to_evaluate:
            metafeature = to_evaluate.pop()
            if metafeature in evaluated:
                evaluated.remove(metafeature)

        self.assertTrue(not evaluated and not to_evaluate)

    def test_metafeatures_computation_numpy(self):
        """Test that the metafeatures are computed."""
        dataset_mf = MetaFeaturesManager(self.dataset)
        returned_value = dataset_mf.metafeatures_as_numpy_array()

        # Assert type is right
        self.assertTrue(isinstance(returned_value, np.ndarray))

        # Assert shape is right
        self.assertTrue(
            returned_value.shape[0] == 1 and
            returned_value.shape[1] == len(self.expected_metafeatures)
        )

    def test_metafeatures_computation_pandas(self):
        """Test that the metafeatures are computed."""
        dataset_mf = MetaFeaturesManager(self.dataset)
        returned_value = dataset_mf.metafeatures_as_pandas_df()

        # Assert type is right
        self.assertTrue(isinstance(returned_value, pd.DataFrame))

        # Assert shape is right
        self.assertTrue(
            returned_value.shape[0] == 1 and
            returned_value.shape[1] == len(self.expected_metafeatures)
        )

        # Assert values are right
        to_evaluate = self.expected_metafeatures.copy()
        evaluated = list(returned_value.columns.values)
        while to_evaluate:
            metafeature = to_evaluate.pop()
            if metafeature in evaluated:
                evaluated.remove(metafeature)

        self.assertTrue(not evaluated and not to_evaluate)

    # TODO: Test the computation is correct... a hard one, isn't?

    if __name__ == '__main__':
        unittest.main()
