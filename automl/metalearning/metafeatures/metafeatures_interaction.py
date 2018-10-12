"""Module intented to expose the classes to provide dataset metafeatures.

This module includes the global variables to refer the meta-features and the
next classes:
    - MetaFeaturesHelper: To perform the opearations.
    - MetaFeaturesManager: To expose the metafeatures for a given dataset.
"""

import pandas as pd
import numpy as np
from automl.metalearning.metafeatures.computation \
    import StatisticalInformation as si

# Define the constants

STATS_CLASS_ENTROPY = 'ClassEntropy'
STATS_CLASS_PROB_MAX = 'ClassProbabilityMax'
STATS_CLASS_PROB_MEAN = 'ClassProbabilityMean'
STATS_CLASS_PROB_MIN = 'ClassProbabilityMin'
STATS_CLASS_PROB_STD = 'ClassProbabilitySTD'

STATS_DS_RATIO = 'DatasetRatio'
STATS_INV_DS_RATIO = 'InverseDatasetRatio'

STATS_KURT_MAX = 'KurtosisMax'
STATS_KURT_MEAN = 'KurtosisMean'
STATS_KURT_MIN = 'KurtosisMin'
STATS_KURT_STD = 'KurtosisSTD'

STATS_LANDMARK_1NN = 'Landmark1NN'
STATS_LANDMARK_DNL = 'LandmarkDecisionNodeLearner'
STATS_LANDMARK_DT = 'LandmarkDecisionTree'
STATS_LANDMARK_NB = 'LandmarkNaiveBayes'
STATS_LANDMARK_RNL = 'LandmarkRandomNodeLearner'

STATS_LOG_DS_RATIO = 'LogDatasetRatio'
STATS_LOG_INV_DS_RATIO = 'LogInverseDatasetRatio'
STATS_LOG_N_FEAT = 'LogNumberOfFeatures'
STATS_LOG_N_INST = 'LogNumberOfInstances'

STATS_N_CAT_FEAT = 'NumberOfCategoricalFeatures'
STATS_N_CLASS = 'NumberOfClasses'
STATS_N_FEAT = 'NumberOfFeatures'
STATS_N_FEAT_NA = 'NumberOfFeaturesWithMissingValues'
STATS_N_INST = 'NumberOfInstances'
STATS_N_INST_NA = 'NumberOfInstancesWithMissingValues'
STATS_N_NA = 'NumberOfMissingValues'
STATS_N_NF = 'NumberOfNumericFeatures'

STATS_PCA_F95V = 'PCAFractionOfComponentsFor95PercentVariance'
STATS_PCA_KURT_1PC = 'PCAKurtosisFirstPC'
STATS_PCA_SKEW_1PC = 'PCASkewnessFirstPC'

STATS_PERC_FEAT_NA = 'PercentageOfFeaturesWithMissingValues'
STATS_PERC_INST_NA = 'PercentageOfInstancesWithMissingValues'
STATS_PERC_NA = 'PercentageOfMissingValues'

STATS_RATIO_NOM_NUM = 'RatioNominalToNumerical'
STATS_RATIO_NUM_NOM = 'RatioNumericalToNominal'

STATS_SKEW_MAX = 'SkewnessMax'
STATS_SKEW_MEAN = 'SkewnessMean'
STATS_SKEW_MIN = 'SkewnessMin'
STATS_SKEW_STD = 'SkewnessSTD'

STATS_SYM_MAX = 'SymbolsMax'
STATS_SYM_MEAN = 'SymbolsMean'
STATS_SYM_MIN = 'SymbolsMin'
STATS_SYM_STD = 'SymbolsSTD'
STATS_SYM_SUM = 'SymbolsSum'


class MetaFeaturesHelper:
    """Helper to obtain the metafeatures without exposing the logic behind.

    This class interacts with the StatisticalInformation class to compute and
    obtain the metafeatures of a given dataset.
    """

    def __init__(self, dataset=None):
        """Constructor.

        It initializes an empty array and saves the dataset to work with.

        Attributes:
            dataset (automl.datahandler.dataloader.Dataset) The dataset to work
                    with.

        """
        self.metafeatures_map = dict()
        self.dataset = dataset

    def _class_statistics(self):
        self.metafeatures_map.update({
            STATS_CLASS_ENTROPY:
                si.class_entropy(self.dataset.y.values),

            STATS_CLASS_PROB_MAX:
                si.class_probability_max(self.dataset.y.values),

            STATS_CLASS_PROB_MEAN:
                si.class_probability_mean(self.dataset.y.values),

            STATS_CLASS_PROB_MIN:
                si.class_probability_min(self.dataset.y.values),

            STATS_CLASS_PROB_STD:
                si.class_probability_std(self.dataset.y.values),
        })

    def _ratio_statistics(self):
        self.metafeatures_map.update({
            STATS_DS_RATIO:
                si.dataset_ratio(self.dataset.X.values),

            STATS_INV_DS_RATIO:
                si.inverse_dataset_ratio(self.dataset.X.values),

            STATS_RATIO_NOM_NUM:
                si.ratio_nominal_numerical(
                    self.dataset.categorical_indicators),

            STATS_RATIO_NUM_NOM:
                si.ratio_numerical_nominal(
                    self.dataset.categorical_indicators),
        })

    def _kurtosis_statistics(self):
        self.metafeatures_map.update({
            STATS_KURT_MAX:
                si.kurtosis_max(self.dataset.X.values,
                                self.dataset.categorical_indicators),

            STATS_KURT_MEAN:
                si.kurtosis_mean(self.dataset.X.values,
                                 self.dataset.categorical_indicators),

            STATS_KURT_MIN:
                si.kurtosis_min(self.dataset.X.values,
                                self.dataset.categorical_indicators),

            STATS_KURT_STD:
                si.kurtosis_std(self.dataset.X.values,
                                self.dataset.categorical_indicators),
        })

    def _landmark_statistics(self):
        self.metafeatures_map.update({
            STATS_LANDMARK_1NN:
                si.landmark_1NN(self.dataset.X.values, self.dataset.y.values),

            STATS_LANDMARK_DNL:
                si.landmark_decision_node_learner(self.dataset.X.values,
                                                  self.dataset.y.values),

            STATS_LANDMARK_DT:
                si.landmark_decision_tree(self.dataset.X.values,
                                          self.dataset.y.values),

            STATS_LANDMARK_NB:
                si.landmark_naive_bayes(self.dataset.X.values,
                                        self.dataset.y.values),

            STATS_LANDMARK_RNL:
                si.landmark_random_node_learner(self.dataset.X.values,
                                                self.dataset.y.values),
        })

    def _log_statistics(self):
        self.metafeatures_map.update({
            STATS_LOG_DS_RATIO:
                si.log_dataset_ratio(self.dataset.X.values),

            STATS_LOG_INV_DS_RATIO:
                si.log_inverse_dataset_ratio(self.dataset.X.values),

            STATS_LOG_N_FEAT:
                si.log_number_of_features(self.dataset.X.values),

            STATS_LOG_N_INST:
                si.log_number_of_instances(self.dataset.X.values),
        })

    def _n_statistics(self):
        self.metafeatures_map.update({
            STATS_N_CAT_FEAT:
                si.number_of_categorical_features(
                    self.dataset.categorical_indicators),

            STATS_N_CLASS:
                si.number_of_classes(self.dataset.y.values),

            STATS_N_FEAT:
                si.number_of_features(self.dataset.X.values),

            STATS_N_FEAT_NA:
                si.number_of_features_with_na(self.dataset.X.values),

            STATS_N_INST:
                si.number_of_instances(self.dataset.X.values),

            STATS_N_INST_NA:
                si.number_of_features_with_na(self.dataset.X.values),

            STATS_N_NA:
                si.number_of_missing_values(self.dataset.X.values),

            STATS_N_NF:
                si.number_of_numeric_features(
                    self.dataset.categorical_indicators),
        })

    def _pca_statistics(self):
        pca = si.pca(self.dataset.X.values)

        self.metafeatures_map.update({
            STATS_PCA_F95V:
                si.pca_fraction_components_95v(self.dataset.X.values, pca),

            STATS_PCA_KURT_1PC:
                si.pca_kurtosis_first_pc(self.dataset.X.values, pca),

            STATS_PCA_SKEW_1PC:
                si.pca_skewness_first_pc(self.dataset.X.values, pca),
        })

    def _percentage_statistics(self):
        self.metafeatures_map.update({
            STATS_PERC_FEAT_NA:
                si.percentage_of_features_with_na(self.dataset.X.values),

            STATS_PERC_INST_NA:
                si.percentage_of_instances_with_na(self.dataset.X.values),

            STATS_PERC_NA:
                si.percentage_of_missing_values(self.dataset.X.values),
        })

    def _skewness_statistics(self):
        self.metafeatures_map.update({
            STATS_SKEW_MAX:
                si.skewness_max(self.dataset.X.values,
                                self.dataset.categorical_indicators),

            STATS_SKEW_MEAN:
                si.skewness_mean(self.dataset.X.values,
                                 self.dataset.categorical_indicators),

            STATS_SKEW_MIN:
                si.skewness_min(self.dataset.X.values,
                                self.dataset.categorical_indicators),

            STATS_SKEW_STD:
                si.skewness_std(self.dataset.X.values,
                                self.dataset.categorical_indicators),
        })

    def _symbol_statistics(self):
        self.metafeatures_map.update({
            STATS_SYM_MAX:
                si.symbols_max(self.dataset.X.values,
                               self.dataset.categorical_indicators),

            STATS_SYM_MEAN:
                si.skewness_mean(self.dataset.X.values,
                                 self.dataset.categorical_indicators),

            STATS_SYM_MIN:
                si.skewness_min(self.dataset.X.values,
                                self.dataset.categorical_indicators),

            STATS_SYM_STD:
                si.symbols_std(self.dataset.X.values,
                               self.dataset.categorical_indicators),

            STATS_SYM_SUM:
                si.symbols_sum(self.dataset.X.values,
                               self.dataset.categorical_indicators),
        })

    def compute_metafeatures(self):
        """Return a dictionary with the computed metafeatures for the dataset.

        Returns:
            dict:   The dictionary with metafeatures where keys are the names
                    of the metafeatures.

        """
        self._class_statistics()
        self._kurtosis_statistics()
        self._landmark_statistics()
        self._log_statistics()
        self._n_statistics()
        self._pca_statistics()
        self._percentage_statistics()
        self._ratio_statistics()
        self._symbol_statistics()

        return self.metafeatures_map


class MetaFeaturesManager:
    """Class to obtain/interact with the metafeatures for a given dataset."""

    def __init__(self, dataset=None):
        """Constructor.

        This constructor initializes the attributes.

        Attributes:
            dataset (automl.datahandler.data_loader.Dataset). The dataset to
                    work with.

        """
        self.dataset = dataset
        self._metafeatures = None

    def metafeatures_as_dict(self, recompute=False):
        """Get the dataset's metafeatures in the form of a dictionary.

        Attributes:
            recompute   (bool). Whether or not to force the recomputing of the
                        metafeatures even if they were already computed.
                        Default is value is False.

        """
        if self._metafeatures is None or recompute:
            helper = MetaFeaturesHelper(self.dataset)
            self._metafeatures = helper.compute_metafeatures()

        return self._metafeatures

    def metafeatures_as_pandas_df(self, recompute=False):
        """Get the dataset's metafeatures in the form of a pandas data frame.

        Attributes:
            recompute   (bool). Whether or not to force the recomputing of the
                        metafeatures even if they were already computed.
                        Default is value is False.

        """
        headers = []
        data = []

        for key, value in self.metafeatures_as_dict(recompute).items():
            headers.append(key)
            data.append(value)

        data_frame = pd.DataFrame(np.asarray(data).reshape((1, len(data))),
                                  columns=headers)
        data_frame.sort_index(axis=1, ascending=True, inplace=True)
        return data_frame

    def metafeatures_as_numpy_array(self, recompute=False):
        """Get the dataset's metafeatures in the form of a numpy darray.

        Attributes:
            recompute   (bool). Whether or not to force the recomputing of the
                        metafeatures even if they were already computed.
                        Default is value is False.

        """
        return self.metafeatures_as_pandas_df(recompute).values
