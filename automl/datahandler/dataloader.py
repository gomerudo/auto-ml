"""Module to expose classes to interact with a given dataset.

Here we abstract any dataset as a class to ease the interaction within the
package's modules. Additionally, we provide a class to load datasets in this
new format.
"""

import pandas as pd
import scipy
import scipy.sparse
import numpy as np
import openml as oml
import sklearn.model_selection
from automl import automl_log
from automl.errors.customerrors import CurrentlyNonSupportedError
from automl.metalearning.metafeatures.metafeatures_interaction \
    import MetaFeaturesManager
from automl.utl.miscellaneous import generate_random_id

_TARGET_NAME = "target"


class Dataset:
    """Class abstracting a dataset.

    In this class we abstract a dataset as an object composed of a features
    pandas.DataFrame and a target pandas.DataFrame (with the name 'target').
    Additionally, a Dataset contains categorical_indicators indicators for each
    of the feature columns, and ID and a problem type to solve (either
    classification or regression).

    Args:
        dataset_id (str): An string identifying the dataset. If `None` is
            passed to the constructor, then the default is a string of 6 random
            letters/digits.
        X (pandas.DataFrame): The Data Frame containing the features of the
            dataset. Shape is `(n, m)`.
        y (pandas.DataFrame): The Data Frame containing the target value (e.g.
            the label for a class). It should be of shape `(n, 1)`.
        categorical_indicators (list): A list of `m` booleans following a
            1-to-1 relation with the features' columns that indicate whether or
            not the feature is categorical_indicators. If `None` is passed to
            the constructor, the default is a list of m `False` values.
        problem_type (int): 0 indicates classification, 1 indicates regression.
            Defaults to 0.

    Raises:
        TypeError: If `X` and/or `y` are not pandas Data Frames.
        ValueError: If `X` and `y`  are not of shape `(n, m)`, `(n, 1)`
            respectively.

    Attributes:
        X (pandas.DataFrame): The Data Frame containing the features of the
            dataset. Shape is `(n, m)`.
        y (pandas.DataFrame): The Data Frame containing the target value (e.g.
            the label for a class). It should be of shape `(n, 1)`.
        categorical_indicators (list): A list of `m` booleans following a
            1-to-1 relation with the features' columns that indicate whether or
            not the feature is categorical_indicators.
        dataset_id (str): An string identifying the dataset.
        problem_type (int): 0 indicates classification, 1 indicates regression.

    """

    def __init__(self, dataset_id=None, X=None, y=None,
                 categorical_indicators=None, problem_type=0):
        """Constructor."""
        # Perform validations.
        if not isinstance(X, pd.DataFrame) and not isinstance(y, pd.DataFrame):
            raise TypeError("X and y must be pandas Data Frames.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of \
                             observations (n).")

        if y.shape[1] > 1:
            raise ValueError("y data frame should have one column only.")

        # Assign the X, y data frames.
        self.X = X  # pylint: disable=C0103
        self.y = y  # pylint: disable=C0103

        # Rename target column
        y.columns = [_TARGET_NAME]

        # Assign the categorical indicators
        if categorical_indicators is None:
            self.categorical_indicators = list(np.zeros(X.shape[1]))
        else:
            self.categorical_indicators = categorical_indicators

        # Validate the categorical indicators
        if X.shape[1] != len(self.categorical_indicators):
            raise ValueError("The length of the categorical indicators should \
                             match the number of features.")

        # Assign the ID
        self.dataset_id = generate_random_id(n_chars=6) if dataset_id is None \
            else str(dataset_id)

        # Assign the problem type
        self.problem_type = problem_type

        # Validate missing and infinite data
        self._internal_validation()

    def _internal_validation(self):
        # Check for missing values
        nan_sum_x = self.X.isna().sum().sum()
        nan_sum_y = self.y.isna().sum().sum()

        # Log missing values messages
        if nan_sum_x > 0:
            automl_log("Features set (X) in dataset '{d_id}' contains {n_na} \
missing values. Please not that with missing data, results can be innacurate. \
Fix the missing values yourself.".format(d_id=self.dataset_id,
                                         n_na=nan_sum_x),
                       'WARNING')

        if nan_sum_y > 0:
            automl_log("Target set (y) in dataset '{d_id}' contains {n_na} \
missing values. Please not that with missing data, results can be innacurate. \
Fix the missing values yourself.".format(d_id=self.dataset_id,
                                         n_na=nan_sum_y),
                       'WARNING')
        # Check for inifinite values
        inf_sum_x = np.isinf(self.X.values).ravel().sum()
        inf_sum_y = np.isinf(self.y.values).ravel().sum()

        # Log infinite values messages
        if inf_sum_x > 0:
            automl_log("Features set (X) in dataset '{d_id}' contains {n_na} \
missing values. Please not that with missing data, results can be innacurate. \
Fix the missing values yourself.".format(d_id=self.dataset_id,
                                         n_na=nan_sum_x))

        if inf_sum_y > 0:
            automl_log("Target set (y) in dataset '{d_id}' contains {n_na} \
missing values. Please not that with missing data, results can be innacurate. \
Fix the missing values yourself.".format(d_id=self.dataset_id,
                                         n_na=nan_sum_y))

    def is_regression_problem(self):
        """Whether or not the dataset is registered as regression task.

        Returns:
            bool: True if the dataset has been marked as a regression problem.

        """
        return self.problem_type == 1

    def is_classification_problem(self):
        """Whether or not the dataset is registered as classification task.

        Returns:
            bool: True if the dataset has been marked a classification problem.

        """
        return self.problem_type == 0

    def metafeatures_vector(self):
        """Return the metafeatures of this dataset as a vector (numpy array).

        Returns:
            np.array: The metafeatures as a numpy array.

        """
        res = MetaFeaturesManager(self).metafeatures_as_numpy_array()
        if np.count_nonzero(~np.isnan(res)) > 0:
            automl_log("It was not possible to compute all metafeatures (see \
log messages). We will replace the NaN values in the meta-features vector \
with 0", 'WARNING')
            np.nan_to_num(res, False)
        return res

    # TODO: How to proceed with sparse values? Do we need to handle that?
    @property
    def n_labels(self):
        """Provide the number of different labels (target) for this dataset."""
        return len(self.y[_TARGET_NAME].unique())

    def is_sparse(self):
        """Return whether or not the X data is sparse or not.

        Returns:
            bool: True if the X data frame is an sparse representation.

        """
        return scipy.sparse.issparse(self.X.values)

    def train_test_split(self, random_state=42, test_size=0.33):
        """Make a train-test split as defined in scikit-learn for the dataset.

        Args:
            random_state (int): The random_state to initialize with so that
                the results can be reproduced.
            test_size (float): Proportion of split for the test set.

        Returns:
            tuple: A 4-tuple as train_test_split in scikit-learn.

        """
        return sklearn.model_selection.train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state
        )

    @property
    def n_features(self):
        """Return the number of features in the dataset.

        Returns:
            int: The number of features.

        """
        return self.X.shape[1]


class DataLoader:
    """Class to load dataset as a Dataset class from different sources.

    It exposes static methods only.
    """

    @staticmethod
    def get_openml_dataset(openml_id, problem_type):
        """Fetch a dataset from OpenML and return a Dataset object.

        Args:
            openml_id (int): ID for the dataset, as stored in OpenML.
            problem_type (int): Type of problem to solve in the dataset.
                0 for classification, 1 for regression.

        Returns:
            Dataset: An auto-ml Dataset, as defined in this module.
                Its default ID will be the concatenation of the OpenML dataset
                name and ID.

        """
        automl_log(
            "Loading dataset {d_id} from OpenML:".format(d_id=openml_id),
            'INFO')
        openml_dataset = oml.datasets.get_dataset(openml_id)
        features, target, categorical_indicators, attribute_names = \
            openml_dataset.get_data(
                target=openml_dataset.default_target_attribute,
                return_attribute_names=True,
                return_categorical_indicator=True
            )

        if scipy.sparse.issparse(features):
            raise CurrentlyNonSupportedError("Sparse datasets are not \
supported yet in Achmea's auto-ml solution")

        features = pd.DataFrame(features, columns=attribute_names)
        target = pd.DataFrame(target, columns=[_TARGET_NAME])

        return Dataset(dataset_id="{}-{}".format(openml_dataset.dataset_id,
                                                 openml_dataset.name),
                       X=features, y=target,
                       categorical_indicators=categorical_indicators,
                       problem_type=problem_type)
