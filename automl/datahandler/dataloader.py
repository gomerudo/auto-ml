"""Module to expose classes to interact with the datasets.

Here we abstract the dataset as a class to ease the interaction within the
package. Namely, we provide the next classes:
    - Dataset: To abstract a dataset together with the properties needed for
               the different AutoML operations.
    - DatasetLoader: To provided methods to obtain dataset from different
                     sources.
"""
import random
import string
import pandas as pd
import scipy
import numpy as np
import openml as oml
from sklearn.model_selection import train_test_split
from automl.metalearning.metafeatures.metafeatures_interaction \
    import MetaFeaturesManager


class Dataset:
    """Class to abstract a dataset.

    In this class we abstract a dataset as an object of features and a target,
    together with categorical indicators for each of the features, and id to
    identify the dataset and a problem type to solve: either classification or
    regression.
    """

    def __init__(self, dataset_id, X, y, categorical=None, problem_type=0):
        """Constructor.

        Atrributes:
            dataset_id      (str) or (int). The identifer for the dataset.
            X               (pandas.DataFrame) The features object.
            y               (pandas.DataFrame) The target object.
            categorical     The categorical indicators for the features.
            problem_type    (int). 0 for classification, 1 for regression.

        """
        if not isinstance(X, pd.DataFrame) and not isinstance(y, pd.DataFrame):
            raise TypeError("X and y must be pandas Data Frames.")

        if y.shape[1] > 1:
            raise ValueError("y data frame should have one column only.")

        self.X = X
        self.y = y

        # Rename target column
        y.columns = ['target']

        if categorical is None:
            self.categorical_indicators = list(np.zeros(X.shape[1]))
        else:
            self.categorical_indicators = categorical

        self.dataset_id = self._random_id() if dataset_id is None \
            else dataset_id
        self.problem_type = problem_type

    def is_regression_problem(self):
        """Whether or not the dataset is registered as regression task."""
        return self.problem_type == 1

    def is_classification_problem(self):
        """Whether or not the dataset is registered as classification task."""
        return self.problem_type == 0

    # TODO: Make it function in utl
    def _random_id(self):
        n_chars = 6
        return ''.join(
            random.choice(
                string.ascii_uppercase + string.digits
            ) for _ in range(n_chars)
        )

    def metafeatures_vector(self):
        """Return the metafeatures of this dataset as a vector (ndarray)."""
        return MetaFeaturesManager(self).metafeatures_as_numpy_array()

    # TODO: How to proceed with sparse values? Do we need to handle that?
    @property
    def n_labels(self):
        """Return the number of different labels (target) for this dataset."""
        return len(self.y['target'].unique())

    def is_sparse(self):
        """Return whether or not the X data is sparse or not."""
        return scipy.sparse.issparse(self.X.values)

    def train_test_split(self, random_state=42, test_size=0.33):
        return train_test_split(self.X, self.y, test_size=test_size,
                                random_state=random_state)

class DataLoader:
    """Class to load dataset as a Dataset class from different sources."""

    @staticmethod
    def parse_dataset(dataset):
        """Parse a pandas data frame as a Dataset class."""
        if dataset is None:
            raise ValueError("Dataset cannot be None")

        # TODO: The rest

    @staticmethod
    def get_openml_dataset(openml_id, problem_type):
        """Fetch a dataset from OpenML and return a Dataset object.

        Attributes:
            - int: openml_id. ID for the dataset, as stored in OpenML.
            - int: problem_type. Type of problem to solve in the dataset.
                0 for classification, 1 for regression.

        """
        openml_dataset = oml.datasets.get_dataset(openml_id)
        features, target, categorical_indicators, attribute_names = \
            openml_dataset.get_data(
                target=openml_dataset.default_target_attribute,
                return_attribute_names=True,
                return_categorical_indicator=True
            )

        features = pd.DataFrame(features, columns=attribute_names)
        target = pd.DataFrame(target, columns=["target"])

        return Dataset(dataset_id="{}-{}".format(openml_dataset.dataset_id,
                                                 openml_dataset.name),
                       X=features, y=target,
                       categorical=categorical_indicators,
                       problem_type=problem_type)
