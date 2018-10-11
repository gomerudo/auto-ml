import random
import string
import pandas as pd
import numpy as np

import openml as oml
from openml import tasks, runs, datasets


class Dataset:

    def __init__(self, id, X, y, categorical = None, problem_type = 0):

        if not isinstance(X, pd.DataFrame) and not isinstance(y, pd.DataFrame):
            raise TypeError("X and y must be pandas Data Frames.")

        if y.shape[1] > 1:
            raise ValueError("y data frame should have one column only.")
        
        self.X = X
        self.y = y
        
        # Rename target column
        y.columns = ['target']

        self.categorical_indicators = list(np.zeros(X.shape[1])) if categorical is None else categorical

        self.id = id if id is not None else self._randomID()
        self.problem_type = problem_type

    def is_regression_problem(self):
        return self.problem_type == 1
    
    def is_classification_problem(self):
        return self.problem_type == 0

    def _randomID(self):
        N = 6
        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))
        
    # TODO: How to proceed with sparse values? Do we need to handle that?
    @property
    def n_labels(self):
        len(self.y['target'].unique())

class DataLoader :

    @staticmethod
    def parse_dataset(dataset) :
        if dataset is None: 
            raise ValueError("Dataset cannot be None")
        else :
            pass

    @staticmethod
    def get_openml_dataset(openml_id, problem_type):
        """Fetch a dataset from OpenML and return a Dataset object.

        Attributes:
            - int: openml_id. ID for the dataset, as stored in OpenML.
            - int: problem_type. Type of problem to solve in the dataset.
                0 for classification, 1 for regression
        """
        openml_dataset = oml.datasets.get_dataset(openml_id)
        X, y, categorical_indicators, attribute_names = \
            openml_dataset.get_data(target = openml_dataset.default_target_attribute,
            return_attribute_names = True, return_categorical_indicator = True)

        X = pd.DataFrame(X, columns = attribute_names)
        y = pd.DataFrame(y, columns = ["target"])
        
        return Dataset(id="{}-{}".format(openml_dataset.dataset_id, openml_dataset.name),
                        X=X, y = y, 
                        categorical=categorical_indicators, 
                        problem_type=problem_type)
