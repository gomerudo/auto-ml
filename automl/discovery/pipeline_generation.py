"""Module defining the functionality to generate a pipeline using TPOT GP."""

import re
from tpot import TPOTClassifier
from tpot import TPOTRegressor
from automl.datahandler.dataloader import Dataset

class PipelineDiscovery:

    def __init__(self, dataset=None, search_space='scikit-learn',
                 tpot_params=None):
        self.dataset = dataset
        self.search_space = search_space
        self.validation_score = None
        self._tpot_optimizer = None
        self._passed_tpot_params = tpot_params # Should be a dict

        if not isinstance(dataset, Dataset):
            raise TypeError("Dataset must be of type AutoML Dataset")

        if isinstance(search_space, str):
            assert(search_space in ['scikit-learn'])

        elif not isinstance(search_space, dict):
            raise TypeError("search-space must be an string or dict")

    def discover(self, train_test=None, limit_time=0, random_state=42):
        # TODO: Handle train_test_split in a different way to allow for argvals

        # Define the arguments as a dictionary
        arguments = {
            "generations": 5,
            "population_size": 20,
            "cv": 5,
            "random_state": random_state,
            "verbosity": 2
        }

        # If the search space is defined, then use it
        if isinstance(self.search_space, dict):
            arguments['config_dict'] = self.search_space

        # If the initially passed tpot params are not none dict, we extend args
        if self._passed_tpot_params is not None \
        and isinstance(self._passed_tpot_params, dict):
            arguments.update(self._passed_tpot_params)

        # Create classifier or regressor, depending on the associated problem
        if self.dataset.is_classification_problem():
            self._tpot_optimizer = TPOTClassifier(**arguments)

        if self.dataset.is_regression_problem():
            self._tpot_optimizer = TPOTRegressor(**arguments)

        # Create the train_test split, for now...

        x_train, x_val, y_train, y_val = self.dataset.train_test_split()

        # Fit TPOT so we discover the pipeline
        self._tpot_optimizer.fit(x_train, y_train)

        # Store the validation score obtained for our validation) set
        self.validation_score = self._tpot_optimizer.score(x_val, y_val)

        return self._tpot_optimizer.fitted_pipeline_

    # Provide the score for any validation set
    def score(self, x_val, y_val):
        self.validation_score = self._tpot_optimizer.score(x_val, y_val)
        return self.validation_score

    def save_pipeline(self, target_dir="", file_name=None):
        # Call the save pipeline object
        # TODO: Export with tpot built-in exporter.

        if file_name is None:
            file_name = "{data_id}.py".format(data_id=self.dataset.dataset_id)

        if target_dir is not None:
            file_name = "{dirname}/{basename}".format(dirname = target_dir,
                                                      basename=file_name)

        self._tpot_optimizer.export(file_name)

    @property
    def tpot_object(self):
        return self._tpot_optimizer

    @property
    def pipeline(self):
        if self._tpot_optimizer is None:
            return None

        return self._tpot_optimizer.fitted_pipeline_