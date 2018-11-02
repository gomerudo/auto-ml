"""Module defining the functionality to generate a pipeline using TPOT GP."""

from tpot import TPOTClassifier
from tpot import TPOTRegressor
from automl.datahandler.dataloader import Dataset


class PipelineDiscovery:
    """Discover a pipeline for a given dataset, using a given metric.

    Args:
        dataset (Dataset): The dataset to work with. Defaults to None.
        search_space (str or dict): The search space to use for the discovery
            operation. If string, it should be any of the following:
            'scikit-learn'. If dict, it must comply with the TPOT config_dict
            format. Defaults to 'scikit-learn'.
        tpot_params (dict): The extra parameters to pass to the TPOT object
            (either a TPOTClassifier or a TPOTRegressor). Defaults to None.

    Raises:
        TypeError: If any of the arguments do not satisfies the type
            constraints described above.

    Attributes:
        dataset (Dataset): The dataset to work with.
        search_space (str or dict): The search space to use for the discovery
            operation. If string, it should be any of the following:
            'scikit-learn'. If dict, it must comply with the TPOT config_dict
            format.
        validation_score (dict): The extra parameters to pass to the TPOT
            object (either a TPOTClassifier or a TPOTRegressor).

    """

    # TODO: accept a metric
    def __init__(self, dataset=None, search_space='scikit-learn',
                 tpot_params=None, evaluation_metric='accuracy'):
        """Constructor."""
        self.dataset = dataset
        self.search_space = search_space
        self.validation_score = None
        self._tpot_optimizer = None
        self._passed_tpot_params = tpot_params
        self.evaluation_metric = evaluation_metric

        if not isinstance(dataset, Dataset):
            raise TypeError("Dataset must be of type AutoML Dataset")

        if isinstance(search_space, str):
            assert search_space in ['scikit-learn']

        elif not isinstance(search_space, dict):
            raise TypeError("search-space must be an string or dict")

        if self._passed_tpot_params is not None and not \
                isinstance(self._passed_tpot_params, dict):
            raise TypeError("The TPOT args must be passed as a dictionary")

        # TODO: Validate evaluation_metric

    def discover(self, limit_time=None, random_state=42):
        """Perform the discovery of a pipeline.

        Args:
            limit_time (int): In minutes, the maximum time to wait for the
                generation of the pipeline. If None, ignored.
            random-state (int): The number to seed the random state with.

        Returns:
            sklearn.pipeline.Pipeline: The resulting pipeline.

        """
        # Define the arguments as a dictionary
        arguments = {
            "generations": 5,
            "population_size": 20,
            "cv": 5,
            "random_state": random_state,
            "verbosity": 2,
            "max_time_mins": limit_time,
        }

        # If the search space is defined, then use it
        if isinstance(self.search_space, dict):
            arguments['config_dict'] = self.search_space

        # Limit time if passed
        if limit_time is not None:
            arguments['max_time_mins'] = limit_time

        if self.evaluation_metric is not None:
            arguments['scoring'] = self.evaluation_metric

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
        """Score a validation set against the discovered pipeline.

        Args:
            x_val (numpy.array): The validation set for the features.
            y_val (numpy.array): The validation set for the target.

        Returns:
            float: The score given by the pipeline for the passed set.

        """
        self.validation_score = self._tpot_optimizer.score(x_val, y_val)
        return self.validation_score

    def save_pipeline(self, target_dir=None, file_name=None):
        """Save the discovered pipeline into a file, as python code.

        Args:
            target_dir (string): If not none, use it as parent dir for the
                resulting file. Defaults to None.
            file_name (string): The name to use for the resulting file.
                Defaults to None.

        """
        if file_name is None:
            file_name = "{data_id}.py".format(data_id=self.dataset.dataset_id)

        if target_dir is not None:
            file_name = "{dirname}/{basename}".format(dirname=target_dir,
                                                      basename=file_name)

        self._tpot_optimizer.export(file_name)

    @property
    def tpot_object(self):
        """Return the TPOT object used in the discovery process.

        Returns:
            TPOTBase: The TPOTBase class (TPOTClassifier or TPOTRegressor).

        """
        return self._tpot_optimizer

    @property
    def pipeline(self):
        """Return the resulting pipeline from the discovery process.

        Returns:
            sklearn.pipeline.Pipeline: The discovered pipeline if any.

        """
        if self._tpot_optimizer is None:
            return None

        return self._tpot_optimizer.fitted_pipeline_
