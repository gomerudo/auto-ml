"""Module containing the classes to allow for easy discovery."""

from automl.errors.customerrors import AutoMLError
from automl.metalearning.metafeatures.metafeatures_interaction \
    import MetaFeaturesManager
from automl.metalearning.database.load_db import MKDatabaseClient
from automl.discovery.pipeline_generation import PipelineDiscovery


class Assistant:
    """Provide methods to return AutoML results.

    This class can compute the metafeatures, similar datasets and discover
    pipelines. For this, a metric and a dataset are required.

    Args:
        dataset (Dataset): A dataset object to work with. Defaults to None.
        metric (string or callable): A metric to evalate the pipeline against.
            Defaults to 'accuracy'.

    Attributes:
        dataset (Dataset): A dataset object to work with.
        metric (string or callable): A metric to evalate the pipeline against.

    """

    def __init__(self, dataset=None, metalearning_metric='accuracy',
                 evaluation_metric='accuracy'):
        """Constructor."""
        self.dataset = dataset
        self.evaluation_metric = evaluation_metric

        # TODO: validate a valid metalearning_metric
        self.metalearning_metric = metalearning_metric

        # Private variables to hide logic
        self._neighbors = None
        self._neighbors_metrics = None
        self._search_space = None
        self._pipeline_discovery = None

    @property
    def metafeatures(self):
        """Return the metafeatures for the dataset attribute of the class.

        Returns
            numpy.array: The metafeatures computed in the form of an array.
                They are sorted alphabetically by its name.

        """
        metafeat_manager = MetaFeaturesManager(self.dataset)
        return metafeat_manager.metafeatures_as_numpy_array()

    # TODO: Allow for other metrics for the similarity
    def compute_similar_datasets(self, k=5, similarity_metric='minkowski'):
        """Compute the similar datasets based on the dataset's metafeatures.

        Args:
            k (int): The number of similar datasets to retrieve. Defaults to 5.

        Returns:
            list: List of neighbors ordered by similarity.
            list: List of the metrics for each of the neighbors.

        """
        mk_dc = MKDatabaseClient()
        similarities, dataset_ids = \
            mk_dc.nearest_datasets(self.dataset, k=k, weighted=False,
                                   distance_metric=similarity_metric)

        # Always set to None cause it has changed. Only recompute if needed.
        self._search_space = None

        self._neighbors = dataset_ids[0]
        self._neighbors_metrics = similarities[0]
        return self._neighbors, self._neighbors_metrics

    @property
    def similar_datasets(self):
        """Retrieve the similar datasets, without recomputing.

        AutoMLError is thrown if no neighbors have been computed yet.

        Returns:
            tuple: A 2-tuple where the first element is the set of similar
                datasets and the second is the metric (distance) between our
                dataset and the similar ones.

        """
        if self._neighbors is None:
            raise AutoMLError("No neighbors available. Call the \
                               compute_similar_datasets method first")

        return self._neighbors, self._neighbors_metrics

    @property
    def reduced_search_space(self):
        """Retrieve the reduced search space based on the similar datasets.

        The similar datasets should have been computed already. AutoMLError is
        thrown if no neighbors have been computed yet.

        Returns:
            list: List of MLSuggestions.

        """
        if self._neighbors is None:
            raise AutoMLError("No neighbors available. Call the \
                               compute_similar_datasets method first")

        mk_dc = MKDatabaseClient()
        return mk_dc.meta_suggestions(
            dataset=self.dataset,
            ids_list=list(self._neighbors),
            metric=self.metalearning_metric
        )

    # TODO: This is the method that should call TPOT using the search space
    # We should consider parameters to force and also to handle TPOTs features
    def generate_pipeline(self, ignore_similar_datasets=False):
        """Automatically generate a pipeline using the dataset and metric.

        If the similar datasets have been already been computed and the default
        value of ignore_similar_datasets is kept, then the search space of
        classifiers is reduced to the suggestions for the similar datasets.

        Args:
            ignore_similar_datasets (bool): Whether to ignore the suggested
                models for the similar datasets or not. Defaults to False.

        Returns:
            PipelineDiscovery: The PipelineDiscoverObject used to find out the
                dataset.

        """
        # Call TPOT
        # Returns a pipeline
        dict_suggestions = self.reduced_search_space.classifiers

        try:
            search_space = dict()
            for classifier in dict_suggestions:
                search_space[classifier] = {}
        except AutoMLError:
            search_space = None

        if ignore_similar_datasets:
            p_disc = PipelineDiscovery(self.dataset)
        else:
            p_disc = PipelineDiscovery(
                dataset=self.dataset,
                search_space=search_space,
                evaluation_metric=self.evaluation_metric
            )

        pipeline = p_disc.discover()

        # TODO: Think of print options.
        print(pipeline)
        print(p_disc.validation_score)
        return p_disc
