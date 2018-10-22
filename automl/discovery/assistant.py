"""Module containing the classes to allow for pipelines discovery."""

import numpy as np
from automl.errors.customerrors import AutoMLError
from automl.metalearning.metafeatures.metafeatures_interaction \
    import MetaFeaturesManager
from automl.metalearning.db.load_db import MKDatabaseClient

class Assistant:

    def __init__(self, dataset=None, metric="accuracy"):
        self.dataset = dataset
        self.metric = metric

        # Private variables to hide logic
        self._neighbors = None
        self._neighbors_metrics = None
        self._search_space = None

    @property
    def metafeatures(self):
        metafeat_manager = MetaFeaturesManager(self.dataset)
        return metafeat_manager.metafeatures_as_numpy_array()

    def compute_similar_datasets(self, k=5):
        mk_dc = MKDatabaseClient()
        similarities, dataset_ids = \
            mk_dc.nearest_datasets(self.dataset, k=k, weighted=False)

        # Always set to None cause it has changed. Only recompute if needed.
        self._search_space = None

        self._neighbors = dataset_ids[0]
        self._neighbors_metrics = similarities[0]
        return self._neighbors, self._neighbors_metrics

    @property
    def similar_datasets(self):
        # TODO: Log a warning or raise a warning if none
        if self._neighbors is None:
            raise AutoMLError("No neighbors available. Call the \
                               compute_similar_datasets method first")

        return self._neighbors, self._neighbors_metrics

    @property
    def reduced_search_space(self):
        # TODO: if no neighbors, then exception, otherwise, provide the reduced
        # search space by combining the MLSuggestions
        if self._neighbors is None:
            raise AutoMLError("No neighbors available. Call the \
                               compute_similar_datasets method first")

        mk_dc = MKDatabaseClient()
        return mk_dc.meta_suggestions(self.dataset, list(self._neighbors))


    # TODO: This is the method that should call TPOT using the search space
    # We should consider parameters to force and also to handle TPOTs features
    def generate_pipeline(self):
        # Call TPOT
        # Returns a pipeline
        pass

