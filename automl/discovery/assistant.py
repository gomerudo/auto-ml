"""Module containing the classes to allow for pipelines discovery."""

from automl.errors.customerrors import AutoMLError
from automl.metalearning.metafeatures.metafeatures_interaction \
    import MetaFeaturesManager


class Assistant:

    def __init__(self, dataset=None, metric="accuracy"):
        self.dataset = dataset
        self.metric = metric

        # Private variables to hide logic
        self._neighbors = None
        self._search_space = None

    @property
    def metafeatures(self):
        metafeat_manager = MetaFeaturesManager(self.dataset)
        return metafeat_manager.metafeatures_as_dict()

    def compute_similar_datasets(self, k=5):
        # 1. Compute the metafeatures
        dataset_mf = self.metafeatures

        
        # Always set to None cause it has changed. Only recompute if needed.
        self._search_space = None

    @property
    def similar_datasets(self):
        # TODO: Log a warning or raise a warning if none
        if self._neighbors is None:
            raise AutoMLError("You need to call the compute_similar_datasets \
                              method first")
        return self._neighbors

    @property
    def reduced_search_space(self):
        # TODO: if no neighbors, then exception, otherwise, provide the reduced
        # search space by combining the MLSuggestions
        if self._neighbors is None:
            raise AutoMLError("You need to call the compute_similar_datasets \
                              method first")
        pass

    # TODO: This is the method that should call TPOT using the search space
    # We should consider parameters to force and also to handle TPOTs features
    def generate_pipeline(self):
        # Call TPOT
        # Returns a pipeline
        pass

