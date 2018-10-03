from scipy.io import arff
import logging
import automl.globalvars
import os
import pkg_resources
import arff
import numpy as np

class LoadMetaDB:

    def __init__(self) :
        """
        """

        self._costsFile = pkg_resources.resource_filename(__name__, 
                    "files/feature_costs.arff")
        self._featuresFile = pkg_resources.resource_filename(__name__, 
                    "files/feature_values.arff")
        self.features = None
        self.costs = None

    def _loadFeatureCosts(self):
        data = np.array(arff.load(open(self._costsFile))['data']) 
        data[:,0] = data[:,0].astype('int')
        data = data[np.argsort(data[:,0])]
        return data # Sorted and casted to int in first column

    def _loadFeatureValues(self):
        data = np.array(arff.load(open(self._featuresFile))['data'])
        data[:,0] = data[:,0].astype('int')
        data = data[np.argsort(data[:,0])]
        return data # Sorted and casted to int in first column

    def loadDatasetsInfo(self):
        if not self.features and not self.costs:
            logging.debug("Loading feature values and costs")
            self.features = self._loadFeatureValues()
            self.costs = self._loadFeatureCosts()

    # def weightedVectors(self):
    #     # TODO: Check None values, sizes and so and throw exceptions

    #     # Assumption: lists share exactly the same IDs. 
    #     # result = self.features.copy()
    #     featuresAsMatrix = np.array(self.features)
    #     costsAsMatrix = np.array(self.costs)

        
    #     arff_file = arff.load(open(self._costsFile))
        
    #     print(result.shape)
    #     print(result)

    #     return result
        
        