from scipy.io import arff
import logging
import automl.globalvars
import os
import pkg_resources
from automl.utl.arff_operations import ARFFWrapper
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
        return ARFFWrapper(arff_filepath = self._costsFile)

        # data = np.array(arff.load(open(self._costsFile))['data']) 
        # data[:,0] = data[:,0].astype('int')
        # data = data[np.argsort(data[:,0])]
        # return data # Sorted and casted to int in first column

    def _loadFeatureValues(self):
        return ARFFWrapper(arff_filepath = self._featuresFile)

        # data = np.array(arff.load(open(self._featuresFile))['data'])
        # data[:,0] = data[:,0].astype('int')
        # data = data[np.argsort(data[:,0])]
        # return data # Sorted and casted to int in first column

    def loadDatasetsInfo(self):
        if not self.features and not self.costs:
            logging.debug("Loading feature values and costs")
            
            # Load the objects
            self.features = self._loadFeatureValues()
            self.costs = self._loadFeatureCosts()

            # Sort the attribute names
            self.features.sort_attributes()
            self.costs.sort_attributes()

            # Get the names
            f_cols = self.features.attribute_names()
            c_cols = self.costs.attribute_names()

            # Get the difference between both lists
            cols_diff = list(set(f_cols).symmetric_difference(c_cols))
            cols_diff.append('repetition') # We dont care about this one

            # Drop everything that is not the intersection
            self.features.drop_attributes(cols_diff)
            self.costs.drop_attributes(cols_diff)

            # Fix the types of instance_id - otherwise the sort won't work
            self.features.change_attribute_type('instance_id', int)
            self.costs.change_attribute_type('instance_id', int)

            # Then we sort by instance_id int representation
            self.features.sort_rows('instance_id')
            self.costs.sort_rows('instance_id')

            # Verify that the instance_id's from both datasets are the same
            f_instid = self.features.values_by_attribute('instance_id')
            c_instid = self.costs.values_by_attribute('instance_id')

            assert(len(set(f_instid).symmetric_difference(c_instid)) == 0)
        return self

    def clean_matrix(self):
        f_instid = self.features.values_by_attribute('instance_id')
        
        aux_f = self.features.copy()
        aux_f.drop_attributes('instance_id')

        return f_instid, aux_f.as_numpy_array()

    def weighted_matrix(self):
        # TODO: Check None values, sizes and so and throw exceptions
        # Then obtain the numpy version and make the multiplication

        f_instid = self.features.values_by_attribute('instance_id')
        aux_f = self.features.copy()
        aux_c = self.costs.copy()
        
        aux_f.drop_attributes('instance_id')
        aux_c.drop_attributes('instance_id')

        res = np.multiply(aux_f.as_numpy_array(), aux_c.as_numpy_array())
        
        return f_instid, res
