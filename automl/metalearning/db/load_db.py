"""File defining classes for load the meta db. 

The MetaDB is the metalearning information for the 140+ datasets defined in the
auto-sklearn paper. These help us to build a space of datasets we can 
query against to obtain similar datasets and consequently, potential algorithms
to work with in a given dataset. 

In this implementation, we provide methods to retrieve the simple space and its
weighted version.
"""

from scipy.io import arff
import logging
import automl.globalvars
import os
import pkg_resources
from automl.utl.arff_operations import ARFFWrapper
import numpy as np

class LoadMetaDB:

    def __init__(self) :
        """Constructor.

        This constructor initializes private variables with the file paths for
        features_costs and feature_values. It does not load the content by 
        default: to achieve this, call the load_datasets_info.
        """

        # File paths in the default directory.
        self._costsFile = pkg_resources.resource_filename(__name__, 
                    "files/feature_costs.arff")
        self._featuresFile = pkg_resources.resource_filename(__name__, 
                    "files/feature_values.arff")
        # Initialize data to None
        self.features = None
        self.costs = None

    def _load_feature_costs(self):
        """Load the feature costs file.

        Creates an ARFFWrapper object with the content of the costs file.
        """
        return ARFFWrapper(arff_filepath = self._costsFile)

    def _load_feature_values(self):
        """Load the feature values file.

        Creates an ARFFWrapper object with the content of the values file.
        """
        return ARFFWrapper(arff_filepath = self._featuresFile)

    def load_datasets_info(self):
        """Load both the costs and values file for the features in the meta db.
        
        It will initialize the features/costs objects with ARFFWrapper objs.

        Returns: 
            LoadMetaDB: Self object with the initialized features/costs
        """
        if not self.features and not self.costs:
            logging.debug("Loading feature values and costs")
            
            # Load the objects
            self.features = self._load_feature_values()
            self.costs = self._load_feature_costs()

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

            # We finally assert that there are no difference between both id's
            # sets.
            assert(len(set(f_instid).symmetric_difference(c_instid)) == 0)
        
        # And return the self object with the initialized values.
        return self

    def simple_matrix(self):
        """Returns the feature values in the meta db - without the weights.

        It will simple

        Returns:
            np.array: The ordered list of instance_id's for the features
            np.darray: The matrix with the features values. Each row corresponds
            to the id in the first return value.
        """

        # Get the instance's ids
        f_instid = self.features.values_by_attribute('instance_id')
        
        # Create a copy of the matrix and drop the instance_id column for
        # simplicity.
        aux_f = self.features.copy()
        aux_f.drop_attributes('instance_id')

        return f_instid, aux_f.as_numpy_array()

    def weighted_matrix(self):
        """Returns a matrix with the weighted (costs) features in the meta db.

        Returns:
            np.array: The ordered list of instance_id's for the features
            np.darray: The matrix with the weighted features values. Each row 
            corresponds to the id in the first return value.
        """
        # TODO: Check None values, sizes and so and throw exceptions
        # Then obtain the numpy version and make the multiplication

        # Get the instance's ids
        f_instid = self.features.values_by_attribute('instance_id')
        
        # Get a copy of the data and drop the instance_id attribute in both
        # costs and values, for simplicity.
        aux_f = self.features.copy()
        aux_f.drop_attributes('instance_id')    
        aux_c = self.costs.copy()
        aux_c.drop_attributes('instance_id')
        
        # Multiply, element-wise, both matrices.
        res = np.multiply(aux_f.as_numpy_array(), aux_c.as_numpy_array())
        
        # Return a tuple, with the id's and the weighted matrix.
        return f_instid, res
