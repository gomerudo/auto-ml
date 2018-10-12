"""File defining classes to load the meta db and interact with it.

The MetaDB is the metalearning information for the 140+ datasets defined in the
auto-sklearn paper. These help us to build a space of datasets we can
query against to obtain similar datasets and consequently, potential algorithms
to work with in a given dataset.

In this implementation, we provide methods to retrieve the simple space and its
weighted version.

Classes defined here:
    - MKDatabaseClient
    - MetaKnowledge
"""


import logging
import pkg_resources
import numpy as np
from sklearn.neighbors import NearestNeighbors
import automl.globalvars
from automl.utl.arff_operations import ARFFWrapper


class MKDatabaseClient:
    """MKDatabase (Meta-Knowledge Database) to perform queries.

    This class serves as a facade to interact with the Meta Knowledge. We would
    like to expose the next features:
        - Ability to find the nearest datasets given a metric
        - Reload the database (in case of any change at running time in the
          arff files).
    TODO // To consider (future implementation):
        - Ability to add observations into the database at running time, e.g.
          MKDatabase().add_dataset_metaknowledge()
    """

    def __init__(self):
        """Constructor.

        We just need to load the MetaKnowledge class.
        """
        self.metaknwoledge = MetaKnowledge().load_datasets_info()

    def reload(self):
        """Reload the metaknowledge object.

        This is helpful if any change is done at runtime in the metaknowledge
        files.
        """
        self.metaknwoledge = self.metaknwoledge.load_datasets_info()

    def nearest_datasets(self, dataset=None, k=5, weighted=False,
                         distance_metric='minkowski'):
        """Find the _k_ nearest datasets in the meta-knwonledge DB.

        This method finds the _k_ nearest neighbors for a given dataset, based
        on a given metric. This helps, for instance, to latter make the
        relation with the saved algorithms for each of the metrics in
        automl/metalearning/db/files.

        Attributes:
            dataset         (automl.datahandler.dataloader.Dataset): The
                            dataset to use. Default is None, which will cause
                            the method to fail.
            weighted        (bool): True if the costs should be used. Default
                            is False.
            k               (int) The number of neighbor datasets to retrieve.
            distance_metric (string or sklearn callable): The distance metric
                            to use for the KNN algorithm.

        """
        # For now, accept only Dataset objects
        if not isinstance(automl.datahandler.dataloader.Dataset):
            raise TypeError("dataset must be an instance of AutoML 'Dataset'")

        # Otherwise ...
        nn_obj = NearestNeighbors(n_neighbors=k, metric=distance_metric)

        if weighted:
            database = nn_obj.fit(self.metaknwoledge.weighted_matrix())
        else:
            database = nn_obj.fit(self.metaknwoledge.simple_matrix())

        similarities, indices = \
            database.kneighbors(dataset.metafeatures_vector())

        return similarities, indices

    # def plot_metaknowledge_space(self):
    #     pass


class MetaKnowledge:
    """This class is a representation of the feature's costs/values.

    The information comes from the meta-knowledge acquired by auto-sklearn. It
    provides a way to load the information in the form of ARFFWrapper objects
    and retrieve pandas and numpy representations of this information to ease
    1) interaction with the features and 2) matrix operations for more
    mathematical methods that are needed for machine learning.
    """

    def __init__(self):
        """Constructor.

        This constructor initializes private variables with the file paths for
        features_costs and feature_values. It does not load the content by
        default: to achieve this, call the load_datasets_info.

        Attributes:
            features    (ARFFWrapper) The meta-features of each dataset in the
                        form of a ARFF dataset. Each row is a dataset.
            costs       (ARFFWrapper) The costs for each dataset, in the form
                        of ARFF dataset that matches the features' shape.

        """
        # File paths in the default directory.
        self._costs_file = \
            pkg_resources.resource_filename(__name__,
                                            "files/feature_costs.arff")

        self._features_file = \
            pkg_resources.resource_filename(__name__,
                                            "files/feature_values.arff")

        # Initialize data to None
        self.features = None
        self.costs = None

    def _load_feature_costs(self):
        """Load the feature costs file.

        Creates an ARFFWrapper object with the content of the costs file.
        """
        return ARFFWrapper(arff_filepath=self._costs_file)

    def _load_feature_values(self):
        """Load the feature values file.

        Creates an ARFFWrapper object with the content of the values file.
        """
        return ARFFWrapper(arff_filepath=self._features_file)

    def load_datasets_info(self):
        """Load both the costs and values file for the features in the meta db.

        It will initialize the features/costs objects with ARFFWrapper objs.

        Returns:
            LoadMetaDB: Self object with the initialized features/costs.

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
            cols_diff.append('repetition')  # We dont care about this one

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
            assert not set(f_instid).symmetric_difference(c_instid)

        # And return the self object with the initialized values.
        return self

    def simple_matrix(self):
        """Return the feature values in the meta db - without the weights.

        Returns:
            np.array: The ordered list of instance_id's for the features
            np.darray: The matrix with the features values. Each row
            corresponds to the id in the first return value.

        """
        # Get the instance's ids
        f_instid = self.features.values_by_attribute('instance_id')

        # Create a copy of the matrix and drop the instance_id column for
        # simplicity.
        aux_f = self.features.copy()
        aux_f.drop_attributes('instance_id')

        return f_instid, aux_f.as_numpy_array()

    def weighted_matrix(self):
        """Return a matrix with the weighted (costs) features in the meta db.

        Returns:
            np.array: The ordered list of instance_id's for the features
            np.darray: The matrix with the weighted features values. Each row
            corresponds to the id in the first return value.

        """
        # TODO: Check None values, sizes and so and throw exceptions.

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
