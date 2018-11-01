"""File defining classes to load the meta db and interact with it.

The MetaDB is the metalearning information for the 140+ datasets defined in the
auto-sklearn paper. These help us to build a space of datasets we can
query against to obtain similar datasets and consequently, potential algorithms
to work with in a given dataset.

In this implementation, we provide methods to retrieve the simple space and its
weighted version.
"""


import logging
import os
import os.path
import re
import pkg_resources
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from automl.metalearning import CONFIGURATIONS_CSV_NAME
from automl.metalearning import ALGORUNS_CSV_NAME
from automl.utl.arff_operations import ARFFWrapper
from automl.errors.customerrors import CurrentlyNonSupportedError
from automl.datahandler.dataloader import Dataset
from automl.metalearning.database.configurations_parsing \
    import ConfigurationBuilder
from automl.metalearning.database.configurations_parsing import mix_suggestions


class MKDatabaseClient:
    """MKDatabase (Meta-Knowledge Database) to perform queries.

    This class serves as a facade to interact with the Meta Knowledge. We would
    like to expose the next features:
    - Ability to find the nearest datasets given a metric
    - Reload the database (in case of any change at running time in the arff
    files).
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
        return self

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

        Returns:
            (np.array, np.array)    A tuple where the first element is a numpy
                                    array of the similarity metrics for the
                                    result datasets and the second element
                                    contains the similar dataset's ids.

        """
        # For now, accept only Dataset objects
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be an instance of AutoML 'Dataset'")

        # If k is greater than the maximum number of elements in the matrix
        max_neighbours = len(self.metaknwoledge.weighted_matrix()[0])
        if k > max_neighbours or k == 0:
            raise ValueError("The number of neighbors must be an 'int' in the \
invterval ({lower}, {upper}]".format(lower=0, upper=max_neighbours))

        # Otherwise ...
        nn_obj = NearestNeighbors(n_neighbors=k, metric=distance_metric)

        if weighted:
            dataset_ids, matrix = self.metaknwoledge.weighted_matrix()
        else:
            dataset_ids, matrix = self.metaknwoledge.simple_matrix()

        # Set nans to 0. TODO: Verify this makes sense and gives good results
        matrix[np.isnan(matrix)] = 0.0
        database = nn_obj.fit(matrix)

        similarities, indices = \
            database.kneighbors(dataset.metafeatures_vector())

        return similarities, dataset_ids[indices]

    def meta_suggestions(self, dataset=None, ids_list=None, metric='accuracy'):
        """Retrieve the Model Suggestions for a set of ids, based on a dataset.

        Using a given metric, retrieve the models suggested.

        Attributes:
            dataset     (Dataset) The dataset to use as areference.
            ids_list    (list) The list of ids to retrieve information about.
            metric      (str) A Metalearning metric.

        Returns:
            list:    A list of MLSuggestions.

        """
        configs = LandmarkModelParser.models_by_metric(ids_list, dataset,
                                                       metric)

        res = mix_suggestions(configs)
        return res


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
            for col in cols_diff:
                flag = False

                try:
                    self.features.drop_attributes(col)
                except ValueError:
                    flag = True
                try:
                    self.costs.drop_attributes(col)
                    flag = flag if not flag else not flag
                except ValueError:
                    flag = True

                if flag:
                    raise ValueError("Column '{}' is not in any of the \
                    feature's sets".format(col))

            # Fix the types of instance_id - otherwise the sort won't work
            self.features.change_attribute_type('instance_id', int)
            self.costs.change_attribute_type('instance_id', int)

            # Then we sort by instance_id int representation
            self.features.sort_rows('instance_id')
            self.costs.sort_rows('instance_id')

            # Resort, just in case ...
            f_cols = self.features.attribute_names()
            c_cols = self.costs.attribute_names()

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


class LandmarkModelParser:
    """Class to interact with the models stored per instance (dataset)."""

    @staticmethod
    def models_by_metric(instances_ids=None, dataset=None,
                         metric='accuracy'):
        """Return the models for a list of instances by the given accuracy.

        Attributes:
            instances_ids   (list) List of integers with the ids of the
                            instances (datasets).
            dataset         The dataset to work with.
            metric          (str) Name of the metric to use. It must be one of
                            the metrics returned by
                            LandmarkModelParser.metrics_available().
        Results:
            list: List of models. One element per instance.

        """
        # Validation of arguments
        if instances_ids is None:
            raise ValueError("A list of instances' ids must be specified.")

        if dataset is None:
            raise ValueError("Please provide a valid dataset.")

        if not isinstance(dataset, Dataset):
            raise TypeError("Dataset must be of type Dataset (automl pkg)")

        if dataset.is_regression_problem():
            raise CurrentlyNonSupportedError("Meta-learning for regression is \
                                             not supported yet")

        # Create helper variables
        if dataset.is_classification_problem():
            problem_type = "classification"
            classif_type = "multiclass" if dataset.n_labels > 2 else "binary"
        # sparse or not
        data_type = "sparse" if dataset.is_sparse() else "dense"

        # metric to use is composed of `metric`_`binary/multiclass`. e.g.
        # accuracy_binary
        internal_metric = "{me}_{c_type}".format(me=metric,
                                                 c_type=classif_type)
        # problem_description is classification_`sparse/dense`. E.g.
        # classficiation_sparse
        problem_desc = "{p_type}_{d_type}".format(p_type=problem_type,
                                                  d_type=data_type)

        # Then the final basename_dir (name of the metric in auto-sklearn) is
        # the mix of the above. E.g. accuracy_binary_classficiation_sparse
        basename_dir = \
            "{metric}.{problem}".format(metric=internal_metric,
                                        problem=problem_desc)

        # Get the available metrics to validate the resolved metric is part of
        # list
        metrics_available = LandmarkModelParser.metrics_available()
        if internal_metric not in metrics_available:
            raise ValueError("Metric '{argument}' is not supported. Try any \
                             of the following metrics: {available}".format(
                                 argument=metric,
                                 available=metrics_available
                             ))

        # Get the corresponding configurations.csv file path
        configs_csv = LandmarkModelParser._configs_file_by_metric(basename_dir)
        # Get the corresponding algorithm_runs.arff file path
        algoruns_arff = \
            LandmarkModelParser._algorithm_runs_file_by_metric(basename_dir)

        # Validate we found valid files
        if configs_csv is None or algoruns_arff is None:
            raise ValueError("Some of the meta-learning files was not found \
in the database for metric '{metric}'".format(metric=basename_dir))

        # Start to resolve the result
        res = []
        # For each of the instances requested
        for instance_id in instances_ids:
            # instanciate the correspondant algorithm runs file
            algoruns_file = AlgorithmRunsFile(algoruns_arff)
            # get the configuration id for that instance
            config_id = \
                algoruns_file.get_associated_configuration_id(instance_id)

            # And then load the configurations.csv file
            config_file = ConfigurationsFile(configs_csv)

            # Resolve the configuration as a list
            mmb = ConfigurationBuilder(
                config_file.get_configuration(config_id)
            )
            # and append it to the list
            res.append(mmb.build_configuration())
        return res

    @staticmethod
    def _algorithm_runs_file_by_metric(metric):
        files_dir = pkg_resources.resource_filename(__name__, "files")
        algoruns_file = "{f_dir}/{m_name}/{c_file}".format(
            f_dir=files_dir,
            c_file=ALGORUNS_CSV_NAME,
            m_name=metric)

        if os.path.exists(algoruns_file):
            return algoruns_file
        return None

    @staticmethod
    def _configs_file_by_metric(metric):
        files_dir = pkg_resources.resource_filename(__name__, "files")
        config_file = "{f_dir}/{m_name}/{c_file}".format(
            f_dir=files_dir,
            c_file=CONFIGURATIONS_CSV_NAME,
            m_name=metric)

        if os.path.exists(config_file):
            return config_file
        return None

    @staticmethod
    def metrics_available():
        """Return the metrics that are available in the meta-knowledge.

        Returns:
            list:   Metrics available in the package's local storage.

        """
        files_dir = pkg_resources.resource_filename(__name__, "files")
        metric_pattern = re.compile("^(\w+)\..+$")  # pylint: disable=W1401

        list_available = []
        for directory in os.walk(files_dir):
            dir_basename = os.path.basename(directory[0])
            res_re = metric_pattern.match(dir_basename)
            if res_re is None:
                pass
            else:
                list_available.append(res_re.group(1))

        return list_available


class ConfigurationsFile:
    """Abstract a configurations.csv file.

    This class represents a configurations.csv file for our meta-knowledge. It
    also provides some useful methods to interact with the file.
    """

    def __init__(self, configs_file):
        """Constructor.

        Attributes:
            configs_file    (str): The configuration's file path.

        """
        self.configs_file = configs_file
        self._load_file()

    def _load_file(self):
        if os.path.exists(self.configs_file):
            self._frame = pd.read_csv(self.configs_file, index_col=0)
        else:
            raise ValueError("Invalid file path: File does not exist.")

    def get_configuration(self, algorithm_id):
        """Get the configurations for a given algorithm id.

        Attributes:
            algorithm_id    (int): The id for the algorithm. This should come
                            from the results in algorithm_runs.arff.

        """
        try:
            elem = self._frame.loc[algorithm_id].dropna()
            return elem
        except KeyError:
            raise ValueError("The algorithm_id={algo_id} was not found in \
file {file}".format(algo_id=algorithm_id, file=self.configs_file))

    def get_configurations(self, algorithms_ids):
        """Get the configurations for a given set of algorithm ids.

        Attributes:
            algorithm_ids   (list): The ids for the algorithms. These should
                            come from the results in algorithm_runs.arff

        """
        results = []
        for alg_id in algorithms_ids:
            results.append(self.get_configuration(alg_id))


class AlgorithmRunsFile:
    """Abstract the algorithm_runs.arff file."""

    def __init__(self, algruns_file):
        """Constructor.

        Arguments:
            algruns_file    (str): The algorithm_runs.arff file path.

        """
        self.algruns_file = algruns_file
        self._load_file()

    def _load_file(self):
        if os.path.exists(self.algruns_file):
            self._arrf_wrapper = ARFFWrapper(arff_filepath=self.algruns_file)
            self._arrf_wrapper.change_attribute_type('instance_id', int)
        else:
            raise ValueError("Invalid file path: File does not exist.")

    def get_associated_configuration_id(self, instance_id):
        """Get the associated configuration for a given instance id.

        A configuration is a solution for the dataset (instance). This returns
        the id for that solution, not the solution itself.

        Attributes:
            instance_id (int): The id of the dataset (instance) to search for.

        Returns:
            int:    The id of the configuration solving the instance_id problem
                    (dataset).

        """
        res = self._arrf_wrapper.row_by_column_value('instance_id',
                                                     instance_id)
        if res.empty:
            raise ValueError(
                "instance_id={val} is not in the database.".format(
                    val=instance_id
                )
            )
        return int(res['algorithm'])

    def get_associated_configuration_ids(self, instances_ids):
        """Get the associated configuration for a given set of instance ids.

        A configuration is a solution for the dataset (instance). This returns
        the ids for the solutions, not the solutions themselves.

        Attributes:
            instances_ids   (list): The ids of the datasets (instances) to
                            search for.

        Returns:
            list(int):  The ids of the configurations solving the instance_id's
                        (datasets) problems.

        """
        result = []
        for instance_id in instances_ids:
            result.append(self.get_associated_configuration_id(instance_id))
        return result
