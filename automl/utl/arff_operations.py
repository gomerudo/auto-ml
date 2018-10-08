"""Define common ARFFWrapper operations, abstracted as classes.

The MetaDB works with ARFF files. These, are expressive but not that straight-
forward to work with. The intention here is then to provide classes that, given
a dataset, perform some basic/common operations and can map the object into 
numpy/pandas objects that are easier to manipulate: for instance, perform matrix
operations with numpy or rows/columns manipulation with pandas.
"""

import arff
from automl.utl.miscellaneous import argsort_list
import numpy as np
import pandas as pd

class ARFFWrapper:
    
    def __init__(self, arff_dataset = None, arff_filepath = None, 
            sort_attributes = False, sort_attr_backwards = False):
        """Constructor for ARFFWrapper.

        It can accept an arff_dataset or an arff_filepath.

        Attributes:
            dict: arff_dataset. A valid ARFF dataset.
            str: arff_filepath. A valid ARFF filepath.
            bool: sort_attributes. Wheter or not to sort the columns 
                  (attributes) of the dataset. This will also sort the data.
            bool: sort_attr_backwards. If sort_attributes is true, this
                  parameter controls the reverse/natural order of the sorting.
        """

        # Make decision: 
        #  if arff_filepath is provided, we load it and ignore arff_dataset
        if arff_filepath is not None:
            self.arff_filepath = arff_filepath
            self.load_dataset(self.arff_filepath)
        # otherwise, we just load the arff_dataset
        else:
            self._arff_dataset = arff_dataset
            self._validate_arff_dataset()
        
        # Check dataset is instance of ARFF
        if not isinstance(self._arff_dataset, dict):
            raise TypeError("ARFF dataset must be a dictionary")

        # Initialize the attributes
        self.data = None
        self.description = None
        self.name = None
        self.key_attributes = None

        self._init_attributes(self._arff_dataset)

        # If requested, sort the attributes - this overrides _columns_sorted_by_method
        self.sort_attributes(sort_attr_backwards, True)

    def load_dataset(self, path):
        """Load the dataset from a ARFF file.

        It loads the file and validate that the resulting arff dict is valid.
        """
        self._arff_dataset = arff.load(open(path))
        self._validate_arff_dataset()

    def _init_attributes(self, arff_dataset):
        """Initialize the attributes of the class.

        We map each of the arff dict keys into an attribute.
        """
        self.description = arff_dataset['description']
        self.name = arff_dataset['relation']
        self.data = self._parse_data(arff_dataset)
        self.key_attributes = [arff_dataset['attributes'][0][0]]

    def _parse_data(self, arff_dataset):
        """Get the arff data as a pandas dataframe for internal usage"""
        return pd.DataFrame(arff_dataset['data'], 
                    columns = self._attributes_arff_as_list(arff_dataset))

    def _attributes_arff_as_list(self, arff_dataset):
        """Build a list from the names specified in dict['attributes'].
        """
        return [item[0] for item in arff_dataset['attributes']]
    
    def _validate_arff_dataset(self):
        """Check that the arff dict is valid.

        In order to be valid, an arff dict must have the next structure:

        { data: list , attributes: list, relation: str, description : str }
        """
        if not self._arff_dataset:
            return False # TODO: Use exception instead to be concordant

        keys =  ['data', 'attributes', 'relation', 'description']
        types = [list, list, str, str]

        # Validate that keys exist
        for key in keys:
            if key not in self._arff_dataset.keys():
                raise KeyError("The key '{key}' is needed for an ARFF object".
                    format(key = key))
                # TODO: Or return false ...

        # Validate type for each key value
        for key, expected_type in zip(keys, types):
            if type(self._arff_dataset[key]) is not expected_type:
                raise TypeError("The expected type for value in '{key}' field \
is '{type}'".format(key = key, type = expected_type))

    def arrf_dict(self):
        result = dict()
        # TODO: Fix attributes
        result['attributes'] = list(zip(self.attribute_names(), self.attribute_types()))
        result['description'] = self.description
        result['relation'] = self.name
        result['data'] = self.data.values.tolist()

        return result

    def sort_rows(self, columns = [], ascending = True, inplace = True):
        """"Sort the rows, similary to pandas dataframe sorting.
        
        Attributes:
            list: columns. Columns (in order) to sort with.
            bool: ascending. Whether or not to sort in ascending order.
            bool: inplace. Whether or not to perform the change in place.

        Returns:
            The self object if inplace = True. A copy with the modifications,
            othewise.
        """
        if inplace:
            self.data.sort_values(by = columns, axis = 0, ascending = ascending,
                                    inplace = inplace)
            return self
        else:
            copy = self.copy()
            copy.data.sort_values(by = columns, axis = 0, ascending = ascending,
                                    inplace = (not inplace))
            return copy

    def sort_attributes(self, backwards = False, inplace = True):
        """Sort the columns in the dataset.

        Attributes:
            bool: backwards. Whether to apply reverse order or not.
            bool: inplace. Whether to perform the mutation inplace or not.

        Returns:
            The self object after changes if inplace was specified. A copy with
            the modification otherwise.
        """
        if inplace:
            self.data.sort_index(axis = 1, ascending = (not backwards), 
                                        inplace = inplace)
            return self
        else:
            copy = self.copy()
            copy.data.sort_index(axis = 1, ascending = (not backwards), 
                                        inplace = (not inplace))
            return copy

    def drop_attributes(self, attributes = [], inplace = True):
        """Drop a column (attribute) from the dataset.

        Attributes:
            list: attributes. List of attributes to drop off.
            bool: inplace. Whether to perform the drop inplace or not.

        Returns:
            pd.DataFrame: the dataframe after dropping the attribute.
        """
        
        # TODO: return a self object.
        return self.data.drop(labels = attributes, axis = 1, inplace = inplace,
                                errors = 'ignore')

    def attribute_names(self):
        """Returns a list with the names of the attributes/features/columns."""
        return list(self.data.columns)

    def attribute_types(self):
        """Returns a list with the attribute types (ordered as the columns are.
        """
        return list(self.data.dtypes)

    def add_attribute_key(self, column = None):
        """Add an attribute to the list of keys.
        
        A key is the identifier in the dataset, similar to what we call primary 
        key in Databases.
        """
        if column not in self.key_attributes and column is not None:
            self.key_attributes.append(column)
        # TODO: log a warning 
    
    def change_attribute_type(self, column, newtype):
        """Changes the dtype of a column.
        
        Attributes:
            column: str or int with the column we want to work with.
            newtype: type to assign to the column dtype.
        """

        # TODO: Check for isinstance and newtype is a type
        self.data[[column]] = self.data[[column]].astype(newtype)

    def values_by_attribute(self, column = None):
        """Gives the values in one column (attribute).

        Attributes:
            column: str or int to index the dataset. This is the column we want
                    to retrieve.
        Returns:
            np.array: The values in the specified column.
        """
        if isinstance(column, str) or isinstance(column, int):
            # TODO: check for columns not available
            return self.data[column].values
        return None

    def summary(self):
        """Prints a summary of the dataset.

        It shows basic information: name, description, shape and first 5
        observations.
        """
        print("Dataset name: {name}".format(name = self.name))
        print("Description: {desc}".format(desc = self.description))
        print("Number of features: {m}".format(m = self.data.shape[1]))
        print("Number of observations: {n}".format(n = self.data.shape[0]))
        print("First 5 observations:")
        print(self.data.head())

    def as_numpy_array(self):
        """Returns the data as a numpy array"""
        return self.data.values
    
    def as_pandas_df(self):
        """Returns the data as a pandas data frame."""
        return self.data
    
    def shape(self):
        """Returns a tuple with the shape of the data.

        Returns:
            tuple: 2-tuple with first element being `n` and second `m`.
        """
        return self.data.shape

    def copy(self):
        """Creates a copy of this object.
        """
        return ARFFWrapper(arff_dataset = self.arrf_dict())
