import arff
from automl.utl.miscellaneous import argsort_list
import numpy as np
import pandas as pd

class ARFFWrapper:
    
    def __init__(self, arff_dataset = None, arff_filepath = None, 
            sort_attributes = False, sort_attr_backwards = False):

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
        # self.sort_attributes(sort_attr_backwards, True)

    def load_dataset(self, path):
        self._arff_dataset = arff.load(open(path))
        self._validate_arff_dataset()

    def _init_attributes(self, arff_dataset):
        self.description = arff_dataset['description']
        self.name = arff_dataset['relation']
        self.data = self._parse_data(arff_dataset)
        self.key_attributes = [arff_dataset['attributes'][0][0]]

    def _parse_data(self, arff_dataset):
        return pd.DataFrame(arff_dataset['data'], 
                    columns = self._attributes_arff_as_list(arff_dataset))

    def _attributes_arff_as_list(self, arff_dataset):
        return [item[0] for item in arff_dataset['attributes']]
    
    def _validate_arff_dataset(self):
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
        return self.data.drop(labels = attributes, axis = 1, inplace = inplace,
                                errors = 'ignore')

    def attribute_names(self):
        return list(self.data.columns)

    def attribute_types(self):
        return list(self.data.dtypes)

    def add_attribute_key(self, column = None):
        if column not in self.key_attributes and column is not None:
            self.key_attributes.append(column)
        # TODO: log a warning 
    
    def change_attribute_type(self, column, newtype):
        self.data[[column]] = self.data[[column]].astype(newtype)

    def values_by_attribute(self, column = None):
        if isinstance(column, str) or isinstance(column, int):
            # TODO: check for columns not available
            return self.data[column].values
        return None

    def summary(self):
        print("Dataset name: {name}".format(name = self.name))
        print("Description: {desc}".format(desc = self.description))
        print("Number of features: {m}".format(m = self.data.shape[1]))
        print("Number of observations: {n}".format(n = self.data.shape[0]))
        print("First 5 observations:")
        print(self.data.head())

    def as_numpy_array(self):
        return self.data.values
    
    def as_pandas_df(self):
        return self.data
    
    def shape(self):
        return self.data.shape

    def copy(self):
        return ARFFWrapper(arff_dataset = self.arrf_dict())
