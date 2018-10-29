"""Read ARFF files and provide an interface to interact with them.

The current MetaDatabase (the place where the meta-knowledge acquired by
auto-sklearn is stored) is based on ARFF files. These, are expressive but not
straightforward to manipulate neither for matrices operations nor data access.

The intention here is to provide classes that can perform some basic/common
operations and can hide the object as numpy/pandas objects that are easier to
manipulate for data science purposes.
"""

import pandas as pd
import arff
import automl


class ARFFWrapper:
    """Class to perform operations on a ARFF dataset.

    Since performing matrix operations or even data manipulation with ARFF
    datasets in python is not an straightforward operation, we expose this
    wrapper that will perform some of these operations.

    Args:
        arff_dataset (dict): The dataset as an ARFF dictionary.
            If `arrff_filepath` is not None, `arff_dataset` is ignored.
        arff_filepath (str): The filepath used to load the dataset. `None` if
            is not desired to load from file.
        sort_attributes (bool): Wheter or not to sort the columns after loading
            the dataset. Defaults to False.
        sort_attr_backwards (bool): If `sort_attributes` is True, this argument
            controls the reverse/natural order of the sorting.
            Defaults to False (non-reverse).

    Raises:
        KeyError: When the dictionary describing the ARFF dataset does not
            contain the official fields: `data`, `relation`, `attributes` and
            `description`.
        TypeError: When the values for each of the official fields in the ARFF
            dataset are not as expected: `list` for `data and `attributes` and
            `str` for `relation` and `description`.
        TypeError: When `arff_dataser` is not a dictionary.
        ValueError: If the path argument is `None` and `arff_dataset` is `None`
            (no dataset passed).

    Attributes:
        data (dict): The `data` field in an ARFF dataset.
        description (str): The `description` as in the original ARFF dataset.
        name (str): Corresponds to the `relation` field in the original ARFF
            dataset.
        key_attributes (list): The names for the columns (attributes) in the
            ARFF dataset.

    """

    def __init__(self, arff_dataset=None, arff_filepath=None,
                 sort_attributes=False, sort_attr_backwards=False):
        """Constructor.

        It can accept an arff_dataset or an arff_filepath.

        Attributes:

        """
        # Make decision:
        #  if arff_filepath is provided, we load it and ignore arff_dataset
        if arff_filepath is not None:
            self.load_dataset(arff_filepath)
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

        # If requested, sort the attributes.
        if sort_attributes:
            self.sort_attributes(sort_attr_backwards, True)

    def load_dataset(self, path=None):
        """Load the dataset from a ARFF file.

        It loads the file and validates that the resulting arff dict is valid.
        If a dataset had already been loaded, then this operation overrides it.

        Args:
            path (str): The path where the file to load is located. Defaults to
            `None`.

        Raises:
            KeyError: When the dictionary describing the ARFF dataset does not
                contain the official fields: `data`, `relation`, `attributes`
                and `description`.
            TypeError: When the values for each of the official fields in the
                ARFF dataset are not as expected: `list` for `data and
                `attributes`, and `str` for `relation` and `description`.
            ValueError: When the path argument is `None`.

        """
        if path is None:
            raise ValueError("Invalid path. Path cannot be None")

        self._arff_dataset = arff.load(open(path))
        self._validate_arff_dataset()

    def _init_attributes(self, arff_dataset):
        """Initialize the attributes of the class.

        We map each of the arff dict keys into an attribute.
        """
        self.description = arff_dataset['description']
        self.name = arff_dataset['relation']
        self.data = self._parse_data(arff_dataset)
        if arff_dataset['attributes']:
            if isinstance(arff_dataset['attributes'][0], tuple):
                self.key_attributes = [arff_dataset['attributes'][0][0]]
            else:
                automl.automl_log(
                    "First element in 'attributes' is not a tuple'. Skipping \
'key_attributes' assignment.",
                    'WARNING'
                )
        else:
            automl.automl_log(
                "'attributes' field is an empty list. Errors may occur. \
Skipping 'key_attributes' assignment.",
                'WARNING'
            )

    def _parse_data(self, arff_dataset):
        """Get the arff data as a pandas dataframe for internal usage."""
        return pd.DataFrame(
            arff_dataset['data'],
            columns=self._attributes_arff_as_list(arff_dataset)
        )

    def _attributes_arff_as_list(self, arff_dataset):
        """Build a list from the names specified in dict['attributes']."""
        return [item[0] for item in arff_dataset['attributes']]

    def _validate_arff_dataset(self):
        """Check that the arff dict is valid.

        In order to be valid, an arff dict must have the next structure:

        { data: list , attributes: list, relation: str, description : str }
        """
        if self._arff_dataset is None:
            raise ValueError("No ARFF dataset has been loaded.")

        keys = ['data', 'attributes', 'relation', 'description']
        types = [list, list, str, str]

        # Validate that keys exist
        for key in keys:
            if key not in self._arff_dataset.keys():
                raise KeyError("The key '{key}' is needed for an ARFF object".
                               format(key=key))

        # Validate type for each key value
        for key, expected_type in zip(keys, types):
            # pylint: disable=C0123
            if type(self._arff_dataset[key]) is not expected_type:
                raise TypeError("The expected type for value in '{key}' field \
is '{type}'".format(key=key, type=expected_type))

    def arrf_dict(self):
        """Return the dataset as an arff dictionary.

        Returns:
            dict:   The resulting ARFF dataset in the form of a dictionary.

        """
        result = dict()
        # TODO: Fix attributes types for ARFF
        result['attributes'] = list(zip(self.attribute_names(),
                                        self.attribute_types()))
        result['description'] = self.description
        result['relation'] = self.name
        result['data'] = self.data.values.tolist()

        return result

    def sort_rows(self, columns=None, ascending=True, inplace=True):
        """Sort the rows, similarly to pandas dataframe sorting.

        Args:
            columns (list): Columns (ordered) to sort by.
                Defaults to `None`.
            ascending (bool): Whether or not to sort in ascending order.
                Defaults to `True`.
            inplace (bool): Whether or not to perform the change in place.
                Defaults to `True`.

        Raises:
            ValueError: If a value in `columns` is not part of the dataset or
                `None` is passed.

        Returns:
            ARFFWrapper: The self object if inplace = True. A copy with the
                modifications otherwise.

        """
        if columns is None:
            raise ValueError("Columns cannot be None")

        if not isinstance(columns, list):
            aux = []
            aux.append(columns)
            columns = aux

        for column in columns:
            if column not in self.attribute_names():
                raise ValueError("Column '{col}' not in the dataset. Indicate \
                                 a valid column".format(col=column))

        if inplace:
            self.data.sort_values(by=columns, axis=0, ascending=ascending,
                                  inplace=inplace)
            return self

        # Otherwise
        copy = self.copy()
        copy.data.sort_values(by=columns, axis=0, ascending=ascending,
                              inplace=(not inplace))
        return copy

    def sort_attributes(self, backwards=False, inplace=True):
        """Sort the columns in the dataset.

        Args:
            backwards (bool): Whether to apply reverse order or not. Defaults
                to `False`.
            inplace (bool): Whether to perform the mutation inplace or not.
                Defaults to `True`.

        Returns:
            ARFFWrapper: The self object after changes if inplace was specified
                or a copy with the modification otherwise.

        """
        if inplace:
            self.data.sort_index(axis=1, ascending=(not backwards),
                                 inplace=inplace)
            return self

        copy = self.copy()
        copy.data.sort_index(axis=1, ascending=(not backwards),
                             inplace=(not inplace))
        return copy

    def drop_attributes(self, attributes=None, inplace=True):
        """Drop a column (attribute) from the dataset.

        Args:
            attributes (bool): List of attributes to drop off. Defaults to
                `None`.
            inplace (bool): Whether to perform the drop inplace or not.
                Defaults to `True`.

        Raises:
            ValueError: If an attribute in `attributes` is not part of the
                dataset columns or `None` is passed.

        Returns:
            pandas.DataFrame: The dataframe after dropping the attribute.

        """
        if attributes is None:
            raise ValueError("attributes cannot be None")

        if not isinstance(attributes, list):
            aux = []
            aux.append(attributes)
            attributes = aux

        for attr in attributes:
            if attr not in self.attribute_names():
                raise ValueError("Column '{col}' not in the dataset. Indicate \
                                 a valid column".format(col=attr))

        if inplace:
            self.data.drop(labels=attributes, axis=1, inplace=inplace,
                           errors='ignore')
            return self

        copy = self.copy()
        copy.data.drop(labels=attributes, axis=1, inplace=inplace,
                       errors='ignore')
        return copy

    def attribute_names(self):
        """Return a list with the names of the attributes/features/columns.

        Returns:
            list: A list with the names of the attributes, in the order they
                appear in the dataset.

        """
        return list(self.data.columns)

    def attribute_types(self):
        """Return a list with the attribute types.

        Returns:
            list: A list with the types of the attributes, in the order they
                appear in the dataset.

        """
        return list(self.data.dtypes)

    def add_attribute_key(self, column=None):
        """Add an attribute to the list of keys.

        A key is the identifier in the dataset, similar to what we call primary
        key in Databases.

        Args:
            column: The column to add as a key in the dataset. Defaults to
                `None`.

        """
        if column is None:
            raise ValueError("column cannot be None")

        if column not in self.attribute_names():
            raise ValueError("Invalid column. Column is not in the dataset")

        if column in self.key_attributes:
            log_msg = "Column '{col}' already existed in key_attributes. \
                      Skipping ...".format(col=column)

            automl.automl_log(log_msg, 'WARNING')
        else:
            self.key_attributes.append(column)

    def change_attribute_type(self, column=None, newtype=None):
        """Change the dtype of a column.

        Args:
            column (str or int): The column we want to work with, either its
                name or its index position. Defaults to `None`.
            newtype (type): The new type to assign as the column's dtype.
                Defaults to `None`.

        Raises:
            TypeError: When no str or int is used for `column` argument.
            ValueError: When column is `None`.
            IndexError: If column is instance of int and is out of bounds with
                respect to the number of columns (attributes) in the dataset.

        """
        if not isinstance(column, str) and not isinstance(column, int):
            raise TypeError("Only int or str can be passed as column value")

        if isinstance(column, str):
            if column not in self.attribute_names():
                raise ValueError("Invalid column. Column is not in the dataset\
                                 ")

        if isinstance(column, int):
            if column < 0 or column >= len(self.attribute_names()):
                raise IndexError("Passed column index is out of bounds")

        self.data[[column]] = self.data[[column]].astype(newtype)

    def values_by_attribute(self, column=None):
        """Return the data in one column (attribute).

        Args:
            column (str or int): This is the column we want to retrieve, either
                by its name or its positional index.

        Returns:
            numpy.array: The values in the specified column.

        """
        if not isinstance(column, str) and not isinstance(column, int):
            raise TypeError("Only int or str can be passed as column value")

        if isinstance(column, str):
            if column not in self.attribute_names():
                raise ValueError("Invalid column. Column is not in the dataset\
                                 ")

        if isinstance(column, int):
            if column < 0 or column >= len(self.attribute_names()):
                raise IndexError("Passed column index is out of bounds")

        return self.data[column].values

    def summary(self):
        """Print a summary of the dataset.

        It shows basic information: name, description, shape and first 5
        observations.
        """
        print("Dataset name: {name}".format(name=self.name))
        print("Description: {desc}".format(desc=self.description))
        print("Number of features: {m}".format(m=self.data.shape[1]))
        print("Number of observations: {n}".format(n=self.data.shape[0]))
        print("First 5 observations:")
        print(self.data.head())

    def as_numpy_array(self):
        """Return the data as a numpy array.

        Returns:
            numpy.array: The dataset as numpy object.

        """
        return self.data.values

    def as_pandas_df(self):
        """Return the data as a pandas data frame.

        Returns:
            pandas.DataFrame: The data as a pandas dataframe.

        """
        return self.data

    @property
    def shape(self):
        """Return a tuple with the shape of the data.

        Returns:
            tuple: 2-tuple with first element being `n` and second `m`.

        """
        return self.data.shape

    def copy(self):
        """Create a copy of this object."""
        return ARFFWrapper(arff_dataset=self.arrf_dict())
