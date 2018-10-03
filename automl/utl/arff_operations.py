import arff

class ARFFWrapper:
    
    
    def __init__(self, arff_dataset):
        self.arff_dataset = arff_dataset
        self._columns_sorted_by_method = False
        self._rows_sorted_by_method = False

        self.is_valid()

    def is_valid(self):
        self._validate_dataset()

    def _validate_dataset(self):
        keys =  ['data', 'attributes', 'relation', 'description']
        types = [list, list, str, str]

        # Validate that keys exist
        for key in keys:
            if key not in self.arff_dataset.keys():
                raise KeyError("The key '{key}' is needed for an ARFF object".
                    format(key = key))

        # Validate type for each key value
        for key, expected_type in zip(keys, types):
            if type(self.arff_dataset[key]) is not expected_type:
                raise TypeError("The expected type for value in '{key}' field \
is '{type}'".format(key = key, type = expected_type))


