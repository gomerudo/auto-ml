"""Test the utl package.

Special emphasis is put in the ARFFWrapper class that should correctly load the
meta-learning files.
"""

import unittest
import os.path
import numpy as np
import pandas as pd
from automl.utl.arff_operations import ARFFWrapper


class TestARFFWrapper(unittest.TestCase):
    """Test the main behavior of the ARFFWrapper."""

    def setUp(self):
        """Perform setup of the path for the ARFF test file."""
        self.automl_path = os.path.dirname(os.path.dirname(__file__))
        self.automl_path = os.path.join(self.automl_path,
                                        "automl",
                                        "metalearning",
                                        "database",
                                        "files",
                                        "feature_costs.arff")

    def test_invalid_keys_arff(self):
        """Test the validity check for *keys* in an ARFF file.

        We check that if an ARFF file with invalid keys is given, the parser
        will throw the correspondant exceptions. The valid keys are: `data`,
        `description`, `relation` and `attributes` - as described in:
        https://waikato.github.io/weka-wiki/arff_stable/
        """
        # Case 1: No keys are passed to the dictionary
        with self.assertRaises(KeyError):
            ARFFWrapper({})

        # Case 2: Only 1 of each are given to the dictionary.
        # TODO: a better test can be achieved if all possible combinations are
        # tested (e.g. using an stack and playing with the push/pop operations
        # can help to satisfy the requirement). This is still pending because
        # of time restrictions.
        with self.assertRaises(KeyError):
            ARFFWrapper({'data': None})

        with self.assertRaises(KeyError):
            ARFFWrapper({'description': None})

        with self.assertRaises(KeyError):
            ARFFWrapper({'relation': None})

        with self.assertRaises(KeyError):
            ARFFWrapper({'attributes': None})

    def test_invalid_values_arff(self):
        """Test the validity check of *values* in the ARFF file.

        If the keys of an ARFF are correct, but the values are not of the
        expected type, the TypeError should happen.
        """
        # Case 1: Everything is passed, but incorrect values are given for all
        # keys. Hence, it should fail with a TypeError.
        with self.assertRaises(TypeError):
            ARFFWrapper({
                'data': None,
                'description': None,
                'relation': None,
                'attributes': None
                })

        # Case 2: Test each value independently.
        # TODO: a better test can be achieved if all possible combinations are
        # tested (e.g. using an stack and playing with the push/pop operations
        # can help to satisfy the requirement). This is still pending because
        # of time restrictions.
        with self.assertRaises(TypeError):
            ARFFWrapper({
                'data': None,
                'description': '',
                'relation': '',
                'attributes': []
                })

        with self.assertRaises(TypeError):
            ARFFWrapper({
                'data': [],
                'description': None,
                'relation': '',
                'attributes': []
                })

        with self.assertRaises(TypeError):
            ARFFWrapper({
                'data': [],
                'description': '',
                'relation': None,
                'attributes': []
                })

        with self.assertRaises(TypeError):
            ARFFWrapper({
                'data': [],
                'description': '',
                'relation': '',
                'attributes': None
                })

    def test_valid_dictionary(self):
        """Test that a valid object does not throw any error."""
        # The correct way of instanciating. Otherwise it'd trow an error
        obj = ARFFWrapper({
            'data': [],
            'description': '',
            'relation': '',
            'attributes': []
            })
        # We assert the object has been correctly created
        # TODO: Assert the log message instead.
        self.assertIsInstance(obj, ARFFWrapper)

    def test_sort_attributes_order(self):
        """Check that sorting of the data by column (attributes).

        It is necessary that we verify that sorting by column does not corrupt
        the dataset. In this test we make this check.
        """
        # Load the data
        arff_wrapper = ARFFWrapper(arff_filepath=self.automl_path)

        # Sort the attributes
        arff_wrapper.sort_attributes()

        # 1. Test order is correct
        previous = arff_wrapper.attribute_names()[0]
        for element in arff_wrapper.attribute_names():
            self.assertLessEqual(previous, element)
            previous = element

    def test_sort_attributes_difference(self):
        """Check that sorting of the data by column (attributes).

        It is necessary that we verify that sorting by column does not corrupt
        the dataset. In this test we make this check.
        """
        # Load the data
        arff_wrapper = ARFFWrapper(arff_filepath=self.automl_path)

        # Create a copy of the original
        original_cp = arff_wrapper.copy()

        # Sort the attributes
        arff_wrapper.sort_attributes()

        # 2. Test lists have exactly the same elements
        set_1 = original_cp.attribute_names()
        set_2 = arff_wrapper.attribute_names()

        # First check lengths
        self.assertTrue(len(set_1) == len(set_2))

        # Check 1 by 1
        while set_1:
            element = set_1.pop()  # Remove from set_1
            self.assertTrue(element in set_2)
            set_2.remove(element)  # If it was present, remove it from set_2

        # Check that both are empty after validation process
        self.assertFalse(set_1 and set_2)

    def test_sort_attributes_values(self):
        """Check that sorting of the data by column (attributes).

        It is necessary that we verify that sorting by column does not corrupt
        the dataset. In this test we make this check.
        """
        # Load the data
        arff_wrapper = ARFFWrapper(arff_filepath=self.automl_path)

        # Create a copy of the original
        original_cp = arff_wrapper.copy()

        # Sort the attributes
        arff_wrapper.sort_attributes()

        # 3. Test that values per column are exactly as in the original object
        for element in original_cp.attribute_names():
            # TODO: This intrinsically checks the values_by_attribute method,
            # but it may require an isolated test.
            values_1 = original_cp.values_by_attribute(element)
            values_2 = arff_wrapper.values_by_attribute(element)
            self.assertFalse(
                np.sum(
                    ~np.equal(values_1, values_2)
                )
            )

    # TODO: Test the copy method
    def test_copy(self):
        """Test the copy method returns different instances but same values."""
        arff_orig = ARFFWrapper(arff_filepath=self.automl_path)
        arff_copy = arff_orig.copy()

        # Test references are different
        self.assertFalse(arff_copy is arff_orig)

        # Test contents are the same
        self.assertEqual(arff_orig.name, arff_copy.name)
        self.assertEqual(arff_orig.description, arff_copy.description)
        self.assertTrue(arff_orig.data.equals(arff_copy.data))

    def test_shape(self):
        """Test the shape method's type and length."""
        arff_wrapper = ARFFWrapper(arff_filepath=self.automl_path)
        self.assertTrue(
            isinstance(arff_wrapper.shape, tuple) and
            len(arff_wrapper.shape) == 2
        )

    def test_as_pandas(self):
        """Test as_pandas_df() always returns a pandas data frame."""
        arff_wrapper = ARFFWrapper(arff_filepath=self.automl_path)
        self.assertTrue(isinstance(arff_wrapper.as_pandas_df(), pd.DataFrame))

    def test_as_numpy(self):
        """Test as_pandas_df() always returns a numpy ndarray."""
        arff_wrapper = ARFFWrapper(arff_filepath=self.automl_path)
        self.assertTrue(isinstance(arff_wrapper.as_numpy_array(), np.ndarray))

    def test_change_attribute_type(self):
        """Test changing an attribute works and do not affect others."""
        # Get the arff object
        arff_wrapper = ARFFWrapper(arff_filepath=self.automl_path)

        # We want the first element. We know is started as str and we want it
        # to be int
        index = 0
        inst_id_col = arff_wrapper.attribute_names()[index]

        # We store the original types
        original_types = arff_wrapper.attribute_types()

        # We change the attribute to int
        arff_wrapper.change_attribute_type(inst_id_col, int)
        # and store the new types
        new_types = arff_wrapper.attribute_types()

        # We compore the index=0 for original and new. They must be different
        col_orig_type = original_types[index]
        col_new_type = new_types[index]
        self.assertNotEqual(col_orig_type, col_new_type)

        # The rest must be equal-wise
        for original, new in zip(original_types[1:], new_types[1:]):
            self.assertEqual(original, new)

    def test_attribute_key_default(self):
        """Test that the defaule key attribute is correctly created."""
        arff_wrapper = ARFFWrapper(arff_filepath=self.automl_path)
        expected = arff_wrapper.attribute_names()[0]
        # Intrinsically checks length 1 and identity of sets.
        self.assertTrue(set([expected]) == set(arff_wrapper.key_attributes))

    def test_attribute_key_add_one(self):
        """Test that the key_attributes attribute is correctly updated."""
        # Get object
        arff_wrapper = ARFFWrapper(arff_filepath=self.automl_path)
        attrs = arff_wrapper.attribute_names()

        # Get the new key to add and create target object
        new_key = attrs[len(attrs) - 1]
        expected_keys = arff_wrapper.key_attributes.copy()
        expected_keys.append(new_key)

        # Add the new key and make assertion
        arff_wrapper.add_attribute_key(new_key)
        self.assertTrue(set(expected_keys) == set(arff_wrapper.key_attributes))

    def test_drop_attributes_single(self):
        """Test that dropping a single attribute is working correctly."""
        # Get the object
        arff_wrapper = ARFFWrapper(arff_filepath=self.automl_path)
        attrs = arff_wrapper.attribute_names()
        to_remove = attrs[int(len(attrs)/2)]

        # Remove from expected list
        expected_keys = arff_wrapper.attribute_names().copy()
        expected_keys.remove(to_remove)

        # Remove from actual arff object
        arff_wrapper.drop_attributes(to_remove)

        # Assert that values are the same ...
        for expected, actual in zip(expected_keys,
                                    arff_wrapper.attribute_names()):
            self.assertTrue(expected == actual)

        # ... and that lengths are equal.
        self.assertTrue(
            len(expected_keys) == len(arff_wrapper.attribute_names())
        )

        # Also test that we actually removed 1 element from the original
        self.assertTrue(
            (len(expected_keys) + 1) == len(attrs)
        )

    def test_drop_attributes_multiple(self):
        """Test that dropping a list of multiple attributes works correctly."""
        # Get the object
        arff_wrapper = ARFFWrapper(arff_filepath=self.automl_path)
        attrs = arff_wrapper.attribute_names()
        to_remove_1 = attrs[int(len(attrs)/2)]
        to_remove_2 = attrs[1]

        # Remove from expected list
        expected_keys = arff_wrapper.attribute_names().copy()
        expected_keys.remove(to_remove_1)
        expected_keys.remove(to_remove_2)

        # Remove from actual arff object
        arff_wrapper.drop_attributes([to_remove_1, to_remove_2])

        # Assert that values are the same ...
        for expected, actual in zip(expected_keys,
                                    arff_wrapper.attribute_names()):
            self.assertTrue(expected == actual)

        # ... and that lengths are the same after removal
        self.assertTrue(
            len(expected_keys) == len(arff_wrapper.attribute_names())
        )

        # Also test that we actually removed 2 elements from the original
        self.assertTrue(
            (len(expected_keys) + 2) == len(attrs)
        )

    def test_sort_rows_single_default(self):
        """Test that default sort for a single column works correctly."""
        arff_wrapper = ARFFWrapper(arff_filepath=self.automl_path)
        attrs = arff_wrapper.attribute_names()
        criteria = attrs[int(len(attrs)/2)]

        arff_wrapper.sort_rows(criteria)

        previous = arff_wrapper.values_by_attribute(criteria)[0]
        for value in arff_wrapper.values_by_attribute(criteria):
            self.assertLessEqual(previous, value)
            previous = value

    def test_sort_rows_single_reverse(self):
        """Test that reverse sort for a single column works correctly."""
        arff_wrapper = ARFFWrapper(arff_filepath=self.automl_path)
        attrs = arff_wrapper.attribute_names()
        criteria = attrs[int(len(attrs)/2)]

        arff_wrapper.sort_rows(criteria, False)

        previous = arff_wrapper.values_by_attribute(criteria)[0]
        for value in arff_wrapper.values_by_attribute(criteria):
            self.assertGreaterEqual(previous, value)
            previous = value

    def test_file_precendece_over_dict(self):
        """Test that loading a file overrides passed dict."""
        expected_relation = 'auto-sklearn_FEATURE_COSTS'

        fake_dict = {
            'data': [],
            'description': '',
            'relation': 'My fake relation',
            'attributes': []
        }
        arff_wrapper = ARFFWrapper(arff_dataset=fake_dict,
                                   arff_filepath=self.automl_path)

        self.assertEqual(arff_wrapper.name, expected_relation)

    # TODO: Maybe add tests for inplace changes/exceptions column string not
    # found/exceptions out of bounds/etc.

if __name__ == '__main__':
    unittest.main()
