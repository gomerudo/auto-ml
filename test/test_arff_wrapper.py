"""Test the ARFFWrapper class."""

import unittest
import os.path
from automl.utl.arff_operations import ARFFWrapper


class TestARFFWrapper(unittest.TestCase):
    """Class to test the methods and main behavior of the ARFFWrapper."""

    def test_valid_keys_arff(self):
        """Test the validity check for keys in an ARFF file.

        We check that if an ARFF file with invalid keys is given, the parser
        will throw the correspondant exceptions. This is, check that
        """
        # Check case when non are given.
        with self.assertRaises(KeyError):
            ARFFWrapper({})

        # Check case when only one of them is given.
        with self.assertRaises(KeyError):
            ARFFWrapper({'data': None})

        with self.assertRaises(KeyError):
            ARFFWrapper({'description': None})

        with self.assertRaises(KeyError):
            ARFFWrapper({'relation': None})

        with self.assertRaises(KeyError):
            ARFFWrapper({'attributes': None})

        # This would launch an exception of other type if the keys are ok,
        # namely TypeError
        with self.assertRaises(TypeError):
            ARFFWrapper({
                'data': None,
                'description': None,
                'relation': None,
                'attributes': None
                })

    def test_valid_values_arff(self):
        """Test the validity check of values in the ARFF file.

        If the keys of an ARFF are correct, but the values are not of the
        expected type, the TypeError should happen.
        """
        with self.assertRaises(TypeError):
            ARFFWrapper({
                'data': None,
                'description': None,
                'relation': None,
                'attributes': None
                })

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

        # The correct way of instanciating. Otherwise it'd trow an error
        obj = ARFFWrapper({
            'data': [],
            'description': '',
            'relation': '',
            'attributes': []
            })
        self.assertIsInstance(obj, ARFFWrapper)

    def test_sorted_default(self):
        """Check that sorting of the data by column (attributes).

        It is necessary that we verify that sorting by column does not corrupt
        the dataset. In this test we make this check.
        """
        automl_path = os.path.dirname(os.path.dirname(__file__))
        automl_path = os.path.join(automl_path,
                                   "automl",
                                   "metalearning",
                                   "db",
                                   "files",
                                   "feature_costs.arff")

        arff_wrapper = ARFFWrapper()
        arff_wrapper.load_dataset(automl_path)
        arff_wrapper.sort_attributes()
        # TODO: finish the test

if __name__ == '__main__':
    unittest.main()
