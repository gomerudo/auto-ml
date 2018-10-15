"""Test the different miscellaneous methods that we implement."""

import unittest
from automl.utl.miscellaneous import argsort_list


class TestMiscellaneous(unittest.TestCase):
    """Test methods in Miscellaneous file."""

    def test_argsort_list_default(self):
        """Test that argsort_list sorts correctly.

        We verify that the order a>b is preserverd as expected.
        """
        input_object = [
            ('e', {1, 2}),
            ('c', 'A'),
            ('a', 0),
            ('b', 1),
            ('d', 2)
        ]

        expected_object = [
            ('a', 0),
            ('b', 1),
            ('c', 'A'),
            ('d', 2),
            ('e', {1, 2})
        ]

        indices = argsort_list(input_object)

        result = []
        for index in indices:
            result.append(input_object[index])

        self.assertListEqual(result, expected_object, "Lists are not equal")

if __name__ == '__main__':
    unittest.main()
