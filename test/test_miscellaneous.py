"""TODO.

Some test
"""

import unittest
from automl.utl.miscellaneous import argsort_list

class Test_TestMiscellaneous(unittest.TestCase):  
    """TODO.

    Add more
    """        

    def test_argsort_list_default(self):

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

        self.assertListEqual(result, expected_object, 
            "Lists are not equal")

if __name__ == '__main__':
    unittest.main()
