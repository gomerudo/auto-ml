"""TODO.

Some test
"""

import unittest
from automl.utl.arff_operations import ARFFWrapper
import pkg_resources
import os.path

class Test_TestARFFWrapper(unittest.TestCase):  
    """TODO.

    Add more
    """

    def test_valid_keys_arff(self):
        automl_path = os.path.dirname(os.path.dirname(__file__))
        automl_path = os.path.join(automl_path, "metalearning", "db", "files", 
            "features_costs.arff")

        with self.assertRaises(KeyError):
            ARFFWrapper({})
        
        with self.assertRaises(KeyError):
            ARFFWrapper({'data' : None})

        with self.assertRaises(KeyError):
            ARFFWrapper({'description' : None})

        with self.assertRaises(KeyError):
            ARFFWrapper({'relation' : None})
        
        with self.assertRaises(KeyError):
            ARFFWrapper({'attributes' : None})

        # This would launch an exception of other type if the keys are ok,
        # namely TypeError
        with self.assertRaises(TypeError):
            ARFFWrapper({
                'data' : None,
                'description' : None,
                'relation': None,
                'attributes' : None
                })

    def test_valid_values_arff(self):
        automl_path = os.path.dirname(os.path.dirname(__file__))
        automl_path = os.path.join(automl_path, "metalearning", "db", "files", 
            "features_costs.arff")

        with self.assertRaises(TypeError):
            ARFFWrapper({
                'data' : None,
                'description' : None,
                'relation': None,
                'attributes' : None
                })
        
        with self.assertRaises(TypeError):
            ARFFWrapper({
                'data' : None,
                'description' : '',
                'relation': '',
                'attributes' : []
                })

        with self.assertRaises(TypeError):
            ARFFWrapper({
                'data' : [],
                'description' : None,
                'relation': '',
                'attributes' : []
                })

        with self.assertRaises(TypeError):
            ARFFWrapper({
                'data' : [],
                'description' : '',
                'relation': None,
                'attributes' : []
                })

        with self.assertRaises(TypeError):
            ARFFWrapper({
                'data' : [],
                'description' : '',
                'relation': '',
                'attributes' : None
                })
        
        # The correct way of instanciating. Otherwise it'd trow an error
        obj = ARFFWrapper({
                'data' : [],
                'description' : '',
                'relation': '',
                'attributes' : []
                })
        self.assertIsInstance(obj, ARFFWrapper)
        
if __name__ == '__main__':
    unittest.main()
