from unittest import TestCase
from sklearn.pipeline import Pipeline
from automl.createconfigspacepipeline.base import ConfigSpacePipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import FastICA
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from tpot.builtins import StackingEstimator
from ConfigSpace import ConfigurationSpace


class TestConfigSpacePipeline(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = Pipeline([('normalize', Normalizer(norm="max")), ('fastica', FastICA()),
                             ('stacking', StackingEstimator(estimator=ExtraTreesClassifier())),
                             ('Xgboost', XGBClassifier(base_score=0.9,
                                                       booster="dart",
                                                       min_child_weight=21,
                                                       n_estimators=10,
                                                       reg_alpha=1e-10))])

        cls.config_space_pipeline_obj = ConfigSpacePipeline(cls.pipeline)
        component_name_xgbc = 'XGBClassifier'
        cls.component_json_xgbc = cls.config_space_pipeline_obj.get_component_json(component_name_xgbc)
        cls.component_dict_xgbc = cls.pipeline.steps[3][1].get_params()

        component_name_etc = 'ExtraTreesClassifier'
        cls.component_json_etc = cls.config_space_pipeline_obj.get_component_json(component_name_etc)
        cls.component_dict_etc = cls.pipeline.steps[2][1].estimator.get_params()

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_config_space(self):
        self.assertIsInstance(self.config_space_pipeline_obj.get_config_space(), ConfigurationSpace)

    def test_get_component_json(self):
        self.assertIsInstance(self.component_json_xgbc, dict)

    def test_component_json_exist(self):
        component_name_1 = 'XGBClassifier'
        component_name_2 = 'Component_name_that_is_not_present_in_components_directory'
        self.assertEqual(self.config_space_pipeline_obj.component_json_exist(component_name_1), True)
        self.assertEqual(self.config_space_pipeline_obj.component_json_exist(component_name_2), False)

    def test_component_reset_default(self):
        self.assertIsInstance(self.config_space_pipeline_obj.component_reset_default(self.component_json_xgbc,
                                                                                     self.component_dict_xgbc), dict)

    def test_is_key_in_dic(self):
        key_1 = "booster"
        key_2 = "some_random_hyperparameter_not_present_in_the_component_dictionary"
        self.assertTrue(self.config_space_pipeline_obj.is_key_in_dic(self.component_dict_xgbc, key_1))
        self.assertFalse(self.config_space_pipeline_obj.is_key_in_dic(self.component_dict_xgbc, key_2))
        self.assertIsInstance(self.config_space_pipeline_obj.is_key_in_dic(self.component_dict_xgbc, key_1), bool)

    def test_is_type_same(self):
        hyperparameter_value_1 = "dart"
        hyperparameter_value_2 = 2.435
        self.assertIsInstance(self.config_space_pipeline_obj.is_type_same(hyperparameter_value_1,
                                                                          hyperparameter_value_2), bool)

    def test_json_process_for_categorical(self):
        component_json_categorical = self.component_json_xgbc['hyperparameters'][1]
        component_dict_categorical = self.component_dict_xgbc[component_json_categorical['name']]
        self.assertIsInstance(self.config_space_pipeline_obj.json_process_for_categorical(component_json_categorical,
                                                                                          component_dict_categorical),
                              dict)

    def test_json_process_for_int_and_float(self):
        component_json_int_float = self.component_json_xgbc['hyperparameters'][2]
        component_dict_int_float = self.component_dict_xgbc[component_json_int_float['name']]
        self.assertIsInstance(self.config_space_pipeline_obj.json_process_for_int_and_float(component_json_int_float,
                                                                                            component_dict_int_float),
                              dict)

    def test_is_string_boolean(self):
        component_json_boolean = self.component_json_etc['hyperparameters'][0]['default']
        component_dict_boolean = self.component_dict_etc[self.component_json_etc['hyperparameters'][0]['name']]
        self.assertIsInstance(self.config_space_pipeline_obj.is_string_boolean(component_json_boolean,
                                                                               component_dict_boolean),
                              bool)
        print(self.component_json_etc)

    def test_is_string_none(self):
        component_json_none = self.component_json_etc['hyperparameters'][2]['value']
        component_dict_none = self.component_dict_etc[self.component_json_etc['hyperparameters'][2]['name']]
        self.assertIsInstance(self.config_space_pipeline_obj.is_string_boolean(component_json_none,
                                                                               component_dict_none),
                              bool)
