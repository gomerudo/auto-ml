from unittest import TestCase
from automl.bayesianoptimizationpiepeline.base \
    import BayesianOptimizationPipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA, FastICA
from xgboost import XGBClassifier
from tpot.builtins import StackingEstimator
from sklearn.ensemble import ExtraTreesClassifier
from ConfigSpace import ConfigurationSpace
from automl.datahandler.dataloader import DataLoader
from sklearn.pipeline import FeatureUnion


class TestBayesianOptimizationPipeline(TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_optimize_pipeline(self):
        dataset = DataLoader.get_openml_dataset(openml_id=46, problem_type=0)
        pipeline = Pipeline([('normalize-1', Normalizer(norm="max")), ('fastica', FastICA()),
                             ('Union-1', FeatureUnion([('pca-3', PCA(n_components=0.3)),
                                                       ('Union-2', FeatureUnion([('pca-5', PCA(n_components=0.5)),
                                                                                 ('normalize-2',
                                                                                  Normalizer(norm="l1"))])),
                                                       ('pca-7', PCA(n_components=0.7))
                                                       ])),
                             ('stacking-1', StackingEstimator(estimator=ExtraTreesClassifier(n_estimators=10))),
                             ('stacking-2', StackingEstimator(estimator=ExtraTreesClassifier(n_estimators=20))),
                             ('stacking-3', StackingEstimator(estimator=ExtraTreesClassifier(n_estimators=300))),
                             ('Xgboost', XGBClassifier(base_score=0.9,
                                                       booster="dart",
                                                       min_child_weight=21,
                                                       n_estimators=10,
                                                       reg_alpha=1e-10))])
        bayesian = BayesianOptimizationPipeline(dataset, pipeline, optimize_on="quality", iteration=2)

        score, opt_pipeline = bayesian.optimize_pipeline()
        for step in opt_pipeline.steps:
            print(step[1])
        print(score)
        self.assertIsInstance(opt_pipeline, Pipeline)
        self.assertIsInstance(score, float)

    def test_create_scenario(self):
        # check if the 'optimize_on' is not set to value other then 'runtime' or 'quality'
        cs = ConfigurationSpace()
        try:
            BayesianOptimizationPipeline.create_scenario(cs=cs, cutoff_time=200, iteration=2, optimize_on='runtime')
        except (UnboundLocalError, NameError, Exception):
            self.fail()
        try:
            BayesianOptimizationPipeline.create_scenario(cs=cs, cutoff_time=200, iteration=2, optimize_on='quality')
        except (UnboundLocalError, NameError, Exception):
            self.fail()

        with self.assertRaises(UnboundLocalError):
            BayesianOptimizationPipeline.create_scenario(cs=cs, cutoff_time=200, iteration=2,
                                                         optimize_on='some_value_other_than_quality_or_runtime')

    def test_convert_string_to_boolean_or_none(self):
        dict_input = {'a': 'True',
                      'b': 'False',
                      'c': 'None'}
        dict_output = {'a': True,
                       'b': False,
                       'c': None}
        self.assertIsInstance(BayesianOptimizationPipeline.convert_string_to_boolean_or_none(dict_input), dict)
        self.assertEqual(BayesianOptimizationPipeline.convert_string_to_boolean_or_none(dict_input), dict_output)

    def test_get_hyperparameter_for_component_from_dict(self):
        config_dict_input = {'Component1:Hyperparameter1': 'random_value_1',
                             'Component2:Hyperparameter1': 'random_value_2',
                             'Component3:Hyperparameter1': 'random_value_3',
                             'Component1:Hyperparameter2': 'random_value_4'}
        config_dict_output = {'Hyperparameter1': 'random_value_1',
                              'Hyperparameter2': 'random_value_4'}
        component_name = "Component1"
        self.assertIsInstance(
            BayesianOptimizationPipeline.get_hyperparameter_for_component_from_dict(config_dict_input, component_name),
            dict)
        self.assertEqual(
            BayesianOptimizationPipeline.get_hyperparameter_for_component_from_dict(config_dict_input, component_name),
            config_dict_output)


