from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from automl.createconfigspacepipeline.base import ConfigSpacePipeline
import numpy as np
from sklearn.model_selection import cross_val_score
from automl.utl import json_utils
from sklearn.pipeline import make_pipeline, make_union


class BayesianOptimizationPipeline:
    """This class deals with the optimization of the hyperparameter of the pipeline.

    This class makes use of SMAC3 library which uses the bayesian optimization technique to optimize the hyperparameter
    of the input pipeline.

    Args:
            dataset (Dataset): A dataset object to work with. Defaults to None.
            pipeline (Pipeline): Input pipeline to be optimized. Defaults to None.
            optimize_on (string): Optimization parameter that takes input either "quality" or "runtime". The pipeline
                can either be optimized based on quality (performance metric) or runtime (time required to complete the
                bayesian optimization process). Defaults to "quality".
            cutoff_time (int): This hyperparameter is only used when "optimize_on" hyperparameter is set to "runtime".
                Maximum time limit in seconds after which the bayesian optimization process stops and best result is
                given as output. Defaults to 600.
            iteration (int): Number of iteration the bayesian optimization process will take. More number of iterations
                gives better results but increases the execution time. Defaults to 15.
            scoring (string or callable): The performance metric on which the pipeline is supposed to be optimized.
                Example : 'auc_roc', 'accuracy', 'precision', 'recall', 'f1', etc. Defaults to None.
            cv (int): Specifies the number of folds in a (Stratified)KFold. Defaults to 5.

    """
    def __init__(self, dataset=None, pipeline=None, optimize_on="quality", cutoff_time=600, iteration=15, scoring=None,
                 cv=5):
        """Initializer of the class BayesianOptimizationPipeline."""

        self.dataset = dataset
        self.pipeline = pipeline
        self.optimize_on = optimize_on
        self.cutoff_time = cutoff_time
        self.iteration = iteration
        self.scoring = scoring
        self.cv = cv

        self.occurrence_no = {}
        self.score = None
        self.opt_pipeline = None

    def optimize_pipeline(self):
        """This function is used to initiate the optimization process and obtained the optimized score and pipeline.

        This function gets the configuration space from the createconfigspacepipeline subpackage on which the bayesian
        optimization technique is performed. This function creates a function which consist of the input pipeline which
        is evaluated based on various hyperparameter setting provided by the bayesian optimization technique (SMAC).


        """
        X = self.dataset.X
        y = self.dataset.y

        def _optimization_algorithm(config_dict):
            config_dict = {k: config_dict[k] for k in config_dict if config_dict[k]}
            config_dict = self._convert_string_to_boolean_or_none(config_dict)
            pipeline_list = []
            for i in range(0, len(self.pipeline.steps)):
                component = self._process_component(self.pipeline.steps[i][1], config_dict)
                pipeline_list.append(component)
            self.opt_pipeline = make_pipeline(*pipeline_list)
            score_array = cross_val_score(self.opt_pipeline, X, y, cv=self.cv, scoring=self.scoring)
            return 1-np.mean(score_array)

        cs = ConfigSpacePipeline(self.pipeline).get_config_space()
        cs_as_json = json_utils._convert_cs_to_json(cs)
        if not cs_as_json['hyperparameters']:
            scores = cross_val_score(self.pipeline, X, y, cv=self.cv, scoring=self.scoring)
            self.score = np.mean(scores)
            self.opt_pipeline = self.pipeline
            return self
        scenario = self._create_scenario(cs, self.optimize_on, self.iteration, self.cutoff_time)
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=_optimization_algorithm)
        incumbent = smac.optimize()
        inc_value = _optimization_algorithm(incumbent)
        self.score = (1-inc_value)
        return self

    def get_optimized_score(self):
        """ This function returns the optimized score

        Raises:
            TypeError: If the function returns None, then the pipeline has'nt been optimized.

        Returns:
            float: The optimized score.

        """
        if self.score is None:
            raise TypeError("None returned. Optimize the pipeline first!")
        else:
            return self.score

    def get_optimized_pipeline(self):
        """ This function returns the optimized pipeline

        Raises:
            TypeError: If the function returns None, then the pipeline has'nt been optimized.

        Returns:
            Pipeline: The optimized pipeline.

        """
        if self.score is None:
            raise TypeError("None returned. Optimize the pipeline first!")
        else:
            return self.opt_pipeline

    def _process_component(self, component, config_dict):
        """Processes each component

        Args:
            component (obj): Object of the component.
            config_dict (dict): Dictionary with hyperparameter configuration.

        Returns:
            obj : Object of the component with reset default values.
        """

        if self._get_component_name(component) == "FeatureUnion":
            union_list=[]
            for i in range(0, len(component.transformer_list)):
                union_component = self._process_component(component.transformer_list[i][1], config_dict)
                union_list.append(union_component)
            return make_union(*union_list)
        else:
            if "estimator" in component.get_params():
                component_name = self._get_component_name(component.estimator)
                self.occurrence_no = self._update_occurrence(self.occurrence_no, component_name)
                component_dict = self._get_hyperparameter_for_component_from_dict(
                    config_dict, component_name
                )
                component.estimator.set_params(**component_dict)
                return component

            else:
                component_name = self._get_component_name(component)
                self.occurrence_no = self._update_occurrence(self.occurrence_no, component_name)
                component_dict = self._get_hyperparameter_for_component_from_dict(
                    config_dict, component_name
                )
                component.set_params(**component_dict)
                return component

    @staticmethod
    def _update_occurrence(occurrence_no, component_name):
        if component_name in occurrence_no:
            occurrence_no[component_name] = occurrence_no[component_name] + 1
        else:
            occurrence_no[component_name] = 1
        return occurrence_no

    @staticmethod
    def _create_scenario(cs, optimize_on, iteration, cutoff_time):
        """This function is used to create the Scenario object which is used by smac.

        Args:
            cs (ConfigurationSpace): Configuration space
            optimize_on (string): Optimization parameter that takes input either "quality" or "runtime". The pipeline
                can either be optimized based on quality (performance metric) or runtime (time required to complete the
                bayesian optimization process).
            iteration (int): Number of iteration the bayesian optimization process will take. More number of iterations
                gives better results but increases the execution time.
            cutoff_time (int): This hyperparameter is only used when "optimize_on" hyperparameter is set to "runtime".
                Maximum time limit in seconds after which the bayesian optimization process stops and best result is
                given as output.

        Returns:
            Scenario: Scenario object.

        """
        if optimize_on == "quality":
            scenario = Scenario({"run_obj": optimize_on,
                                 "runcount-limit": iteration,
                                 "cs": cs,
                                 "deterministic": "true"
                                })
        elif optimize_on == "runtime":
            scenario = Scenario({"run_obj": optimize_on,
                                 "runcount-limit": iteration,
                                 "cs": cs,
                                 "deterministic": "true",
                                 "cutoff_time" : cutoff_time
                                 })
        try:
            return scenario
        except UnboundLocalError:
            raise UnboundLocalError("'optimize_on' parameter can only be set to 'quality' or 'runtime'")

    @staticmethod
    def _convert_string_to_boolean_or_none(config_dict):
        """ This function replaces string boolean or none with it's actual form

        For example, it replaces "True" with True (String to Boolean)

        Args:
            config_dict (dict): configuration dictionary

        Returns:
            dict: Reset configuration dictionary

        """
        for hyperparameter in config_dict:
            if config_dict[hyperparameter] == "True":
                config_dict[hyperparameter] = True
            elif config_dict[hyperparameter] == "False":
                config_dict[hyperparameter] = False
            elif config_dict[hyperparameter] == "None":
                config_dict[hyperparameter] = None
        return config_dict

    def _get_hyperparameter_for_component_from_dict(self, config_dict, component_name):
        """This function returns the component's hyperparameter from the configuration space.

        This function is used to search for the component's hyperparameter from the configuration space and remove the
        prefix component name from the configuration space.

        Args:
            config_dict (dict): Dictionary of all the component's hyperparameter in the pipeline.
            component_name (string): Name of the component.

        Returns:
            dict: Dictionary of the specific component's hyperparameter.

        """
        component_dict = {}
        for hyperparameter in list(config_dict):
            if hyperparameter.startswith((component_name+'-' + str(self.occurrence_no[component_name]) + ':')):
                left, right = hyperparameter.split(":", 1)
                component_dict[right] = config_dict[hyperparameter]
        return component_dict

    @staticmethod
    def _get_component_name(component):
        return component.__class__.__name__
