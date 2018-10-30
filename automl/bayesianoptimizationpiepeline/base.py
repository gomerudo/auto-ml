from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC
from automl.createconfigspacepipeline.base import ConfigSpacePipeline
import numpy as np
from sklearn.model_selection import cross_val_score
from automl.utils import json_utils


class BayesianOptimizationPipeline:
    """This class deals with the optimization of the hyperparameter of the pipeline.

    This class makes use of SMAC3 library which uses the bayesian optimization technique to optimize the hyperparameter
    of the input pipeline.

    """
    def __init__(self, X, y, pipeline, optimize_on="quality", cutoff_time=600, iteration=7, scoring=None):
        """Initializer of the class BayesianOptimizationPipeline.

        Args:
            X: Training sample.
            y: Test sample.
            pipeline: Input pipeline to be optimized.
            optimize_on: Optimization parameter that takes input either "quality" or "runtime". The pipeline can either
                be optimized based on quality (performance metric) or runtime (time required to complete the bayesian
                optimization process).
            cutoff_time: This hyperparameter is only used when "optimize_on" hyperparameter is set to "runtime". Maximum
                time limit in seconds after which the bayesian optimization process stops and best result is given as
                output.
            iteration: Number of iteration the bayesian optimization process will take. More number of iterations gives
                better results but increases the execution time.
            scoring: The performance metric on which the pipeline is supposed to be optimized.
        """
        self.X = X
        self.y = y
        self.pipeline = pipeline
        self.optimize_on = optimize_on
        self.cutoff_time = cutoff_time
        self.iteration = iteration
        self.scoring = scoring

    def optimize_pipeline(self):
        """This function is used to initialize the optimization process.

        This function creates a function which consist of the input pipeline which is evaluated based on various
        hyperparameter setting provided by the bayesian optimization technique.

        Returns:
            float: The optimized test score.
            Pipeline: The pipeline with optimized hyperparameter.

        """

        def optimization_algorithm(config_dict):
            config_dict = {k: config_dict[k] for k in config_dict if config_dict[k]}
            config_dict = self.convert_string_to_boolean_or_none(config_dict)

            for i in range(0, len(self.pipeline.steps)):
                component = self.pipeline.steps[i][1]
                if "estimator" in component.get_params():
                    component_name = component.estimator.__class__.__name__
                    component_dict = self.get_hyperparameter_for_component_from_dict(config_dict, component_name)
                    self.pipeline.steps[i][1].estimator.set_params(**component_dict)
                else:
                    component_name = component.__class__.__name__
                    component_dict = self.get_hyperparameter_for_component_from_dict(config_dict, component_name)
                    self.pipeline.steps[i][1].set_params(**component_dict)

            score_array = cross_val_score(self.pipeline, self.X, self.y, cv=5, scoring=self.scoring)
            return 1-np.mean(score_array)

        cs = ConfigSpacePipeline(self.pipeline).get_config_space()
        cs_as_json = json_utils.convert_cs_to_json(cs)
        if not cs_as_json['hyperparameters']:
            scores = cross_val_score(self.pipeline, self.X, self.y, cv=5)
            return np.mean(scores), self.pipeline
        scenario = self.create_scenario(cs, self.optimize_on, self.iteration, self.cutoff_time)
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                    tae_runner=optimization_algorithm)
        incumbent = smac.optimize()
        inc_value = optimization_algorithm(incumbent)
        return (1-inc_value), self.pipeline

    @staticmethod
    def create_scenario(cs, optimize_on, iteration, cutoff_time):
        """This function is used to create the Scenario object which is used by smac.

        Args:
            cs: Configuration space
            optimize_on: Optimization parameter that takes input either "quality" or "runtime". The pipeline can either
                be optimized based on quality (performance metric) or runtime (time required to complete the bayesian
                optimization process).
            iteration: Number of iteration the bayesian optimization process will take. More number of iterations gives
                better results but increases the execution time.
            cutoff_time: This hyperparameter is only used when "optimize_on" hyperparameter is set to "runtime". Maximum
                time limit in seconds after which the bayesian optimization process stops and best result is given as
                output.

        Returns:
            Scenario object

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
    def convert_string_to_boolean_or_none(config_dict):
        """ This function replaces string boolean or none with it's actual form

        Example : replaces "True" with True (String to Boolean)

        Args:
            config_dict: configuration dictionary

        Returns:
            Reset configuration dictionary

        """
        for hyperparameter in config_dict:
            if config_dict[hyperparameter] == "True":
                config_dict[hyperparameter] = True
            elif config_dict[hyperparameter] == "False":
                config_dict[hyperparameter] = False
            elif config_dict[hyperparameter] == "None":
                config_dict[hyperparameter] = None
        return config_dict

    @staticmethod
    def get_hyperparameter_for_component_from_dict(config_dict, component_name):
        """This function returns the component's hyperparameter from the configuration space.

        This function is used to search for the component's hyperparameter from the configuration space and remove the
        prefix component name from the configuration space.

        Args:
            config_dict: Dictionary of all the component's hyperparameter in the pipeline.
            component_name: Name of the component.

        Returns:
            Dictionary of the specific component's hyperparameter.

        """
        component_dict = {}
        for hyperparameter in list(config_dict):
            if hyperparameter.startswith((component_name+':')):
                left, right = hyperparameter.split(":", 1)
                component_dict[right] = config_dict[hyperparameter]
        return component_dict
