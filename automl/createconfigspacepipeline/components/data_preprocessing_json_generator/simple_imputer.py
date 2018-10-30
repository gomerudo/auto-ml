from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from automl.utils import json_utils

strategy = CategoricalHyperparameter(
    "strategy", ["mean", "median", "most_frequent"], default_value="mean")
cs = ConfigurationSpace()
cs.add_hyperparameter(strategy)

json_utils.write_cs_to_json_file(cs, "SimpleImputer")
