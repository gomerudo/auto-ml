from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter
from ConfigSpace.conditions import EqualsCondition
from automl.utils import json_utils

cs = ConfigurationSpace()

json_utils.write_cs_to_json_file(cs, "OneHotEncoder")
