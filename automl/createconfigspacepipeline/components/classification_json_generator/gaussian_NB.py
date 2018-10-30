from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from automl.utils import json_utils

cs = ConfigurationSpace()

var_smoothing = UniformFloatHyperparameter(
    "var_smoothing", 1e-11, 1e-7, default_value=1e-9, log=True)
cs.add_hyperparameter(var_smoothing)

json_utils.write_cs_to_json_file(cs, "GaussianNB")
