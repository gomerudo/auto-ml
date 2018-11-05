from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter
from automl.utl import json_utils

gamma = UniformFloatHyperparameter(
    "gamma", 3.0517578125e-05, 8, default_value=1.0, log=True)
n_components = UniformIntegerHyperparameter(
    "n_components", 50, 10000, default_value=100, log=True)
cs = ConfigurationSpace()
cs.add_hyperparameters([gamma, n_components])

json_utils.write_cs_to_json_file(cs, "RBFSampler")
