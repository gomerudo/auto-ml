from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant
from automl.utl import json_utils

cs = ConfigurationSpace()

alpha = UniformFloatHyperparameter(
    name="alpha", lower=1e-14, upper=1.0, default_value=1e-10, log=True)

cs.add_hyperparameter(alpha)

json_utils.write_cs_to_json_file(cs, "GaussianProcessRegressor")
