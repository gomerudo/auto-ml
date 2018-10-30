from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant
from automl.utils import json_utils

cs = ConfigurationSpace()

n_neighbors = UniformIntegerHyperparameter(
    name="n_neighbors", lower=1, upper=100, log=True, default_value=5)
weights = CategoricalHyperparameter(
    name="weights", choices=["uniform", "distance"], default_value="uniform")
p = CategoricalHyperparameter(name="p", choices=[1, 2], default_value=2)

cs.add_hyperparameters([n_neighbors, weights, p])

json_utils.write_cs_to_json_file(cs, "KNeighborsRegressor")
