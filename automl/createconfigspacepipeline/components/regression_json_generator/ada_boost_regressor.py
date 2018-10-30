from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant
from automl.utils import json_utils

cs = ConfigurationSpace()

# base_estimator = Constant(name="base_estimator", value="None")
n_estimators = UniformIntegerHyperparameter(
    name="n_estimators", lower=50, upper=500, default_value=50,
    log=False)
learning_rate = UniformFloatHyperparameter(
    name="learning_rate", lower=0.01, upper=2, default_value=0.1,
    log=True)
loss = CategoricalHyperparameter(
    name="loss", choices=["linear", "square", "exponential"],
    default_value="linear")
max_depth = UniformIntegerHyperparameter(
    name="max_depth", lower=1, upper=10, default_value=1, log=False)

cs.add_hyperparameters([n_estimators, learning_rate, loss, max_depth])

json_utils.write_cs_to_json_file(cs, "AdaBoostRegressor")
