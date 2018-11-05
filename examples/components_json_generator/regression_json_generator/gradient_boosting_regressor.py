from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant
from ConfigSpace.conditions import InCondition
from automl.utl import json_utils

cs = ConfigurationSpace()
loss = CategoricalHyperparameter(
    "loss", ["ls", "lad", "huber", "quantile"], default_value="ls")
learning_rate = UniformFloatHyperparameter(
    name="learning_rate", lower=0.01, upper=1, default_value=0.1, log=True)
n_estimators = UniformIntegerHyperparameter(
    "n_estimators", 50, 500, default_value=100)
max_depth = Constant("max_depth_none", "None")
min_samples_split = UniformFloatHyperparameter(
    "min_samples_split", 0., 1., default_value=0.5)
min_samples_leaf = UniformFloatHyperparameter(
    "min_samples_leaf", 0., 0.5, default_value=0.0001)
min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.)
subsample = UniformFloatHyperparameter(
    name="subsample", lower=0.01, upper=1.0, default_value=1.0, log=False)
max_features = UniformFloatHyperparameter(
    "max_features", 0., 1., default_value=0.5)
max_leaf_nodes = Constant("max_leaf_nodes", "None")
min_impurity_decrease = Constant('min_impurity_decrease', 0.0)
alpha = UniformFloatHyperparameter(
    "alpha", lower=0.75, upper=0.99, default_value=0.9)

cs.add_hyperparameters([loss, learning_rate, n_estimators, max_depth,
                        min_samples_split, min_samples_leaf,
                        min_weight_fraction_leaf, subsample, max_features,
                        max_leaf_nodes, min_impurity_decrease, alpha])

cs.add_condition(InCondition(alpha, loss, ['huber', 'quantile']))

json_utils.write_cs_to_json_file(cs, "GradientBoostingRegressor")
