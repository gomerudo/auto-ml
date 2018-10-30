from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant
from automl.utils import json_utils

cs = ConfigurationSpace()
n_estimators = UniformIntegerHyperparameter("n_estimators", 50, 500, default_value=100)
criterion = CategoricalHyperparameter(
    "criterion", ["gini", "entropy"], default_value="gini")

max_features = UniformFloatHyperparameter(
    "max_features", 0., 1., default_value=0.5)

max_depth = Constant("max_depth", "None")
min_samples_split = UniformFloatHyperparameter(
    "min_samples_split", 0., 1., default_value=0.5)
min_samples_leaf = UniformFloatHyperparameter(
    "min_samples_leaf", 0., 0.5, default_value=0.5)
min_weight_fraction_leaf = Constant("min_weight_fraction_leaf", 0.)
max_leaf_nodes = Constant("max_leaf_nodes", "None")
min_impurity_decrease = Constant('min_impurity_decrease', 0.0)
bootstrap = CategoricalHyperparameter(
    "bootstrap", ["True", "False"], default_value="True")
cs.add_hyperparameters([n_estimators, criterion, max_features, max_depth, min_samples_split, min_samples_leaf,
                        min_weight_fraction_leaf, max_leaf_nodes,
                        bootstrap, min_impurity_decrease])

json_utils.write_cs_to_json_file(cs, "ExtraTreesRegressor")
