from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from automl.utils import json_utils

cs = ConfigurationSpace()

n_estimators = UniformIntegerHyperparameter(
    name="n_estimators", lower=50, upper=500, default_value=100)
learning_rate = UniformFloatHyperparameter(
    name="learning_rate", lower=0., upper=1., default_value=1.)
algorithm = CategoricalHyperparameter(
    name="algorithm", choices=["SAMME.R", "SAMME"], default_value="SAMME.R")

cs.add_hyperparameters([n_estimators, learning_rate, algorithm])

json_utils.write_cs_to_json_file(cs, "AdaBoostClassifier")
