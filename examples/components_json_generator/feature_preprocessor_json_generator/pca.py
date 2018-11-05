from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    CategoricalHyperparameter
from automl.utl import json_utils

cs = ConfigurationSpace()

n_components = UniformFloatHyperparameter(
    "n_components", 0.5, 0.9999, default_value=0.9999)
whiten = CategoricalHyperparameter(
    "whiten", ["False", "True"], default_value="False")

cs.add_hyperparameters([n_components, whiten])

json_utils.write_cs_to_json_file(cs, "PCA")
