from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from automl.utl import json_utils

cs = ConfigurationSpace()

degree = UniformIntegerHyperparameter("degree", 2, 3, 2)
interaction_only = CategoricalHyperparameter("interaction_only",
                                             ["False", "True"], "False")
include_bias = CategoricalHyperparameter("include_bias",
                                         ["True", "False"], "True")

cs.add_hyperparameters([degree, interaction_only, include_bias])

json_utils.write_cs_to_json_file(cs, "PolynomialFeatures")
