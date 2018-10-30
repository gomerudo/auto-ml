from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter
from automl.utils import json_utils

cs = ConfigurationSpace()

norm = CategoricalHyperparameter("norm", ["l1", "l2", "max"], "l2")

cs.add_hyperparameter(norm)

json_utils.write_cs_to_json_file(cs, "Normalizer")
