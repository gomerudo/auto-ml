from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, Constant
from ConfigSpace.conditions import EqualsCondition
from automl.utils import json_utils

cs = ConfigurationSpace()

n_components = Constant(
    "n_components", "None")
algorithm = CategoricalHyperparameter('algorithm',
                                      ['parallel', 'deflation'], 'parallel')
whiten = CategoricalHyperparameter('whiten',
                                   ['False', 'True'], 'False')
fun = CategoricalHyperparameter(
    'fun', ['logcosh', 'exp', 'cube'], 'logcosh')
cs.add_hyperparameters([n_components, algorithm, whiten, fun])

cs.add_condition(EqualsCondition(n_components, whiten, "True"))

json_utils.write_cs_to_json_file(cs, "FastICA")
