from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant
from automl.utils import json_utils

C = UniformFloatHyperparameter("C", 1e-5, 10, 1.0, log=True)
fit_intercept = UnParametrizedHyperparameter("fit_intercept", "True")
loss = CategoricalHyperparameter(
    "loss", ["hinge", "squared_hinge"], default_value="hinge"
)

tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4,
                                 log=True)
# Note: Average could also be an Integer if > 1
average = CategoricalHyperparameter('average', ['False', 'True'],
                                    default_value='False')

cs = ConfigurationSpace()
cs.add_hyperparameters([loss, fit_intercept, tol, C, average])

json_utils.write_cs_to_json_file(cs, "PassiveAggressiveClassifier")
