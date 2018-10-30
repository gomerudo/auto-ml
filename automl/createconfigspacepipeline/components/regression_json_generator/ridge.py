from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant
from automl.utils import json_utils

cs = ConfigurationSpace()
alpha = UniformFloatHyperparameter(
    "alpha", 10 ** -5, 10., log=True, default_value=1.)
fit_intercept = Constant("fit_intercept", "True")
tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1,
                                 default_value=1e-3, log=True)
cs.add_hyperparameters([alpha, fit_intercept, tol])

json_utils.write_cs_to_json_file(cs, "Ridge")
