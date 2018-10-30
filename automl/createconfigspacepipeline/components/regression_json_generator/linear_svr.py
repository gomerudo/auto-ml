from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction
from automl.utils import json_utils

cs = ConfigurationSpace()
C = UniformFloatHyperparameter(
    "C", 0.03125, 32768, log=True, default_value=1.0)
loss = CategoricalHyperparameter(
    "loss", ["epsilon_insensitive", "squared_epsilon_insensitive"],
    default_value="squared_epsilon_insensitive")
epsilon = UniformFloatHyperparameter(
    name="epsilon", lower=0.001, upper=1, default_value=0.1, log=True)
dual = Constant("dual", "False")
tol = UniformFloatHyperparameter(
    "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
fit_intercept = Constant("fit_intercept", "True")
intercept_scaling = Constant("intercept_scaling", 1)

cs.add_hyperparameters([C, loss, epsilon, dual, tol, fit_intercept,
                        intercept_scaling])

dual_and_loss = ForbiddenAndConjunction(
    ForbiddenEqualsClause(dual, "False"),
    ForbiddenEqualsClause(loss, "epsilon_insensitive")
)
cs.add_forbidden_clause(dual_and_loss)

json_utils.write_cs_to_json_file(cs, "LinearSVR")
