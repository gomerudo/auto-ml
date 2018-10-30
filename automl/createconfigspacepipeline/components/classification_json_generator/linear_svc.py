from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter,\
    CategoricalHyperparameter, Constant
from ConfigSpace.forbidden import ForbiddenEqualsClause, \
    ForbiddenAndConjunction
from automl.utils import json_utils

cs = ConfigurationSpace()

penalty = CategoricalHyperparameter(
    "penalty", ["l1", "l2"], default_value="l2")
loss = CategoricalHyperparameter(
    "loss", ["hinge", "squared_hinge"], default_value="squared_hinge")
dual = Constant("dual", "False")
# This is set ad-hoc
tol = UniformFloatHyperparameter(
    "tol", 1e-5, 1e-1, default_value=1e-4, log=True)
C = UniformFloatHyperparameter(
    "C", 0.03125, 32768, log=True, default_value=1.0)
multi_class = Constant("multi_class", "ovr")
# These are set ad-hoc
fit_intercept = Constant("fit_intercept", "True")
intercept_scaling = Constant("intercept_scaling", 1)
cs.add_hyperparameters([penalty, loss, dual, tol, C, multi_class,
                        fit_intercept, intercept_scaling])

penalty_and_loss = ForbiddenAndConjunction(
    ForbiddenEqualsClause(penalty, "l1"),
    ForbiddenEqualsClause(loss, "hinge")
)
constant_penalty_and_loss = ForbiddenAndConjunction(
    ForbiddenEqualsClause(dual, "False"),
    ForbiddenEqualsClause(penalty, "l2"),
    ForbiddenEqualsClause(loss, "hinge")
)
penalty_and_dual = ForbiddenAndConjunction(
    ForbiddenEqualsClause(dual, "False"),
    ForbiddenEqualsClause(penalty, "l1")
)
cs.add_forbidden_clause(penalty_and_loss)
cs.add_forbidden_clause(constant_penalty_and_loss)
cs.add_forbidden_clause(penalty_and_dual)

json_utils.write_cs_to_json_file(cs, "LinearSVC")
