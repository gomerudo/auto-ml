from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import InCondition, EqualsCondition
from automl.utl import json_utils

cs = ConfigurationSpace()

kernel = CategoricalHyperparameter('kernel', ['poly', 'rbf', 'sigmoid', 'cosine'], 'rbf')
n_components = UniformIntegerHyperparameter(
    "n_components", 50, 10000, default_value=100, log=True)
gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                   log=True, default_value=0.1)
degree = UniformIntegerHyperparameter('degree', 2, 5, 3)
coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)

cs.add_hyperparameters([kernel, degree, gamma, coef0, n_components])

degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])

gamma_kernels = ["poly", "rbf", "sigmoid"]

gamma_condition = InCondition(gamma, kernel, gamma_kernels)
cs.add_conditions([degree_depends_on_poly, coef0_condition, gamma_condition])

json_utils.write_cs_to_json_file(cs, "Nystroem")
