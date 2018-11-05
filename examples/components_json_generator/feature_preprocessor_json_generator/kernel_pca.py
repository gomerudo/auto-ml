from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter, UniformFloatHyperparameter, Constant
from ConfigSpace.conditions import EqualsCondition, InCondition
from automl.utl import json_utils

cs = ConfigurationSpace()
n_components = Constant("n_components", "None")
kernel = CategoricalHyperparameter('kernel',
                                   ['poly', 'rbf', 'sigmoid', 'cosine'], 'rbf')
gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                   log=True, default_value=1.0)
degree = UniformIntegerHyperparameter('degree', 2, 5, 3)
coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
cs.add_hyperparameters([n_components, kernel, degree, gamma, coef0])

degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
gamma_condition = InCondition(gamma, kernel, ["poly", "rbf"])
cs.add_conditions([degree_depends_on_poly, coef0_condition, gamma_condition])

json_utils.write_cs_to_json_file(cs, "KernelPCA")
