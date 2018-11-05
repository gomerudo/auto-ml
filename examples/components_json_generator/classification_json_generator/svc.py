from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, Constant
from ConfigSpace.conditions import EqualsCondition, InCondition
from automl.utl import json_utils

C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True,
                               default_value=1.0)
# No linear kernel here, because we have LinearSVC
kernel = CategoricalHyperparameter(name="kernel",
                                   choices=["linear", "rbf", "poly", "sigmoid"],
                                   default_value="rbf")
degree = UniformIntegerHyperparameter("degree", 2, 5, default_value=3)
gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8,
                                   log=True, default_value=0.1)

coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
# probability is no hyperparameter, but an argument to the SVM algo
shrinking = CategoricalHyperparameter("shrinking", ["True", "False"],
                                      default_value="True")
tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3,
                                 log=True)
# cache size is not a hyperparameter, but an argument to the program!
max_iter = Constant("max_iter", -1)

cs = ConfigurationSpace()
cs.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking,
                        tol, max_iter])

degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
cs.add_condition(degree_depends_on_poly)
cs.add_condition(coef0_condition)

json_utils.write_cs_to_json_file(cs, "SVC")
