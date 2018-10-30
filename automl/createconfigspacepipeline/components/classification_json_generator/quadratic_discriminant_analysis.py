from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from automl.utils import json_utils

cs =ConfigurationSpace()
reg_param = UniformFloatHyperparameter('reg_param', 0.0, 1.0,
                                       default_value=0.0)
tol = UniformFloatHyperparameter("tol", 1e-6, 1e-2, default_value=1e-4, log=True)
cs.add_hyperparameter(reg_param)

json_utils.write_cs_to_json_file(cs, "QuadraticDiscriminantAnalysis")
