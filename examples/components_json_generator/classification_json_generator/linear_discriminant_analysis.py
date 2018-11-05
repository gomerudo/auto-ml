from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from automl.utl import json_utils

cs = ConfigurationSpace()
shrinkage = UniformFloatHyperparameter(
    "shrinkage", 0., 1., 0.5)
tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-4, log=True)
cs.add_hyperparameters([shrinkage, tol])

json_utils.write_cs_to_json_file(cs, "LinearDiscriminantAnalysis")
