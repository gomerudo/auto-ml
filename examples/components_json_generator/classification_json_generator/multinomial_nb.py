from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant
from automl.utl import json_utils

cs = ConfigurationSpace()

# the smoothing parameter is a non-negative float
# I will limit it to 100 and put it on a logarithmic scale. (SF)
# Please adjust that, if you know a proper range, this is just a guess.
alpha = UniformFloatHyperparameter(name="alpha", lower=1e-2, upper=100,
                                   default_value=1, log=True)

fit_prior = CategoricalHyperparameter(name="fit_prior",
                                      choices=["True", "False"],
                                      default_value="True")

cs.add_hyperparameters([alpha, fit_prior])

json_utils.write_cs_to_json_file(cs, "MultinomialNB")
