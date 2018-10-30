from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter, \
    CategoricalHyperparameter
from automl.utils import json_utils

cs = ConfigurationSpace()

n_quantiles = UniformIntegerHyperparameter(
    'n_quantiles', lower=10, upper=2000, default_value=1000
)
output_distribution = CategoricalHyperparameter(
    'output_distribution', ['uniform', 'normal']
)
cs.add_hyperparameters((n_quantiles, output_distribution))

json_utils.write_cs_to_json_file(cs, "QuantileTransformer")
