from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, CategoricalHyperparameter, \
    UnParametrizedHyperparameter, Constant
from ConfigSpace.conditions import EqualsCondition
from automl.utl import json_utils

cs = ConfigurationSpace()

# Parameterized Hyperparameters
max_depth = UniformIntegerHyperparameter(
    name="max_depth", lower=1, upper=20, default_value=3
)
learning_rate = UniformFloatHyperparameter(
    name="learning_rate", lower=0.001, upper=1, default_value=0.1,
    log=True,
)
n_estimators = UniformIntegerHyperparameter(
    name="n_estimators", lower=50, upper=500, default_value=100
)
booster = CategoricalHyperparameter(
    "booster", ["gbtree", "dart"]
)
subsample = UniformFloatHyperparameter(
    name="subsample", lower=0.01, upper=1.0, default_value=1.0, log=False
)
min_child_weight = UniformIntegerHyperparameter(
    name="min_child_weight", lower=0,
    upper=20, default_value=1, log=False
)
colsample_bytree = UniformFloatHyperparameter(
    name="colsample_bytree", lower=0.1, upper=1.0, default_value=1,
)
colsample_bylevel = UniformFloatHyperparameter(
    name="colsample_bylevel", lower=0.1, upper=1.0, default_value=1,
)
reg_alpha = UniformFloatHyperparameter(
    name="reg_alpha", lower=1e-10, upper=1e-1, log=True,
    default_value=1e-10)
reg_lambda = UniformFloatHyperparameter(
    name="reg_lambda", lower=1e-10, upper=1e-1, log=True,
    default_value=1e-10)

# Unparameterized Hyperparameters
# https://xgboost.readthedocs.io/en/latest//parameter.html
# minimum loss reduction required to make a further partition on a
# leaf node of the tree
gamma = Constant(
    name="gamma", value=0)
# absolute regularization (in contrast to eta), comparable to
# gradient clipping in deep learning - according to the internet this
#  is most important for unbalanced data
max_delta_step = Constant(
    name="max_delta_step", value=0)
base_score = Constant(
    name="base_score", value=0.5)
scale_pos_weight = Constant(
    name="scale_pos_weight", value=1)

cs.add_hyperparameters([
    # Active
    max_depth, learning_rate, n_estimators, booster,
    subsample, colsample_bytree, colsample_bylevel,
    reg_alpha, reg_lambda,
    # Inactive
    min_child_weight, max_delta_step, gamma,
    base_score, scale_pos_weight
])

json_utils.write_cs_to_json_file(cs, "XGBRegressor")

