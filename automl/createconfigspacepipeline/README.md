**Building Configuration Space**

To demonstrate the usage of ConfigSpacePipeline we will consider the following pipeline.

```python
# import pipeline and all the necessary components for the pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import FastICA
from tpot.builtins import StackingEstimator
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

# Build the Pipeline
pipe = Pipeline([('normalize', Normalizer(norm="max")),
                 ('Xgboost', XGBClassifier(base_score=0.9,
                                           booster="dart",
                                           min_child_weight=21,
                                           n_estimators=10,
                                           reg_alpha=1e-10))])
```

Now, that the pipeline is built, we can now create the configuration space as follows :

```python
#import the module
from automl.createconfigspacepipeline.base import ConfigSpacePipeline

cs = ConfigSpacePipeline(pipeline=pipe)

print(cs)
``` 
Output : 

```
Configuration space object:
  Hyperparameters:
    Normalizer-1:norm, Type: Categorical, Choices: {l1, l2, max}, Default: max
    XGBClassifier-1:base_score, Type: Constant, Value: 0.5
    XGBClassifier-1:booster, Type: Categorical, Choices: {gbtree, dart}, Default: dart
    XGBClassifier-1:colsample_bylevel, Type: UniformFloat, Range: [0.1, 1.0], Default: 1.0
    XGBClassifier-1:colsample_bytree, Type: UniformFloat, Range: [0.1, 1.0], Default: 1.0
    XGBClassifier-1:gamma, Type: Constant, Value: 0
    XGBClassifier-1:learning_rate, Type: UniformFloat, Range: [0.001, 1.0], Default: 0.1, on log-scale
    XGBClassifier-1:max_delta_step, Type: Constant, Value: 0
    XGBClassifier-1:max_depth, Type: UniformInteger, Range: [1, 20], Default: 3
    XGBClassifier-1:min_child_weight, Type: UniformInteger, Range: [0, 21], Default: 21
    XGBClassifier-1:n_estimators, Type: UniformInteger, Range: [10, 500], Default: 10
    XGBClassifier-1:reg_alpha, Type: UniformFloat, Range: [1e-10, 0.1], Default: 1e-10, on log-scale
    XGBClassifier-1:reg_lambda, Type: UniformFloat, Range: [1e-10, 0.1], Default: 1e-10, on log-scale
    XGBClassifier-1:scale_pos_weight, Type: Constant, Value: 1
    XGBClassifier-1:subsample, Type: UniformFloat, Range: [0.01, 1.0], Default: 1.0
  Conditions:
```

Note : The class ConfigSpacePipeline is not supposed to be instantiated but is used by the class 
BayesianOptimizationPipeline