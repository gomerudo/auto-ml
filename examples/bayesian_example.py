from automl.datahandler.dataloader import DataLoader
from automl.discovery.assistant import Assistant

dataset = DataLoader.get_openml_dataset(openml_id=46, problem_type=0)
assistant = Assistant(dataset)

# import pipeline and all the necessary components for the pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import FastICA
from tpot.builtins import StackingEstimator
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

# Build the Pipeline
pipe = Pipeline([('normalize', Normalizer(norm="max")),
                                 ('fast_ica', FastICA()),
                                 ('stacking_estimator', StackingEstimator(estimator=ExtraTreesClassifier())),
                                 ('Xgboost', XGBClassifier(base_score=0.9,
                                                           booster="dart",
                                                           min_child_weight=21,
                                                           n_estimators=10,
                                                           reg_alpha=1e-10))])

from automl.bayesianoptimizationpiepeline.base import BayesianOptimizationPipeline

bayesian = BayesianOptimizationPipeline(
    dataset=dataset,
    pipeline=pipe,
    optimize_on="quality",  # alternatively on 'runtime' but it is not advised due to bad results
    # cutoff_time=200, # time in seconds after which the optimization stops. Used only when optimize_on='runtime'
    iteration=20,  # number of iteration the bayesian optimization will take. More iteration yields better result
    # but will hamper the runtime.
    scoring="accuracy",  # The performance metric on which the pipeline is supposed to be optimized
    cv=5  # Specifies the number of folds in a (Stratified)KFold
)

bayesian.optimize_pipeline()
'''
Note that some iteration might give error such as 'array must not contain infs or NaNs' due to a bug in sklearn(for 
FastICA). The iteration with errors are discarded and further iterations are continued to be executed.
Link to the bug : https://github.com/scikit-learn/scikit-learn/pull/2738
'''

print("Score : {}".format(bayesian.get_optimized_score()))
print("Optimized Pipeline : ")
print(bayesian.get_optimized_pipeline())