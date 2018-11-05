**AutoML Bayesian Optimization**

In order to apply bayesian optimization for a given pipeline on a dataset we do as follows : 

1. Initialize objects of DataLoader

    ```python
    from automl.datahandler.dataloader import DataLoader
    
    dataset = DataLoader.get_openml_dataset(openml_id=46, problem_type=0)
    ```

2. Create a pipeline

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
                     ('fast_ica', FastICA()),
                     ('stacking_estimator', StackingEstimator(estimator=ExtraTreesClassifier())),
                     ('Xgboost', XGBClassifier(base_score=0.9,
                                               booster="dart",
                                               min_child_weight=21,
                                               n_estimators=10,
                                               reg_alpha=1e-10))])
    ```

3. Now that we have the dataset and pipeline, we can perform bayesian optimization as follows:

    ```python
    from automl.bayesianoptimizationpiepeline.base import BayesianOptimizationPipeline
    
    bayesian = BayesianOptimizationPipeline(
                dataset=dataset,
                pipeline=pipe,
                optimize_on="quality", # alternatively on 'runtime' but it is not advised due to bad results
                #cutoff_time=200, # time in seconds after which the optimization stops. Used only when optimize_on='runtime'
                iteration=20, # number of iteration the bayesian optimization will take. More iteration yields better result
                              # but will hamper the runtime.
                scoring="accuracy", # The performance metric on which the pipeline is supposed to be optimized
                cv = 5 #Specifies the number of folds in a (Stratified)KFold
            )
            
    bayesian.optimize_pipeline()
    
    print("Score : {}".format(bayesian.get_optimized_score()))
    print("Optimized Pipeline : ")
    print(bayesian.get_optimized_pipeline())
    ```
Output:
```
Score : 0.8705248088790724
Optimized Pipeline : 
Pipeline(memory=None,
     steps=[('normalizer', Normalizer(copy=True, norm='l2')), ('fastica', FastICA(algorithm='parallel', fun='cube', 
    fun_args=None, max_iter=200, n_components=None, random_state=None, tol=0.0001, w_init=None,
    whiten=False)), ('stackingestimator', StackingEstimator(estimator=ExtraTreesClassifier(bo...37933e-09,
       scale_pos_weight=1, seed=None, silent=True,
       subsample=0.8418124314528033))])
```

Note : After the execution of the above code, a log folder named "smac3-output_*' is created.