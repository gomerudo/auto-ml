




Pre-defined Configuration Space
-------------------------------

The library consists of configuration spaces in JSON format of many data preprocessor, feature preprocessor and machine 
learning algorithms. All the JSON files are present in automl.createconfigspacepipeline.json_files_for_all_components.
A list of configuration spaces that are present are listed below:
1. Data Preprocessor
    1. Imputer
    2. Normalizer
    3. OneHotEncoder
    4. QuantileTransformer
    5. SimpleImputer
2. Feature Preprocessor
    1. FastICA
    2. FeatureAgglomeration
    3. KernelPCA
    4. Nystroem
    5. PCA
    6. PolynomialFeatures
    7. RBFSampler
3. Classification ML Algorithm
    1. AdaBoostClassifier
    2. BernoulliNB
    3. DecisionTreeClassifier
    4. ExtraTreesClassifier
    5. GaussianNB
    6. GradientBoostingClassifier
    7. KNeighborsClassifier
    8. LinearDiscriminantAnalysis
    9. LinearSVC
    10. MultinomialNB
    11. PassiveAggressiveClassifier
    12. QuadraticDiscriminantAnalysis
    13. RandomForestClassifier
    14. SGDClassifier
    15. SVC
    16. XGBClassifier
4. Regression ML Algorithm
    1. AdaBoostRegressor
    2. ARDRegression
    3. DecisionTreeRegressor
    4. ExtraTreesRegressor
    5. GaussianProcessRegressor
    6. GradientBoostingRegressor
    7. KNeighborsRegressor
    8. LinearSVR
    9. RandomForestRegressor
    10. Ridge
    11. SGDRegressor
    12. SVR
    13. XGBRegressor
    
If the pipeline consist of component other than the one mentioned above, the hyperparameter of that component will not
be optimized.

Creating Configuration space of a component
-------------------------------------------

Example of creating a configuration space for SVC (Support Vector Classification) is as follows : 
1. Initialize ConfigurationSpace object.

    ```python
    
    from ConfigSpace import ConfigurationSpace
    cs = ConfigurationSpace()
 
    ```

2. Create hyperparameter of the components.
    
    ```python
    
    from ConfigSpace.hyperparameters import UniformIntegerHyperparameter,UniformFloatHyperparameter, \
    CategoricalHyperparameter, Constant
    
    # The C parameter is declared as float. It is given range [0.03125, 32768] and a default value of 1.0. Log scale has
    # been set to true. Similarly, we can set other float hyperparameter as well.
    C = UniformFloatHyperparameter("C", 0.03125, 32768, log=True, default_value=1.0)
    gamma = UniformFloatHyperparameter("gamma", 3.0517578125e-05, 8, log=True, default_value=0.1)
    coef0 = UniformFloatHyperparameter("coef0", -1, 1, default_value=0)
    tol = UniformFloatHyperparameter("tol", 1e-5, 1e-1, default_value=1e-3, log=True)
 
    # Integer hyperparameter can be set as follows.
    degree = UniformIntegerHyperparameter("degree", 2, 5, default_value=3)
    

    # Categorical hyperparameter can be set as follows.
    kernel = CategoricalHyperparameter(name="kernel", choices=["linear", "rbf", "poly", "sigmoid"], 
                                       default_value="rbf")
                                    
    # Boolean and None types are given input as a string (categorical). The library converts it to the original form
    # during execution.
    shrinking = CategoricalHyperparameter("shrinking", ["True", "False"], default_value="True")
    
    # Some hyperparameter are set to constant in order to achieve optimal solution.
    max_iter = Constant("max_iter", -1)
    ```
    
3. Add all the hyperparameters to the configuration space.

    ```python
    cs.add_hyperparameters([C, kernel, degree, gamma, coef0, shrinking, tol, max_iter])
    ```
    
4. Creating and Adding conditions. For example, the degree hyperparameter is only used when kernel = "poly", using the 
degree hyperparameter with any other kernel leads to error.

    ```python   
    # Creating condition.
    degree_depends_on_poly = EqualsCondition(degree, kernel, "poly")
    coef0_condition = InCondition(coef0, kernel, ["poly", "sigmoid"])
    
    # Adding condition to the configuration space.
    cs.add_condition(degree_depends_on_poly)
    cs.add_condition(coef0_condition)    
    ```
    
5. Add the configuration space in JSON format to the library. It is really important to note that the parameter 
"json_name" in the function "write_cs_to_json_file" should be set to the name of the class of the component which in our
case is "SVC". For example, for Random Forest Classifier json_name="RandomForestClassifier""

    ```python
    write_cs_to_json_file(cs=cs, json_name="SVC")
    ```    

For further details on ConfigSpace library please refer [here](https://github.com/automl/ConfigSpace)

