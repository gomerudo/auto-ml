"""Handle the metalearning information and provide classes to interact.

The metalearning step is a core component of this solution. We use the results
obtained by `auto-sklearn <https://github.com/automl/auto-sklearn>`_ to obtain
suggestions of candidate models (classifiers only) that may work for a new
dataset problem.

In a nutshell, auto-sklearn computed - for about 140 datasets -, metafeatures
(statistics and landmark results) that characterize each of these datasets in
a vector form. Additionally, for each of the datasets, one configuration that
solved the problem (a pipeline combaning 1 pre-processor, 1
feature-engineering-algorithm and 1 model) was stored for each of the built-in
scikit-learn metrics.

Hence, for a new dataset, we can compute the metafeatures, get its most similar
stored datasets and assume that the models that solved these similar datasets
can also perform well in the new data.

This module contains submodels implementing this approach.
"""

CONFIGURATIONS_CSV_NAME = "configurations.csv"
"""Name of the auto-sklearn's file storing the discoverd configurations for a
given metric."""

ALGORUNS_CSV_NAME = "algorithm_runs.arff"
"""Name of the auto-sklearn's file storing the relations dataset-configuration
for a given metric."""
