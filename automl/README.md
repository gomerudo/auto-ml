# The package's directory structure

In the `automl` package there are 5 main subpackages that abstract and isolate
the logic for each of the architectural components and 2 more with python code
used along the other pieces of code. Here we explain them all.

## The main `automl` package

Here we define global operations and variables that can be used by the
subpackages, such as the log function or the log prefix to achieve a standard
logging behaviour.

## The `errors` subpackage

This package contains classes to raise customized exceptions (errors).

## The `utl` subpackage

Utility methods and/or variables that are not part of the auto-ml solution but
help to achieve python operations. An example can be a generic sorting function.

## The `datahandler` subpackage

In the `datahandler` subpackage the `Dataset` class used within all the `automl`
package is defined together with the `DataLoader` class to retrieve datasets
from different sources (only OpenML is currently supported).

## The `metalearning` subpackage

In this subpackages two other subpackages are defined as next

### The `database` subsubpackage

Here, different classes to load the database copied from `auto-sklearn` are
created together with the high level class `MKDatabaseClient` that can be used
to easily interact with the database.

It also hosts the `arff` and `csv` files storing the meta-knowledge.

### The `metafeatures` subsubpackage

Classes and methods to compute meta-features for a given `Dataset` object are
defined here. The most importat high level class here is the
`MetaFeaturesManager` class.

## The `discovery` subpackage

This package hosts the code to run TPOT using a `Dataset` object and, possibly,
a restricted search space based on meta-suggestions. The top classes here are
`Assistant` and `PipelineDiscovery`.

## The `bayesianoptimizationpipeline` subpackage

This module makes use of SMAC3 library which uses the bayesian optimization
technique to optimize the hyperparameter of the input pipeline. The most
important class is `BayesianOptimizationPipeline`.

## The `createconfigspacepipeline` subpackage

This module deals with the creation and manipulation of configuration space
from the given input pipeline. The reference class in this module is 
`ConfigSpacePipeline`, which is not supposed to be instantiated in a workflow
but used internally by the bayesian module.
