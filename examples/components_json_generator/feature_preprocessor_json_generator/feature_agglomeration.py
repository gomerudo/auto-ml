from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.forbidden import ForbiddenInClause, \
    ForbiddenAndConjunction, ForbiddenEqualsClause
from automl.utl import json_utils

cs = ConfigurationSpace()
n_clusters = UniformIntegerHyperparameter("n_clusters", 2, 400, 2)
affinity = CategoricalHyperparameter(
    "affinity", ["euclidean", "manhattan", "cosine", "l1", "l2"], "euclidean")
linkage = CategoricalHyperparameter(
    "linkage", ["ward", "complete", "average", "single"], "ward")

cs.add_hyperparameters([n_clusters, affinity, linkage])

affinity_and_linkage = ForbiddenAndConjunction(
    ForbiddenAndConjunction(ForbiddenEqualsClause(affinity, "manhattan"),
                            ForbiddenEqualsClause(affinity, "cosine"),
                            ForbiddenEqualsClause(affinity, "l1"),
                            ForbiddenEqualsClause(affinity, "l2")
                            ),
    ForbiddenEqualsClause(linkage, "ward"))
cs.add_forbidden_clause(affinity_and_linkage)

json_utils.write_cs_to_json_file(cs, "FeatureAgglomeration")
