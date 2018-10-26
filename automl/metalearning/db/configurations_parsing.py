"""Provide functions, classes and private variables to build configurations.

We define a configuration as a set of classifiers / imputation-methods /
rescalers / preprocessors for a given problem. The set of these configurations
can be understood as The Search Space for a problem.
"""

import pandas as pd
import automl
from automl.errors.customerrors import CurrentlyNonSupportedError

# CONSTANTS (PRIVATE SCOPE)

# Prefixes. Always start with _PF
_PF_CATEGORICAL_ENCODING = "categorical_encoding"
_PF_CLASSIFIER = "classifier"
_PF_PREPROCESSOR = "preprocessor"
_PF_RESCALING = "rescaling"
_PF_STRATEGY = "strategy"
_PF_IMPUTATION = "imputation"

# Encoders. Always start with _ENC
_ENC_ONE_HOT = "one_hot_encoding"

# Classifiers. Always start with _CL
_CL_ADABOOST = "adaboost"
_CL_BERNOULLI_NB = "bernoulli_nb"
_CL_DECISION_TREE = "decision_tree"
_CL_EXTRA_TREES = "extra_trees"
_CL_GRADIENT_BOOSTING = "gradient_boosting"
_CL_K_NEAREST_NEIGHBORS = "k_nearest_neighbors"
_CL_LDA = "lda"
_CL_LIBLINEAR_SVC = "liblinear_svc"
_CL_LIBSVM_SVC = "libsvm_svc"
_CL_MULTINOMIAL_NB = "multinomial_nb"
_CL_PASSIVE_AGGRESSIVE = "passive_aggressive"
_CL_QDA = "qda"
_CL_RANDOM_FOREST = "random_forest"
_CL_SGD = "sgd"

# Pre-processors. Always start with _PP
_PP_EXTRA_TREES_FOR_CL = "extra_trees_preproc_for_classification"
_PP_FAST_ICA = "fast_ica"
_PP_FEATURE_AGGLOMERATION = "feature_agglomeration"
_PP_KERNEL_PCA = "kernel_pca"
_PP_KITCHEN_SINKS = "kitchen_sinks"
_PP_LIBLINEAR_SVC_PREPROCESSOR = "liblinear_svc_preprocessor"
_PP_NYSTROEM_SAMPLER = "nystroem_sampler"
_PP_PCA = "pca"
_PP_POLYNOMIAL = "polynomial"
_PP_RANDOM_TREES_EMBEDDING = "random_trees_embedding"
_PP_SELECT_PERCENTILE_CL = "select_percentile_classification"
_PP_SELECT_RATES = "select_rates"
_PP_TRUNCATED_SVD = "truncatedSVD"

# Rescaling. Always start with _RSC
_RSC_QUANTILE_TRANSFORMER = "quantile_transformer"
_RSC_ROBUST_SCALER = "robust_scaler"
_RSC_MINMAX = "minmax"
_RSC_NORMALIZE = "normalize"
_RSC_STANDARDIZE = "standardize"

# CSV Strings. Always start with _CSV
_CSV_CHOICE = "__choice__"
_CSV_COL_SEP = ":"


def mix_suggestions(suggestion_list):
    """Mix a list of MLSuggestion's into a single one.

    Args:
        suggestion_list     (list). List of MLSuggestion objects.

    Raises:
        TypeError: If the argument is not a list or any of the elements in the
            list is not an instance of MLSuggestion.

    Returns:
        MLSuggestion        The resulting MLSuggestion.

    """
    if isinstance(suggestion_list, list):
        raise TypeError("Argument must be a list")

    result = MLSuggestion()
    for suggestion in suggestion_list:
        if not isinstance(suggestion, MLSuggestion):
            raise TypeError("Invalide type. Not a MLSuggestion")

        result.add_classifier(suggestion.classifiers)
        result.add_encoder(suggestion.encoders)
        result.add_imputation(suggestion.imputations)
        result.add_preprocessor(suggestion.preprocessors)
        result.add_rescaler(suggestion.rescalers)
    return result


class ConfigurationBuilder:
    """Build a configuration: model/pre-processor/encoder/scaler/imputation.

    A set of configurations can be used to feed TPOT with.

    Args:
        model_row (pandas.Series or pandas.DataFrame): Represents a row
            coming from a configurations.csv file. Defaults to None

    Attributes:
        model_row (pandas.Series or pandas.DataFrame): Represents a row
            coming from a configurations.csv file.

    Raises:
        TypeError: If no pandas Series or DataFrame is passed as argument.

    """

    def __init__(self, model_row=None):
        """Constructor."""
        if not isinstance(model_row, pd.DataFrame) and \
                not isinstance(model_row, pd.Series):
            raise TypeError("Only pandas DataFrame or pandas Series are \
                            accepted")

        self.model_row = model_row
        self._internal_list_loader()

    def build_configuration(self):
        """Build a ML Suggestion with the row passed at instaciation time."""
        imputation_col = _PF_IMPUTATION + _CSV_COL_SEP + _PF_STRATEGY
        classifier_choice_col = _PF_CLASSIFIER + _CSV_COL_SEP + _CSV_CHOICE
        preprocessor_choice_col = _PF_PREPROCESSOR + _CSV_COL_SEP + _CSV_CHOICE
        rescaler_choice_col = _PF_RESCALING + _CSV_COL_SEP + _CSV_CHOICE
        encoding_choice_col = _PF_CATEGORICAL_ENCODING + _CSV_COL_SEP + \
            _CSV_CHOICE

        suggestions_dict = {
            _PF_CLASSIFIER: [],
            _PF_PREPROCESSOR: [],
            _PF_RESCALING: [],
            _PF_CATEGORICAL_ENCODING: [],
            _PF_IMPUTATION: [],
        }

        for attribute in [imputation_col, classifier_choice_col,
                          preprocessor_choice_col, rescaler_choice_col,
                          encoding_choice_col]:
            if attribute in self.model_row.index:
                list_name = attribute.split(_CSV_COL_SEP)[0]
                suggestion = self._from_internal_list(attribute)
                suggestions_dict[list_name].append(suggestion)
            else:
                msg = "No attribute '{attr}' in current element\
                      ".format(attr=attribute)
                automl.automl_log(msg, 'WARNING')

        mlsuggestion = MLSuggestion(
            classifiers=suggestions_dict[_PF_CLASSIFIER],
            preprocessors=suggestions_dict[_PF_PREPROCESSOR],
            encoders=suggestions_dict[_PF_CATEGORICAL_ENCODING],
            rescalers=suggestions_dict[_PF_RESCALING],
            imputations=suggestions_dict[_PF_IMPUTATION],
        )

        return mlsuggestion

    def _from_internal_list(self, element_type):
        # TODO: Check existance and handle exception

        if ':' not in element_type:
            raise ValueError("Invalid format in element type. It must contain \
                             at least two strings separated by semicolon")

        split_type = element_type.split(_CSV_COL_SEP)
        l_split_type = len(split_type)

        if l_split_type < 2:
            raise ValueError("Invalid element type format. It must contain \
                             at least two strings separated by semicolon")
        if l_split_type == 2:
            # Case for imputation
            if split_type[0] == _PF_IMPUTATION \
               and split_type[1] == _PF_STRATEGY:
                return _ConfigurationBuilderHelper.resolve_strategy(
                    self._internal_list,
                    split_type[0],
                )
            if split_type[1] == _CSV_CHOICE:
                value = self.model_row[element_type]
                return _ConfigurationBuilderHelper.resolve_choice(
                    self._internal_list,
                    split_type[0],
                    value
                )
            if l_split_type > 2:
                raise ValueError("Not supported yet")

    def _internal_list_loader(self):
        self._internal_list = dict()
        self._add_strategies()
        self._add_categorical_encoders()
        self._add_classifiers()
        self._add_preprocessors()
        self._add_rescalers()

    def _add_strategies(self):
        strategies_map = {
            _PF_IMPUTATION: "sklearn.impute.SimpleImputer"
        }

        self._internal_list.update({
            _PF_STRATEGY: strategies_map,
        })

    def _add_categorical_encoders(self):
        encoders_map = {
            _ENC_ONE_HOT: "sklearn.preprocessing.OneHotEncoder",
        }

        self._internal_list.update({
            _PF_CATEGORICAL_ENCODING: encoders_map,
        })

    def _add_classifiers(self):
        classifiers_map = {
            _CL_ADABOOST: "sklearn.ensemble.AdaBoostClassifier",
            _CL_BERNOULLI_NB: "sklearn.naive_bayes.BernoulliNB",
            _CL_DECISION_TREE: "sklearn.tree.DecisionTreeClassifier",
            _CL_EXTRA_TREES: "sklearn.ensemble.ExtraTreesClassifier",
            _CL_GRADIENT_BOOSTING:
                "sklearn.ensemble.GradientBoostingClassifier",
            _CL_K_NEAREST_NEIGHBORS: "sklearn.neighbors.KNeighborsClassifier",
            _CL_LDA: "sklearn.lda.LDA",
            _CL_LIBLINEAR_SVC: "sklearn.svm.LinearSVC",
            _CL_LIBSVM_SVC: "sklearn.svm.SVC",
            _CL_MULTINOMIAL_NB: "sklearn.naive_bayes.MultinomialNB",
            _CL_PASSIVE_AGGRESSIVE:
                "sklearn.linear_model.PassiveAggressiveClassifier",
            _CL_QDA: "sklearn.qda.QDA",
            _CL_RANDOM_FOREST: "sklearn.ensemble.RandomForestClassifier",
            _CL_SGD: "sklearn.linear_model.SGDClassifier",
        }

        self._internal_list.update({
            _PF_CLASSIFIER: classifiers_map,
        })

    def _add_preprocessors(self):
        preprocessors_map = {
            # We ignore the next two that come from auto-sklearn
            # _PP_EXTRA_TREES_FOR_CL: "extra_trees_preproc_for_classification",
            # _PP_LIBLINEAR_SVC_PREPROCESSOR: "liblinear_svc_preprocessor",
            _PP_FAST_ICA: "sklearn.decomposition.FastICA",
            _PP_FEATURE_AGGLOMERATION: "sklearn.cluster.FeatureAgglomeration",
            _PP_KERNEL_PCA: "sklearn.decomposition.KernelPCA",
            _PP_KITCHEN_SINKS: "sklearn.kernel_approximation.RBFSampler",
            _PP_NYSTROEM_SAMPLER: "sklearn.kernel_approximation.Nystroem",
            _PP_PCA: "sklearn.decomposition.PCA",
            _PP_POLYNOMIAL: "sklearn.preprocessing.PolynomialFeatures",
            _PP_RANDOM_TREES_EMBEDDING:
                "sklearn.ensemble.RandomTreesEmbedding",
            _PP_SELECT_PERCENTILE_CL:
                "sklearn.feature_selection.SelectPercentile",
            _PP_SELECT_RATES: "sklearn.feature_selection.SelectFdr",
            _PP_TRUNCATED_SVD: "sklearn.decomposition.TruncatedSVD",
        }

        self._internal_list.update({
            _PF_PREPROCESSOR: preprocessors_map,
        })

    def _add_rescalers(self):
        rescalers_map = {
            _RSC_QUANTILE_TRANSFORMER:
                "sklearn.preprocessing.QuantileTransformer",
            _RSC_ROBUST_SCALER: "sklearn.preprocessing.RobustScaler",
            _RSC_MINMAX: "sklearn.preprocessing.MinMaxScaler",
            _RSC_NORMALIZE: "sklearn.preprocessing.Normalizer",
            _RSC_STANDARDIZE: "sklearn.preprocessing.StandardScaler",
        }

        self._internal_list.update({
            _PF_RESCALING: rescalers_map,
        })


class _ConfigurationBuilderHelper:

    @staticmethod
    def resolve_choice(dictionary, choice_type, name):
        """Evaluate a generic __choice__ column and resolves the value.

        Resolving means to map the value in the CSV into a scikit-learn class.

        Args:
            dictionary (dict): The dictionary to resolve the values from.
            choice_type (str): The prefix of the __choice__ field.
            name (str): The name to resolve.

        Raises:
            ValueError: If the choice_type is not recognized.

        Returns:
            str:        The auto-sklearn class name to use for 'name'.

        """
        if choice_type == _PF_CLASSIFIER:
            return _ConfigurationBuilderHelper._resolve_classifier(
                dictionary,
                name
            )
        if choice_type == _PF_CATEGORICAL_ENCODING:
            return _ConfigurationBuilderHelper._resolve_encoder(dictionary,
                                                                name)
        if choice_type == _PF_PREPROCESSOR:
            return _ConfigurationBuilderHelper._resolve_preprocessor(
                dictionary,
                name
            )
        if choice_type == _PF_RESCALING:
            return _ConfigurationBuilderHelper._resolve_rescaler(
                dictionary, name
            )

        raise ValueError("Unknown choice type: '{choice}'\
                         ".format(choice=choice_type))

    @staticmethod
    def resolve_strategy(dictionary, strategy_type):
        """Resolve an strategy column.

        Resolving means to map the value in the CSV into a scikit-learn class.

        Args:
            dictionary (dict): The dictionary to resolve from.
            strategy_type (str): The type of strategy to support.

        Raises:
            CurrentlyNonSupportedError: If the strategy_type indicated is not
                supported yet.

        Returns:
            str: The value of the strategy.

        """
        if strategy_type == _PF_IMPUTATION:
            return _ConfigurationBuilderHelper._resolve_imputation(dictionary)

        raise CurrentlyNonSupportedError("Option non supported")

    @staticmethod
    def _resolve_imputation(dictionary):
        return dictionary[_PF_STRATEGY][_PF_IMPUTATION]

    @staticmethod
    def _resolve_classifier(dictionary, name):

        return dictionary[_PF_CLASSIFIER][name]

    @staticmethod
    def _resolve_rescaler(dictionary, name):
        # This does not raise exceptions because 'none' is a possible value
        # in configurations.csv
        if name == 'none':
            return None
        return dictionary[_PF_RESCALING][name]

    @staticmethod
    def _resolve_preprocessor(dictionary, name):
        # This does not raise exceptions because 'none' is a possible value
        # in configurations.csv
        if name == 'no_preprocessing':
            return None
        return dictionary[_PF_PREPROCESSOR][name]

    @staticmethod
    def _resolve_encoder(dictionary, name):
        # This does not raise exceptions because 'none' is a possible value
        # in configurations.csv
        if name == 'no_encoding':
            return None
        return dictionary[_PF_CATEGORICAL_ENCODING][name]


class MLSuggestion:
    """Class that represents a MLSuggestion.

    A Machine Learning suggestion can (in the biggest picture we consider)
    contain imputation methods, encoders, pre-processors, rescalers and/or
    classifiers.

    Please note that in principle we are restricted to the scikit-learn
    classes available and moreover to the auto-sklearn results.

    Args:
        classifiers (list): A list of strings defining the full class path of
            the classifiers (e.g. [sklearn.subgroup.MyClass]).
        rescalers (list): A list of strings defining the full class path of the
            rescalers (e.g. [sklearn.subgroup.MyClass]).
        preprocessors (list):A list of strings defining the full class path of
            the preprocessors (e.g. [sklearn.subgroup.MyClass]).
        encoders (list): A list of strings defining the full class path of the
            encoders (e.g. [sklearn.subgroup.MyClass]).
        imputations (list): A list of strings defining the full class path of
            the imputations (e.g. [sklearn.subgroup.MyClass]).

    Attributes:
        None.
    
    """

    # pylint: disable=R0913
    def __init__(self, classifiers=None, rescalers=None, preprocessors=None,
                 encoders=None, imputations=None):
        """Constructor."""
        self._classifiers = classifiers
        self._rescalers = rescalers
        self._preprocessors = preprocessors
        self._encoders = encoders
        self._imputations = imputations

        if self._classifiers is None:
            self._classifiers = set()
        if self._rescalers is None:
            self._rescalers = set()
        if self._preprocessors is None:
            self._preprocessors = set()
        if self._encoders is None:
            self._encoders = set()
        if self._imputations is None:
            self._imputations = set()

        if isinstance(self._classifiers, list):
            self._classifiers = set(self._classifiers)
        if isinstance(self._rescalers, list):
            self._rescalers = set(self._rescalers)
        if isinstance(self._preprocessors, list):
            self._preprocessors = set(self._preprocessors)
        if isinstance(self._encoders, list):
            self._encoders = set(self._encoders)
        if isinstance(self._imputations, list):
            self._imputations = set(self._imputations)

    def add_classifier(self, classifier):
        """Add a new classifier or set of classifiers.

        Attributes:
            classifier (str or list): If str, then the element is added. If
                list, then all elements in the list are added.

        """
        self._add_to_list(self._classifiers, classifier)

    def add_preprocessor(self, preprocessor):
        """Add a new pre-processor or set of pre-processors.

        Attributes:
            preprocessor (str or list): If str, then the element is added. If
                list, then all elements in the list are added.

        """
        self._add_to_list(self._preprocessors, preprocessor)

    def add_rescaler(self, rescaler):
        """Add a new rescaler or set of rescalers.

        Attributes:
            rescaler (str or list): If str, then the element is added. If list,
                then all elements in the list are added.

        """
        self._add_to_list(self._rescalers, rescaler)

    def add_encoder(self, encoder):
        """Add a new encoder or set of encoders.

        Attributes:
            encoder (str or list): If str, then the element is added. If list,
                then all elements in the list are added.

        """
        self._add_to_list(self._encoders, encoder)

    def add_imputation(self, imputation):
        """Add a new imputation or set of imputation methods.

        Attributes:
            imputation (str or list): If str, then the element is added. If 
                list, then all elements in the list are added.

        """
        self._add_to_list(self._imputations, imputation)

    @staticmethod
    def _add_to_list(input_list, element):
        if isinstance(element, set):
            input_list.update(element)
        else:
            input_list.add(element)

    @property
    def classifiers(self):
        """Return the valid classifiers.

        None values are considered invalid, and hence not included.

        Returns:
            list: The classifiers.
        """
        self._clean_set(self._classifiers)
        return self._classifiers

    @property
    def rescalers(self):
        """Return the valid rescalers.

        None values are considered invalid, and hence not included.
        
        Returns:
            list: The rescalers.
        """
        self._clean_set(self._rescalers)
        return self._rescalers

    @property
    def preprocessors(self):
        """Return the valid preprocessors.

        None values are considered invalid, and hence not included.
        
        Returns:
            list: The preprocessors.
        """
        self._clean_set(self._preprocessors)
        return self._preprocessors

    @property
    def encoders(self):
        """Return the valid _encoders.

        None values are considered invalid, and hence not included.
        
        Returns:
            list: The encoders.
        """
        self._clean_set(self._encoders)
        return self._encoders

    @property
    def imputations(self):
        """Return the valid imputations.

        None values are considered invalid, and hence not included.
        
        Returns:
            list: The imputations.
        """
        self._clean_set(self._imputations)
        return self._imputations

    @staticmethod
    def _clean_set(src_set):
        if src_set is None:
            src_set = set()
        src_set.discard(None)

    def get_all_elements(self):
        """Return all elements.

        Returns:
            list: A single list with all classifiers, rescalers, 
                pre-processors, encoders and imputation methods.

        """
        result = []
        result.extend(self.classifiers)
        result.extend(self.rescalers)
        result.extend(self.preprocessors)
        result.extend(self.encoders)
        result.extend(self.imputations)
        return result
