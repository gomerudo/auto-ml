"""Module that exposes the classes to compute statistics on a dataset."""

from collections import defaultdict
import logging
import scipy.stats
from scipy.linalg import LinAlgError
import numpy as np
from sklearn.multiclass import OneVsRestClassifier


class StatisticalInformation:
    """Compute statistics on a dataset."""

    ###########################################################################

    # CLASS STATISTICS

    @staticmethod
    def class_entropy(y):  # pylint: disable=C0103
        """Compute statistic."""
        labels = 1 if len(y.shape) == 1 else y.shape[1]
        if labels == 1:
            y = y.reshape((-1, 1))

        entropies = []
        for i in range(labels):
            occurence_dict = defaultdict(float)
            for value in y[:, i]:
                occurence_dict[value] += 1
            entropies.append(
                scipy.stats.entropy(
                    [occurence_dict[key] for key in occurence_dict], base=2
                )
            )

        return np.mean(entropies)

    @staticmethod
    def class_ocurrences(y):  # pylint: disable=C0103
        """Compute statistic."""
        if len(y.shape) == 2:
            occurences = []
            for i in range(y.shape[1]):
                occurences.append(
                    StatisticalInformation.class_ocurrences(y[:, i])
                )
            return occurences
        else:
            occurence_dict = defaultdict(float)
            for value in y:
                occurence_dict[value] += 1
            return occurence_dict

    @staticmethod
    def class_probability_max(y):  # pylint: disable=C0103
        """Compute statistic."""
        occurences = StatisticalInformation.class_ocurrences(y)
        max_value = -1

        if len(y.shape) == 2:
            for i in range(y.shape[1]):
                for num_occurences in occurences[i].values():
                    if num_occurences > max_value:
                        max_value = num_occurences
        else:
            for num_occurences in occurences.values():
                if num_occurences > max_value:
                    max_value = num_occurences

        return float(max_value) / float(y.shape[0])

    @staticmethod
    def class_probability_mean(y):  # pylint: disable=C0103
        """Compute statistic."""
        occurence_dict = StatisticalInformation.class_ocurrences(y)

        if len(y.shape) == 2:
            occurences = []
            for i in range(y.shape[1]):
                occurences.extend(
                    [occurrence for occurrence in occurence_dict[
                        i].values()])
            occurences = np.array(occurences)
        else:
            occurences = np.array(
                [occurrence for occurrence in occurence_dict.values()],
                dtype=np.float64
            )
        return (occurences / y.shape[0]).mean()

    @staticmethod
    def class_probability_min(y):  # pylint: disable=C0103
        """Compute statistic."""
        occurences = StatisticalInformation.class_ocurrences(y)

        min_value = np.iinfo(np.int64).max
        if len(y.shape) == 2:
            for i in range(y.shape[1]):
                for num_occurences in occurences[i].values():
                    if num_occurences < min_value:
                        min_value = num_occurences
        else:
            for num_occurences in occurences.values():
                if num_occurences < min_value:
                    min_value = num_occurences
        return float(min_value) / float(y.shape[0])

    @staticmethod
    def class_probability_std(y):  # pylint: disable=C0103
        """Compute statistic."""
        occurence_dict = StatisticalInformation.class_ocurrences(y)

        if len(y.shape) == 2:
            stds = []
            for i in range(y.shape[1]):
                std = np.array(
                    [occurrence for occurrence in occurence_dict[i].values()],
                    dtype=np.float64)
                std = (std / y.shape[0]).std()
                stds.append(std)
            return np.mean(stds)
        else:
            occurences = np.array(
                [occurrence for occurrence in occurence_dict.values()],
                dtype=np.float64
            )
            return (occurences / y.shape[0]).std()

    ###########################################################################

    # KURTOSIS STATISTICS

    @staticmethod
    def kurtosisses(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        if scipy.sparse.issparse(X):
            kurts = []
            X_new = X.tocsc()  # pylint: disable=C0103
            for i in range(X_new.shape[1]):
                if not categorical_indicators[i]:
                    start = X_new.indptr[i]
                    stop = X_new.indptr[i+1]
                    kurts.append(scipy.stats.kurtosis(X_new.data[start:stop]))
            return kurts
        else:
            kurts = []
            for i in range(X.shape[1]):
                if not categorical_indicators[i]:
                    kurts.append(scipy.stats.kurtosis(X[:, i]))
            return kurts

    @staticmethod
    def kurtosis_max(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        kurts = StatisticalInformation.kurtosisses(X, categorical_indicators)
        # pylint: disable=C1801
        maximum = np.nanmax(kurts) if len(kurts) > 0 else 0
        return maximum if np.isfinite(maximum) else 0

    @staticmethod
    def kurtosis_mean(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        kurts = StatisticalInformation.kurtosisses(X, categorical_indicators)
        # pylint: disable=C1801
        mean = np.nanmean(kurts) if len(kurts) > 0 else 0
        return mean if np.isfinite(mean) else 0

    @staticmethod
    def kurtosis_min(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        kurts = StatisticalInformation.kurtosisses(X, categorical_indicators)
        # pylint: disable=C1801
        minimum = np.nanmin(kurts) if len(kurts) > 0 else 0
        return minimum if np.isfinite(minimum) else 0

    @staticmethod
    def kurtosis_std(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        kurts = StatisticalInformation.kurtosisses(X, categorical_indicators)
        # pylint: disable=C1801
        std = np.nanstd(kurts) if len(kurts) > 0 else 0
        return std if np.isfinite(std) else 0

    ###########################################################################

    # LANDMARK STATISTICS

    @staticmethod
    def landmark_1NN(X, y):  # pylint: disable=C0103
        """Compute statistic."""
        import sklearn.neighbors

        # pylint: disable=C0103
        if len(y.shape) == 1 or y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)

        accuracy = 0.
        for train, test in kf.split(X, y):
            kNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            if len(y.shape) == 1 or y.shape[1] == 1:
                kNN.fit(X[train], y[train].ravel())
            else:
                kNN = OneVsRestClassifier(kNN)
                kNN.fit(X[train], y[train])
            predictions = kNN.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 10

    @staticmethod
    def landmark_decision_node_learner(X, y):  # pylint: disable=C0103
        """Compute statistic."""
        if scipy.sparse.issparse(X):
            return np.NaN

        import sklearn.tree

        # pylint: disable=C0103
        if len(y.shape) == 1 or y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)

        accuracy = 0.
        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy", max_depth=1, random_state=random_state,
                min_samples_split=2, min_samples_leaf=1, max_features=None)
            if len(y.shape) == 1 or y.shape[1] == 1:
                node.fit(X[train], y[train])
            else:
                node = OneVsRestClassifier(node)
                node.fit(X[train], y[train])
            predictions = node.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 10

    @staticmethod
    def landmark_decision_tree(X, y):  # pylint: disable=C0103
        """Compute statistic."""
        if scipy.sparse.issparse(X):
            return np.NaN

        import sklearn.tree

        # pylint: disable=C0103
        if len(y.shape) == 1 or y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)

        accuracy = 0.
        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            tree = sklearn.tree.DecisionTreeClassifier(
                random_state=random_state
            )

            if len(y.shape) == 1 or y.shape[1] == 1:
                tree.fit(X[train], y[train])
            else:
                tree = OneVsRestClassifier(tree)
                tree.fit(X[train], y[train])

            predictions = tree.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 10

    @staticmethod
    def landmark_lda(X, y):  # pylint: disable=C0103
        """Compute statistic."""
        if scipy.sparse.issparse(X):
            return np.NaN

        # pylint: disable=C0103
        import sklearn.discriminant_analysis
        if len(y.shape) == 1 or y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)

        accuracy = 0.
        try:
            for train, test in kf.split(X, y):
                lda = \
                    sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

                if len(y.shape) == 1 or y.shape[1] == 1:
                    lda.fit(X[train], y[train])
                else:
                    lda = OneVsRestClassifier(lda)
                    lda.fit(X[train], y[train])

                predictions = lda.predict(X[test])
                accuracy += sklearn.metrics.accuracy_score(
                    predictions,
                    y[test]
                )
            return accuracy / 10
        except scipy.linalg.LinAlgError as ex:
            # pylint: disable=W1201
            logging.warning("LDA failed: %s Returned 0 instead!" % ex)
            return np.NaN
        except ValueError as ex:
            # pylint: disable=W1201
            logging.warning("LDA failed: %s Returned 0 instead!" % ex)
            return np.NaN

    @staticmethod
    def landmark_naive_bayes(X, y):  # pylint: disable=C0103
        """Compute statistic."""
        if scipy.sparse.issparse(X):
            return np.NaN

        import sklearn.naive_bayes

        # pylint: disable=C0103
        if len(y.shape) == 1 or y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)

        accuracy = 0.
        for train, test in kf.split(X, y):
            nb = sklearn.naive_bayes.GaussianNB()  # pylint: disable=C0103

            if len(y.shape) == 1 or y.shape[1] == 1:
                nb.fit(X[train], y[train])
            else:
                nb = OneVsRestClassifier(nb)  # pylint: disable=C0103
                nb.fit(X[train], y[train])

            predictions = nb.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 10

    @staticmethod
    def landmark_random_node_learner(X, y):  # pylint: disable=C0103
        """Compute statistic."""
        if scipy.sparse.issparse(X):
            return np.NaN

        import sklearn.tree

        # pylint: disable=C0103
        if len(y.shape) == 1 or y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)
        accuracy = 0.

        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy", max_depth=1, random_state=random_state,
                min_samples_split=2, min_samples_leaf=1, max_features=1)
            node.fit(X[train], y[train])
            predictions = node.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 10

    ###########################################################################

    # LOG STATISTICS

    @staticmethod
    def log_dataset_ratio(X):  # pylint: disable=C0103
        """Compute statistic."""
        return np.log(StatisticalInformation.dataset_ratio(X))

    @staticmethod
    def log_inverse_dataset_ratio(X):  # pylint: disable=C0103
        """Compute statistic."""
        return np.log(StatisticalInformation.inverse_dataset_ratio(X))

    @staticmethod
    def log_number_of_features(X):  # pylint: disable=C0103
        """Compute statistic."""
        return np.log(StatisticalInformation.number_of_features(X))

    @staticmethod
    def log_number_of_instances(X):  # pylint: disable=C0103
        """Compute statistic."""
        return np.log(StatisticalInformation.number_of_instances(X))

    ###########################################################################

    # NUMBER STATISTICS

    @staticmethod
    def number_of_categorical_features(categorical_indicators):
        """Compute statistic."""
        return np.sum(categorical_indicators)

    @staticmethod
    def number_of_classes(y):  # pylint: disable=C0103
        """Compute statistic."""
        if len(y.shape) == 2:
            return np.mean(
                [len(np.unique(y[:, i])) for i in range(y.shape[1])]
            )
        else:
            return float(len(np.unique(y)))

    @staticmethod
    def number_of_features(X):  # pylint: disable=C0103
        """Compute statistic."""
        return X.shape[1]

    @staticmethod
    def number_of_features_with_na(X):  # pylint: disable=C0103
        """Compute statistic."""
        if scipy.sparse.issparse(X):
            missing = StatisticalInformation.missing_values(X)
            new_missing = missing.tocsc()
            num_missing = [np.sum(
                new_missing.data[
                    new_missing.indptr[i]:new_missing.indptr[i+1]
                ]) for i in range(missing.shape[1])]

            return float(np.sum([1 if num > 0 else 0 for num in num_missing]))
        else:
            missing = StatisticalInformation.missing_values(X)
            num_missing = missing.sum(axis=0)
            return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

    @staticmethod
    def number_of_instances_with_na(X):  # pylint: disable=C0103
        """Compute statistic."""
        if scipy.sparse.issparse(X):
            missing = StatisticalInformation.missing_values(X)
            new_missing = missing.tocsr()
            num_missing = [np.sum(
                new_missing.data[
                    new_missing.indptr[i]:new_missing.indptr[i + 1]
                ]) for i in range(new_missing.shape[0])]

            return float(np.sum([1 if num > 0 else 0 for num in num_missing]))
        else:
            missing = StatisticalInformation.missing_values(X)
            num_missing = missing.sum(axis=1)
            return float(np.sum([1 if num > 0 else 0 for num in num_missing]))

    @staticmethod
    def number_of_missing_values(X):  # pylint: disable=C0103
        """Compute statistic."""
        return float(StatisticalInformation.missing_values(X).sum())

    @staticmethod
    def number_of_instances(X):  # pylint: disable=C0103
        """Compute statistic."""
        return X.shape[0]

    @staticmethod
    def number_of_numeric_features(categorical_indicators):
        """Compute statistic."""
        return len(categorical_indicators) - np.sum(categorical_indicators)

    @staticmethod
    def missing_values(X):  # pylint: disable=C0103
        """Compute statistic."""
        if scipy.sparse.issparse(X):
            data = [True if not np.isfinite(x) else False for x in X.data]
            return X.__class__(
                (data, X.indices, X.indptr), shape=X.shape,
                dtype=np.bool
            )
        else:
            return ~np.isfinite(X)

    ###########################################################################

    # PCA STATISTICS

    @staticmethod
    def pca(X):  # pylint: disable=C0103
        """Compute statistic."""
        import sklearn.decomposition
        rs = np.random.RandomState(42)  # pylint: disable=C0103
        indices = np.arange(X.shape[0])

        if scipy.sparse.issparse(X):
            pca = sklearn.decomposition.PCA(copy=True)
            for i in range(10):
                try:
                    rs.shuffle(indices)
                    pca.fit(X[indices])
                    return pca
                except LinAlgError:
                    pass
            logging.warning("Failed to compute a Principle Component Analysis")
            return None
        else:
            # This is expensive, but necessary with scikit-learn 0.15
            Xt = X.astype(np.float64)  # pylint: disable=C0103
            for i in range(10):
                try:
                    rs.shuffle(indices)
                    truncated_svd = sklearn.decomposition.TruncatedSVD(
                        n_components=X.shape[1]-1, random_state=i,
                        algorithm="randomized")
                    truncated_svd.fit(Xt[indices])
                    return truncated_svd
                except LinAlgError:
                    pass
            logging.warning("Failed to compute a Truncated SVD")

    @staticmethod
    def pca_fraction_components_95v(X, pca=None):  # pylint: disable=C0103
        """Compute statistic."""
        if pca is None:
            return np.NaN

        sum_ = 0.
        idx = 0
        while sum_ < 0.95 and idx < len(pca.explained_variance_ratio_):
            sum_ += pca.explained_variance_ratio_[idx]
            idx += 1
        return float(idx)/float(X.shape[1])

    @staticmethod
    def pca_kurtosis_first_pc(X, pca=None):  # pylint: disable=C0103
        """Compute statistic."""
        if pca is None:
            return np.NaN

        components = pca.components_
        pca.components_ = components[:1]
        transformed = pca.transform(X)
        pca.components_ = components

        kurtosis = scipy.stats.kurtosis(transformed)
        return kurtosis[0]

    @staticmethod
    def pca_skewness_first_pc(X, pca=None):  # pylint: disable=C0103
        """Compute statistic."""
        if pca is None:
            return np.NaN

        components = pca.components_
        pca.components_ = components[:1]
        transformed = pca.transform(X)
        pca.components_ = components

        skewness = scipy.stats.skew(transformed)
        return skewness[0]

    ###########################################################################
    # PERCENTAGE STATISTICS

    @staticmethod
    def percentage_of_features_with_na(X):  # pylint: disable=C0103
        """Compute statistic."""
        n_feat_na = float(StatisticalInformation.number_of_features_with_na(X))
        n_feat = float(StatisticalInformation.number_of_features(X))
        return n_feat_na/n_feat

    @staticmethod
    def percentage_of_instances_with_na(X):  # pylint: disable=C0103
        """Compute statistic."""
        n_ins_na = float(StatisticalInformation.number_of_instances_with_na(X))
        n_ins = float(StatisticalInformation.number_of_instances(X))
        return n_ins_na/n_ins

    @staticmethod
    def percentage_of_missing_values(X):  # pylint: disable=C0103
        """Compute statistic."""
        n_mv = float(StatisticalInformation.number_of_missing_values(X))
        return n_mv / float(X.shape[0]*X.shape[1])

    ###########################################################################

    # RATIO STATISTICS

    @staticmethod
    def dataset_ratio(X):  # pylint: disable=C0103
        """Compute statistic."""
        return float(StatisticalInformation.number_of_features(X)) /\
            float(StatisticalInformation.number_of_instances(X))

    @staticmethod
    def inverse_dataset_ratio(X):  # pylint: disable=C0103
        """Compute statistic."""
        return 1 / StatisticalInformation.dataset_ratio(X)

    @staticmethod
    def ratio_nominal_numerical(categorical_indicators):
        """Compute statistic."""
        num_categorical = float(
            StatisticalInformation.number_of_categorical_features(
                categorical_indicators
            )
        )
        num_numerical = float(
            StatisticalInformation.number_of_numeric_features(
                categorical_indicators
            )
        )

        if num_numerical == 0.0:
            return 0.
        else:
            return num_categorical / num_numerical

    @staticmethod
    def ratio_numerical_nominal(categorical_indicators):
        """Compute statistic."""
        num_categorical = float(
            StatisticalInformation.number_of_categorical_features(
                categorical_indicators)
            )
        num_numerical = float(
            StatisticalInformation.number_of_numeric_features(
                categorical_indicators
            )
        )

        if num_categorical == 0.0:
            return 0.
        else:
            return num_numerical / num_categorical

    ###########################################################################
    # SKEWNESS STATISTICS

    @staticmethod
    def skewnesses(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        if scipy.sparse.issparse(X):
            skews = []
            X_new = X.tocsc()  # pylint: disable=C0103
            for i in range(X_new.shape[1]):
                if not categorical_indicators[i]:
                    start = X_new.indptr[i]
                    stop = X_new.indptr[i + 1]
                    skews.append(scipy.stats.skew(X_new.data[start:stop]))
            return skews
        else:
            skews = []
            for i in range(X.shape[1]):
                if not categorical_indicators[i]:
                    skews.append(scipy.stats.skew(X[:, i]))
            return skews

    @staticmethod
    def skewness_max(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        skews = StatisticalInformation.skewnesses(X, categorical_indicators)
        # pylint: disable=C1801
        maximum = np.nanmax(skews) if len(skews) > 0 else 0
        return maximum if np.isfinite(maximum) else 0

    @staticmethod
    def skewness_mean(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        skews = StatisticalInformation.skewnesses(X, categorical_indicators)
        # pylint: disable=C1801
        mean = np.nanmean(skews) if len(skews) > 0 else 0
        return mean if np.isfinite(mean) else 0

    @staticmethod
    def skewness_min(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        skews = StatisticalInformation.skewnesses(X, categorical_indicators)
        # pylint: disable=C1801
        minimum = np.nanmin(skews) if len(skews) > 0 else 0
        return minimum if np.isfinite(minimum) else 0

    @staticmethod
    def skewness_std(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        skews = StatisticalInformation.skewnesses(X, categorical_indicators)
        # pylint: disable=C1801
        std = np.nanstd(skews) if len(skews) > 0 else 0
        return std if np.isfinite(std) else 0

    ###########################################################################

    # SYMBOLS STATISTICS

    @staticmethod
    def number_of_symbols(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        if scipy.sparse.issparse(X):
            symbols_per_column = []
            new_X = X.tocsc()  # pylint: disable=C0103
            for i in range(new_X.shape[1]):
                if categorical_indicators[i]:
                    unique_values = np.unique(new_X.getcol(i).data)
                    num_unique = np.sum(np.isfinite(unique_values))
                    symbols_per_column.append(num_unique)
            return symbols_per_column
        else:
            symbols_per_column = []
            for i, column in enumerate(X.T):
                if categorical_indicators[i]:
                    unique_values = np.unique(column)
                    num_unique = np.sum(np.isfinite(unique_values))
                    symbols_per_column.append(num_unique)
            return symbols_per_column

    @staticmethod
    def symbols_max(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        values = StatisticalInformation.number_of_symbols(
            X, categorical_indicators
        )

        if len(values) == 0:  # pylint: disable=C1801
            return 0
        return max(max(values), 0)

    @staticmethod
    def symbols_mean(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        values = [val for val in StatisticalInformation.number_of_symbols(
            X, categorical_indicators) if val > 0]
        mean = np.nanmean(values)
        return mean if np.isfinite(mean) else 0

    @staticmethod
    def symbols_min(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        minimum = None
        n_symbols = StatisticalInformation.number_of_symbols(
            X, categorical_indicators)
        for unique in n_symbols:
            if unique > 0 and (minimum is None or unique < minimum):
                minimum = unique
        return minimum if minimum is not None else 0

    @staticmethod
    def symbols_std(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        n_symbols = StatisticalInformation.number_of_symbols(
            X, categorical_indicators)

        values = [val for val in n_symbols if val > 0]
        std = np.nanstd(values)
        return std if np.isfinite(std) else 0

    @staticmethod
    def symbols_sum(X, categorical_indicators):  # pylint: disable=C0103
        """Compute statistic."""
        sum_res = np.nansum(
            StatisticalInformation.number_of_symbols(X, categorical_indicators)
        )
        return sum_res if np.isfinite(sum_res) else 0
