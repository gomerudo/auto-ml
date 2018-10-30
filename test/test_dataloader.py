"""Test the dataloader module. Special emphasis in the Dataset class."""

import unittest
from automl.datahandler.dataloader import DataLoader, Dataset


class TestDataLoader(unittest.TestCase):
    """Test the DataLoader class' methods."""

    def setUp(self):
        """Initialize the global variables for the tests."""
        self.dataset_classification = DataLoader.get_openml_dataset(
            openml_id=46,
            problem_type=0
        )
        self.dataset_regression = DataLoader.get_openml_dataset(
            openml_id=46,
            problem_type=1
        )

    def test_return_from_openml(self):
        """Test the returned type of get_openml_dataset() method is Dataset."""
        # Test that the correct type is returned
        self.assertTrue(isinstance(self.dataset_classification, Dataset))

    def test_dataset_values_classification(self):
        """Test that the downloaded dataset is the same as in OpenML."""
        # Expected values (hardcoded from OpenML)
        expected_n_features = 60
        expected_n_labels = 3
        expected_composed_id = "46-splice"
        expected_target_header = "target"

        # Check the n_labels is correct
        self.assertTrue(
            self.dataset_classification.n_labels == expected_n_labels
        )

        # Check the n_features is correct
        self.assertTrue(
            self.dataset_classification.n_features == expected_n_features
        )

        # Check the built id is compliant
        self.assertTrue(
            self.dataset_classification.dataset_id == expected_composed_id
        )

        # Check the target column is correct.
        self.assertTrue(
            len(self.dataset_classification.y.columns) == 1 and
            self.dataset_classification.y.columns[0] == expected_target_header
        )

        # Check the problem type is correct
        self.assertTrue(
            not self.dataset_classification.problem_type and
            self.dataset_classification.is_classification_problem() and
            not self.dataset_classification.is_regression_problem()
        )

    # TODO: Test categorical indicators (pick a different dataset to ease the
    # test)

    def test_dataset_values_regression(self):
        """Test that the downloaded dataset is the same as in OpenML."""
        # Expected values (hardcoded from OpenML)
        expected_n_features = 60
        expected_n_labels = 3
        expected_composed_id = "46-splice"
        expected_target_header = "target"

        # Check the n_labels is correct
        self.assertTrue(
            self.dataset_regression.n_labels == expected_n_labels
        )

        # Check the n_features is correct
        self.assertTrue(
            self.dataset_regression.n_features == expected_n_features
        )

        # Check the built id is compliant
        self.assertTrue(
            self.dataset_regression.dataset_id == expected_composed_id
        )

        # Check the target column is correct.
        self.assertTrue(
            len(self.dataset_regression.y.columns) == 1 and
            self.dataset_regression.y.columns[0] == expected_target_header
        )

        # Check the problem type is correct
        self.assertTrue(
            self.dataset_regression.problem_type and
            not self.dataset_regression.is_classification_problem() and
            self.dataset_regression.is_regression_problem()
        )


class TestDataset(unittest.TestCase):
    """Test the Dataset class' behaviour."""

    def setUp(self):
        """Initialize the global variables for the tests."""
        self.dataset_classification = DataLoader.get_openml_dataset(
            openml_id=46,
            problem_type=0
        )
        self.dataset_regression = DataLoader.get_openml_dataset(
            openml_id=46,
            problem_type=0
        )

    def test_type_errors(self):
        """If not a pandas DataFrame, then it should fail with TypeError."""
        with self.assertRaises(TypeError):
            Dataset(dataset_id="test_errors", X="test", y="test")

    def test_value_errors(self):
        """If not the same length, then it should fail with ValueError."""
        custom_x = self.dataset_classification.X.copy()
        custom_y = self.dataset_classification.y.copy()
        with self.assertRaises(ValueError):
            Dataset(dataset_id="test_errors", X=custom_x, y=custom_y[1:])

    def test_from_pandas(self):
        """Test that loading a dataset from pandas object is correct."""
        d_id = "test_errors"
        data = Dataset(
            dataset_id=d_id,
            X=self.dataset_classification.X.copy(),
            y=self.dataset_classification.y.copy(),
            problem_type=0
        )

        self.assertTrue(self.dataset_classification.X.equals(data.X))
        self.assertTrue(self.dataset_classification.y.equals(data.y))
        self.assertTrue(data.is_classification_problem())
        self.assertTrue(data.dataset_id == d_id)

    def test_train_split_returns(self):
        """Test that the returned arrays from the split are correct.

        For correct, we understand that sizes for train/test features and
        target are concordant.
        """
        # pylint: disable=C0103
        X_train, X_test, y_train, y_test = \
            self.dataset_classification.train_test_split()

        # Assert dimensions
        self.assertTrue(X_train.shape[0] == y_train.shape[0])
        self.assertTrue(X_test.shape[0] == y_test.shape[0])

    def test_train_split_randomness_equality(self):
        """Test randomness with an specified value is deterministic.

        We test random_state=100 cause the default is 42 and we want to test
        the parameter also. In a different test we compare inequality with the
        default.
        """
        # pylint: disable=C0103
        X_train_1, X_test_1, y_train_1, y_test_1 = \
            self.dataset_classification.train_test_split(random_state=100)

        X_train_2, X_test_2, y_train_2, y_test_2 = \
            self.dataset_classification.train_test_split(random_state=100)

        # Assert equality of the resulting arrays
        self.assertTrue(X_test_1.equals(X_test_2))
        self.assertTrue(X_train_1.equals(X_train_2))
        self.assertTrue(y_test_1.equals(y_test_2))
        self.assertTrue(y_train_1.equals(y_train_2))

    def test_train_split_randomness_inequality(self):
        """Test randomness with two different values.

        We test random_state=100 vs. default=42. We assert that they give
        different results.
        """
        # pylint: disable=C0103
        X_train_1, X_test_1, y_train_1, y_test_1 = \
            self.dataset_classification.train_test_split(random_state=100)

        X_train_2, X_test_2, y_train_2, y_test_2 = \
            self.dataset_classification.train_test_split()

        # Assert inequality of the resulting arrays
        self.assertFalse(X_test_1.equals(X_test_2))
        self.assertFalse(X_train_1.equals(X_train_2))
        self.assertFalse(y_test_1.equals(y_test_2))
        self.assertFalse(y_train_1.equals(y_train_2))

    def test_train_split_percentage_default(self):
        """Test that the default percentage of the split is correct."""
        # pylint: disable=C0103
        _, _, y_train, y_test = \
            self.dataset_classification.train_test_split()

        # Assert test percentage (we are tolerant, but it must be in the
        # interval)
        percentage = y_test.shape[0] / (y_train.shape[0] + y_test.shape[0])
        self.assertTrue(0.32 <= percentage <= 0.34)

    def test_train_split_percentage_custom(self):
        """Test the percentage of the split for a custom value is correct."""
        # pylint: disable=C0103
        _, _, y_train, y_test = \
            self.dataset_classification.train_test_split(test_size=0.5)

        # Assert test percentage (we are tolerant, but it must be in the
        # interval)
        percentage = y_test.shape[0] / (y_train.shape[0] + y_test.shape[0])
        self.assertTrue(0.49 <= percentage <= 0.50)

    def test_meta_features_vector(self):
        """Test that computing a metafeature vector is compliant."""
        meta_vector = self.dataset_classification.metafeatures_vector()
        self.assertTrue(
            meta_vector.shape[1] == 46 and meta_vector.shape[0] == 1
        )

if __name__ == '__main__':
    unittest.main()
