"""Test the dataloader module. Special emphasis in the Dataset class."""

import unittest
from automl.datahandler.dataloader import DataLoader, Dataset


class TestDataLoader(unittest.TestCase):
    """Test the DataLoader class' methods."""
    def setUp(self):
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
        self.dataset_classification = DataLoader.get_openml_dataset(
            openml_id=46,
            problem_type=0
        )
        self.dataset_regression = DataLoader.get_openml_dataset(
            openml_id=46,
            problem_type=0
        )

    # TODO: Test type exceptions raise
    def test_type_errors(self):
        pass

    # TODO: Test load from dataframe
    def test_from_pandas(self):
        pass

    def test_train_split(self):
        pass

    def test_meta_features_vector(self):
        # Get the hardcoded row from auto-sklearn and test that the computation
        # works in a similary way.
        pass

    # TODO: test (maybe) the sparse.
    def test_sparse(self):
        pass

if __name__ == '__main__':
    unittest.main()
