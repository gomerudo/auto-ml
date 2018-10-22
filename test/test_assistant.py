"""Module to test the model builder."""

import unittest
import numpy as np
from automl.discovery.assistant import Assistant
from automl.metalearning.db.configurations_parsing import ConfigurationBuilder
from automl.metalearning.db.load_db import ConfigurationsFile
from automl.datahandler.dataloader import DataLoader


class TestAssistant(unittest.TestCase):
    """Test the Assistan features."""

    def test_workflow(self):
        """Test a workflow for the Assistant."""
        dataset = DataLoader.get_openml_dataset(openml_id=46, problem_type=0)
        assistant = Assistant(dataset)
        assistant.compute_similar_datasets()

        print("Similar datasets", assistant.similar_datasets)
        red_ss = assistant.reduced_search_space

        print("classifiers:", red_ss.classifiers)
        print("encoders:", red_ss.encoders)
        print("rescalers:", red_ss.rescalers)
        print("preprocessors:", red_ss.preprocessors)
        print("imputations:", red_ss.imputations)
