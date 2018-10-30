"""Module to test the model builder."""

import unittest
from automl.discovery.assistant import Assistant
# from automl.metalearning.database.configurations_parsing \
#     import ConfigurationBuilder
# from automl.metalearning.database.load_db import ConfigurationsFile
from automl.datahandler.dataloader import DataLoader
from automl.bayesianoptimizationpiepeline.base \
    import BayesianOptimizationPipeline


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
        print("Generating pipeline...")
        pipeline_obj = assistant.generate_pipeline()
        pipeline = pipeline_obj.pipeline

        bayesian = BayesianOptimizationPipeline(
            dataset.X,
            dataset.y,
            pipeline,
            optimize_on="quality",
            iteration=1)

        score, opt_pipeline = bayesian.optimize_pipeline()
        print("Score", score)
        print(opt_pipeline)
