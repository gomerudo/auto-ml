"""Module to test the model builder."""

import unittest
from automl.discovery.assistant import Assistant
from automl.datahandler.dataloader import DataLoader
from automl.bayesianoptimizationpiepeline.base \
    import BayesianOptimizationPipeline


class TestAssistant(unittest.TestCase):
    """Test the Assistan features."""

    def test_workflow(self):
        """Test a workflow for the Assistant."""
        # Get dataset
        dataset = DataLoader.get_openml_dataset(openml_id=179, problem_type=0)

        # start assistant
        assistant = Assistant(dataset)

        # Compute similar datasets
        assistant.compute_similar_datasets()
        print("Similar datasets", assistant.similar_datasets)

        # Output the resulting reduced search space
        red_ss = assistant.reduced_search_space

        print("classifiers:", red_ss.classifiers)
        print("encoders:", red_ss.encoders)
        print("rescalers:", red_ss.rescalers)
        print("preprocessors:", red_ss.preprocessors)
        print("imputations:", red_ss.imputations)

        # TPOT pipeline genration
        print("Generating pipeline...")
        pipeline_obj = assistant.generate_pipeline()
        pipeline_obj.save_pipeline(target_dir="results")
        # Save tpot's pipeline
        print(pipeline_obj.validation_score)

        # Run bayestian opt
        pipeline = pipeline_obj.pipeline
        bayesian = BayesianOptimizationPipeline(
            dataset,
            pipeline,
            optimize_on="quality",
            iteration=20)

        score, opt_pipeline = bayesian.optimize_pipeline()

        # Prive bayesian's score
        print("Score", score)
        print(opt_pipeline)

        print("TPOT: {} vs. Bayesian: {}".format(
            pipeline_obj.validation_score,
            score
        ))
