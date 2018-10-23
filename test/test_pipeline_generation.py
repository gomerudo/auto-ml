import unittest
import os.path
from automl.datahandler.dataloader import Dataset, DataLoader
from automl.discovery.pipeline_generation import PipelineDiscovery

class TestPipelineDiscovery(unittest.TestCase):

    def test_discovery(self):
        dataset = DataLoader.get_openml_dataset(openml_id=46, problem_type=0)
        p_disc = PipelineDiscovery(dataset=dataset)
        pipeline = p_disc.discover()

        print(pipeline)
        print(p_disc.validation_score)

