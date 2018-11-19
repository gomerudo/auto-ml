How to use the tool
===================================

There are different use cases that can be handled from the tool. Here, we
present the ones we think are the most important and helpful to the Data
Scientist.

Build a Dataset object from an OpenML dataset
---------------------------------------------
.. code-block:: python
   :name: get-openml-dataset

    from automl.datahandler.dataloader import DataLoader

    # Download OpenML dataset 179, and consider it for classification (0)
    dataset = DataLoader.get_openml_dataset(openml_id=179, problem_type=0)

Build a Dataset object from a pandas data frame
-----------------------------------------------
.. code-block:: python
   :name: build-dataset

    from automl.datahandler.dataloader import Dataset

    features_df, target_df = ... # Some data frames

    data = Dataset(
        dataset_id="test-dataset",
        X=features_df,
        y=target_df,
        problem_type=0  # Problem type for classification
    )

Get similar datasets, based on a valid meta-learning metric
-----------------------------------------------------------
.. code-block:: python
   :name: metalearning-hints

    from automl.datahandler.dataloader import DataLoader
    from automl.discovery.assistant import Assistant

    dataset = DataLoader.get_openml_dataset(openml_id=179, problem_type=0)

    # start assistant
    assistant = Assistant(dataset=dataset, 
                          metalearning_metric='accuracy')

    # Compute similar datasets
    datasets, distances = assistant.compute_similar_datasets()

Query the reduced search space, based on the similar datasets
-------------------------------------------------------------
.. code-block:: python
   :name: reduced-searchspace

    from automl.datahandler.dataloader import DataLoader
    from automl.discovery.assistant import Assistant

    dataset = DataLoader.get_openml_dataset(openml_id=179, problem_type=0)

    # start assistant
    assistant = Assistant(dataset=dataset, 
                          metalearning_metric='accuracy')

    # Compute similar datasets
    assistant.compute_similar_datasets()
    reduced_ss = assistant.reduced_search_space

    # Get the reduced search space
    classifiers = reduced_search_space.classifiers
    encoders = reduced_search_space.encoders
    scalers = reduced_search_space.rescalers
    preprocessors = reduced_search_space.preprocessors
    imputations = reduced_search_space.imputations

Discover a Pipeline using the reduced search space
--------------------------------------------------
.. code-block:: python
   :name: pipeline-reduced

    from automl.datahandler.dataloader import DataLoader
    from automl.discovery.assistant import Assistant

    dataset = DataLoader.get_openml_dataset(openml_id=179, problem_type=0)

    # start assistant
    assistant = Assistant(dataset=dataset, 
                          metalearning_metric='accuracy',
                          evaluation_metric='accuracy')

    # Compute similar datasets
    assistant.compute_similar_datasets()

    pipeline_obj = assistant.generate_pipeline()
    pipeline_obj.save_pipeline(target_dir="results")

    # Get the scikit-learn pipeline object
    sklearn_pipeline = pipeline_obj.pipeline

Discover a pipeline from scratch
--------------------------------
.. code-block:: python
   :name: pipeline-scratch

    from automl.datahandler.dataloader import DataLoader
    from automl.discovery.assistant import Assistant

    dataset = DataLoader.get_openml_dataset(openml_id=179, problem_type=0)

    # start assistant
    assistant = Assistant(dataset=dataset, 
                          metalearning_metric='accuracy',
                          evaluation_metric='accuracy')

    pipeline_obj = assistant.generate_pipeline()
    pipeline_obj.save_pipeline(target_dir="results")

    # Get the scikit-learn pipeline object
    sklearn_pipeline = pipeline_obj.pipeline

Optimize a pipeline with Bayesian Optimization
----------------------------------------------
.. code-block:: python
   :name: bayesian-full

    from automl.datahandler.dataloader import DataLoader
    from automl.discovery.assistant import Assistant

    dataset = DataLoader.get_openml_dataset(openml_id=179, problem_type=0)

    # start assistant
    assistant = Assistant(dataset=dataset, 
                          metalearning_metric='accuracy',
                          evaluation_metric='accuracy')

    # Compute similar datasets
    assistant.compute_similar_datasets()

    pipeline_obj = assistant.generate_pipeline()
    pipeline_obj.save_pipeline(target_dir="results")

    # Get the scikit-learn pipeline object
    sklearn_pipeline = pipeline_obj.pipeline

    # Run the optimizer
    assistant.bayesian_optimize()

Optimize any pipeline using Bayesian Optimization
-------------------------------------------------
.. code-block:: python
   :name: bayesian-only

    from automl.datahandler.dataloader import DataLoader
    from automl.discovery.assistant import Assistant

    dataset = DataLoader.get_openml_dataset(openml_id=179, problem_type=0)

    # start assistant
    assistant = Assistant(dataset=dataset, evaluation_metric='accuracy')

    # Get the scikit-learn pipeline object
    my_pipeline = ... # A pipeline

    # Run the optimizer
    assistant.bayesian_optimize(my_pipeline)

