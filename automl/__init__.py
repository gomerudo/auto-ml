"""Automated Machine Learning tool for Python.

This package has been developed by students of the Eindhoven University of
Technology (TU/e) for Achmea's internal use. It is intended to assist the Data
Scientists to solve classification (and possibly regression) problems by
automatically finding pipelines that include pre-processing, feature
engineering and classification/regression models to solve a dataset.

In more detail, our solution makes use of the pre-learned meta-knowledge
acquired by `auto-sklearn <https://github.com/automl/auto-sklearn>`_ to find
candidate models for a given dataset,
`TPOT <https://github.com/EpistasisLab/tpot>`_'s Genetic Programming approach
to find pipelines in an automated way and the Bayesian Optimization implemeted
in `SMAC <https://github.com/automl/SMAC3>`_ to fine-tune a given pipeline.

Points of contact at TU/e:

- j.gomez.robles@student.tue.nl
- f.a.a.ansari@student.tue.nl
- j.vanschoren@tue.nl

Points of contact at Achmea:

- leon.vink@achmea.nl
- cyril.cleven@achmea.nl
"""
