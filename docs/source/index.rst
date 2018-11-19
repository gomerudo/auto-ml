.. auto-ml documentation master file, created by
   sphinx-quickstart on Wed Oct 24 14:51:04 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to auto-ml's documentation!
===================================

This is the official documentation for our Automated Machine Learning solution
in Python.

This package has been developed by students of the Eindhoven University of
Technology (TU/e) for `Achmea <https://www.achmea.nl>`_’s internal use as part
of an internship program, but it is also shared Open Source
`via github <https://github.com/gomerudo/auto-ml>`_ as agreed by all parts.

The package is intended to assist the Data Scientists to solve classification
(and possibly regression) problems by automatically finding pipelines that
include pre-processing, feature engineering and classification/regression
models for a given dataset. This tool does not implement a new algorithm from
scratch to come up with a solution, but instead it tries to gather the most
relevant features from well known approaches that have proved efficient so that
a usable framework can assist Data Scientsts.

In more detail, our solution makes use of the pre-learned meta-knowledge
acquired by `auto-sklearn <https://github.com/automl/auto-sklearn>`_ to find
candidate models for a given dataset,
`TPOT <https://github.com/EpistasisLab/tpot>`_'s Genetic Programming approach
to find pipelines in an automated way and the Bayesian Optimization implemeted
in `SMAC <https://github.com/automl/SMAC3>`_ to fine-tune a given pipeline.

Authors:

- j.gomez.robles@student.tue.nl
- f.a.a.ansari@student.tue.nl

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   AutoML in a nutshell <nutshell>
   Overview of our solution <solution>
   Installing the tool <install>
   Using the tool <howto>
   Results <results>
   API <modules>
   Python basics to understand the tool <python>
   Contribute <contribute>

