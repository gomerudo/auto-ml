Installing the tool
===================================

-------------
Requirements
-------------

In order to work, you need to verify the next requirements are satisfied

- Work in a UNIX system: SMAC3 is one of our core dependencies and has no official support for windows.
- Install SWIG>=3.0 for your unix system
- In general, make sure to satisfy any dependency descriped in `TPOT's <https://epistasislab.github.io/tpot/installing/>`_ and `SMAC3's <https://github.com/automl/SMAC3>`_ official docs

-------------
Installation
-------------

As of today, this tool is not available in the PyPi repository because of its
immature state. However, it is possible to install it by running the command
below.

.. code-block:: bash
   :name: install

    pip install git+https://github.com/gomerudo/auto-ml.git
