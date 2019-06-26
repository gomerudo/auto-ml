"""Setup the nasgym pkg with setuptools."""

from setuptools import setup, find_packages

setup(
    # For installation
    name='automl',
    version='0.0.1',
    install_requires=['openml', 'tpot', 'auto-sklearn'],
    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),

    # Metadata to display on PyPI
    author="Jorge Gomez Robles",
    author_email="j.gomezrb.dev@gmail.com",
    description="An AutoML framework",
    license="MIT",
    keywords="auto-ml machine-learning smac tpot",
    url="https://gomerudo.github.io/auto-ml",
)
