#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

__dirname = os.path.abspath(os.path.dirname(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="QA4QUBO-BONOM",
    version="1.0.0",
    author="Andrea Bonomi",
    author_email="andrea.bonomi-2@studenti.unitn.it",
    description="Quantum Annealing for solving QUBO Problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bonom/Quantum-Annealing-for-solving-QUBO-Problems",
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    packages=find_packages(),
    include_package_data=True,
    scripts=['QA4QUBO/solver'],
)

