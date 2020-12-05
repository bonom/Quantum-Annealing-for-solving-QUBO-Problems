import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="QA4QUBO-BONOM", # Replace with your own username
    version="0.0.1",
    author="Andrea Bonomi",
    author_email="andrea.bonomi-2@studenti.unitn.it",
    description="Quantum Annealing for solving QUBO Problems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bonom/Quantum-Annealing-for-solving-QUBO-Problems",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

