# Ensuring Gradient Privacy in Personalized Heterogeneous Federated Learning Environment

A collection of experiments focused on unifying gradient privacy and device heterogeneity in federated learning.

The benchmark folder contains experiments concerning the benchmark algorithm in our proposed procedure.

Performance contains experiments evaluating the performance of our algorithm, comparing to others state-of-the-art algorithms.

Secure aggregation includes complete code with full gradient privacy.


## Recreation

Each folder contains an `experiments.sh` shell file that allows for the recreation of experiments that are a part of this work.

We have set up a [poetry](https://python-poetry.org/) environment to handle dependencies. To use it, with poetry installed, run `poetry install` (if you have an nvidia gpu, run `poetry install --with nvidia` instead) to get the dependencies, then `poetry shell` to start the environment. With the environment started, you can then simply run the `experiments.sh` shell files to recreate the experiments.
