# Ensuring Gradient Privacy in Personalized Heterogeneous Federated Learning Environment

A collection of experiments focused on unifying gradient privacy and device heterogeneity in federated learning.

The benchmark folder contains experiments concerning the benchmark algorithm in our proposed procedure.

Performance contains experiments evaluating the performance of our algorithm, comparing to others state-of-the-art algorithms.

Secure aggregation includes complete code with full gradient privacy.


## Recreation

Each folder contains an `experiments.sh` shell file that allows for the recreation of experiments that are a part of this work.

Extra Python libraries may be installed prior to recreation, each folder contains a `requirements.txt` file specifying the requirements, installation of them is reduced to executing `pip install -r requirements.txt`. The JAX library will also need to be installed seperately by following the guide at https://github.com/google/jax.
