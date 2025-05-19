import numpy as np


def euclidean_dist(example, training_examples):
    """Compute the Euclidean distance between a single example
    vector and all training_examples.

    Inputs:
        example: shape (D,)
        training_examples: shape (NxD)
    Outputs:
        Euclidean distances: shape (N,)
    """
    return np.sqrt(np.sum((training_examples - example) ** 2, axis=1))
