import numpy as np
from .distance_functions import *
from .normalize_functions import *
from ..methods.knn import *

def knn_helper(test_norm, x_train, y_train, k):
    """
    Helper function to compute the label prediction for a single test example
    using the k-Nearest Neighbors algorithm.

    Args:
        test_norm (np.ndarray): The normalized test example, shape (D,)
        x_train (np.ndarray): The normalized training data, shape (N, D)
        y_train (np.ndarray): The labels for the training data, shape (N,)
        k (int): Number of nearest neighbors to consider

    Returns:
        int: Predicted label for the test example
    """

    distances = euclidean_dist(test_norm, x_train)

    # Get the indices of the k closest training examples
    nn_indices = np.argsort(distances)[:k]

    # Get the labels of the k nearest neighbors
    neighbor_labels = y_train[nn_indices]

    # Ensure labels are integers for bincount
    neighbor_labels = neighbor_labels.astype(int)

    # Count the frequency of each label among neighbors
    freq = np.bincount(neighbor_labels)
    best_label = np.argmax(freq)

    return best_label
