import numpy as np
from .distance_functions import *
from .normalize_functions import *


def knn_helper(test_norm, x_train, y_train, k):
    distances = euclidean_dist(test_norm, x_train)
    nn_indices = np.argsort(distances)[:k]
    neighbor_labels = y_train[nn_indices]

    neighbor_labels = neighbor_labels.astype(int)
    freq = np.bincount(neighbor_labels)
    best_label = np.argmax(freq)

    return best_label
