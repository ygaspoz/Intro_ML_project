# Method to train a kMeans classifier.
#
# Author(s):
# - Ilias Bouraoui <ilias.bouraoui@epfl.ch>
# - Leonardo Matteo Bolognese <leonardo.bolognese@epfl.ch>
# Version: 20250723

"""Method to train a kMeans classifier."""

import numpy as np
from ..helpers.normalize_functions import normalize


class KMeans(object):
    """
    kMeans classifier object.
    """

    def __init__(self, k=10, max_iters=100, n_init=10):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.max_iters = max_iters
        # number of random starts; we'll keep the run with the lowest inertia to find a better clustering
        self.n_init = n_init

        self.centroids = None
        self.cluster_to_label = None
        self.sum_of_squared_distances = None # inertia of best run

        self.mean = None
        self.std = None

    def fit(self, training_data, training_labels):
        """
        Trains the KMeans model, returns predicted labels for the training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)

        Returns:
            pred_labels (np.array): predicted labels for the training data (N,)
        """
        N, D = training_data.shape
        training_data, self.mean, self.std  = normalize(training_data)

        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_cluster_to_label = None

        global_majority = np.bincount(training_labels.astype(int)).argmax()  # fallback label if cluster empty

        for _ in range(self.n_init):
            picks = np.random.choice(N, self.k, replace=False)
            centroids = training_data[picks].copy()

            for _ in range(self.max_iters):
                distances = np.linalg.norm(training_data[:, np.newaxis] - centroids, axis=2)
                labels = np.argmin(distances, axis=1)

                new_centroids = centroids.copy()
                for cluster_idx in range(self.k):
                    members = training_data[labels == cluster_idx]
                    if members.size:
                        new_centroids[cluster_idx] = members.mean(axis=0)
                    else:
                        # leave centroid unchanged to prevent NaN
                        pass

                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids

            inertia = np.sum((training_data - centroids[labels]) ** 2)

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids.copy()
                best_labels = labels.copy()

                mapping = {}
                for cluster_idx in range(self.k):
                    members = training_labels[labels == cluster_idx]
                    if members.size:
                        mapping[cluster_idx] = np.bincount(members.astype(int)).argmax()
                    else:
                        mapping[cluster_idx] = global_majority  # use overall majority for empty cluster
                best_cluster_to_label = mapping

        self.centroids = best_centroids
        self.cluster_to_label = best_cluster_to_label
        self.sum_of_squared_distances = best_inertia  # inertia of best run

        return np.array([self.cluster_to_label[c] for c in best_labels])

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        test_data, _, _ = normalize(test_data, mean=self.mean, std=self.std)
        distances = np.linalg.norm(test_data[:, np.newaxis] - self.centroids, axis=2)
        test_cluster_labels = np.argmin(distances, axis=1)
        return np.array([self.cluster_to_label[label] for label in test_cluster_labels])
