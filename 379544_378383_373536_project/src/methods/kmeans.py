import numpy as np
import itertools


class KMeans(object):
    """
    kNN classifier object.
    """

    def __init__(self, K, max_iters):
        """
        Call set_arguments function of this class.
        """
        self.K = K
        self.max_iters = max_iters
        self.centroids = None
        self.best_permutation = None

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.
        Hint:
            (1) Since Kmeans is unsupervised clustering, we don't need the labels for training. But you may want to use it to determine the number of clusters.
            (2) Kmeans is sensitive to initialization. You can try multiple random initializations when using this classifier.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,).
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        N, D = training_data.shape

        # Initialize centroids by selecting K random data points
        random_indices = np.random.choice(N, self.K, replace=False)
        centroids = training_data[random_indices]

        best_centroids = None
        best_permutation = None

        for _ in range(self.max_iters):
            # Distances from each data point to each centroid
            distances = np.linalg.norm(training_data[:, np.newaxis] - centroids, axis=2)

            # Assign each point to the nearest centroid (this is the label)
            labels = np.argmin(distances, axis=1)

            # Update centroids as the mean of the assigned points
            new_centroids = np.array([training_data[labels == k].mean(axis=0) for k in range(self.K)])

            # If centroids do not change, break the loop
            if np.allclose(centroids, new_centroids):  # => Convergence
                break

            centroids = new_centroids

        # Store the best centroids and permutation found
        self.centroids = centroids
        self.best_permutation = labels

        # Return the predicted labels based on the best permutation
        pred_labels = labels  # These are the cluster assignments for each data point
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        # Calculate distances from each test point to each centroid
        # Distances is of shape (N, K) where N is the number of test points and K is the number of centroids
        # With each element being the distance from the test point to the centroid
        distances = np.linalg.norm(test_data[:, np.newaxis] - self.centroids, axis=2)

        # Assign each test point to the nearest centroid
        test_labels = np.argmin(distances, axis=1)

        # Apply the best permutation found during training
        test_labels = np.array([self.best_permutation[label] for label in test_labels])

        return test_labels