import numpy as np
import itertools


class KMeans(object):
    """
    kNN classifier object.
    """

    def __init__(self, max_iters=500):
        """
        Call set_arguments function of this class.
        """
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
        n_clusters = len(np.unique(training_labels))

        # initialize centroids
        self.centroids = training_data[np.random.choice(N, n_clusters, replace=False)]

        for i in range(self.max_iters):
            distances = np.linalg.norm(
                training_data[:, np.newaxis] - self.centroids, axis=2
            )
            pred_labels = np.argmin(distances, axis=1)

            new_centroids = np.array(
                [
                    training_data[pred_labels == k].mean(axis=0)
                    for k in range(n_clusters)
                ]
            )

            # check for convergence
            if np.all(new_centroids == self.centroids):
                print(f"Converged at iteration {i}...")
                break

            # update centroids
            self.centroids = new_centroids

        # find the best permutation of labels that matches the ground truth
        all_permutations = np.array(list(itertools.permutations(range(n_clusters))))
        best_score = 0
        for permutation in all_permutations:
            score = np.sum(training_labels == permutation[pred_labels])
            if score > best_score:
                best_score = score
                self.best_permutation = permutation
        pred_labels = self.best_permutation[pred_labels]

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """
        distances = np.linalg.norm(test_data[:, np.newaxis] - self.centroids, axis=2)
        test_labels = np.argmin(distances, axis=1)

        # permute the test labels
        test_labels = self.best_permutation[test_labels]

        return test_labels
