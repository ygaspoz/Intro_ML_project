import numpy as np

class KMeans(object):
    """
    kMeans classifier object.
    """

    def __init__(self, k=10, max_iters=100):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.cluster_to_label = None
        self.sum_of_squared_distances = None

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

        best_sum_of_squared_distances = np.inf
        best_centroids = None
        best_labels = None
        best_cluster_to_label = None

        # Try multiple random initializations
        for _ in range(10):
            # Initialize centroids by selecting K random data points
            random_indices = np.random.choice(N, self.k, replace=False)
            centroids = training_data[random_indices]

            for _ in range(self.max_iters):
                # Distances from each data point to each centroid
                distances = np.linalg.norm(training_data[:, np.newaxis] - centroids, axis=2)

                # Assign each point to the nearest centroid (this is the label)
                labels = np.argmin(distances, axis=1)

                # Update centroids as the mean of the assigned points
                new_centroids = np.array([training_data[labels == k].mean(axis=0) for k in range(self.k)])

                # If centroids do not change, break the loop
                if np.allclose(centroids, new_centroids):  # => Convergence
                    break

                centroids = new_centroids

            # Compute "inertia" (sum of squared distances to centroids)
            sum_of_squared_distances = np.sum((training_data - centroids[labels]) ** 2)

            # If this initialization is better (lower inertia), keep it
            if sum_of_squared_distances < best_sum_of_squared_distances:
                best_sum_of_squared_distances = sum_of_squared_distances
                best_centroids = centroids
                best_labels = labels

                # Map each cluster to the most frequent class label in it
                cluster_to_label = {}
                for k in range(self.k):
                    if np.any(labels == k):
                        cluster_labels = training_labels[labels == k]
                        most_common_label = np.bincount(cluster_labels.astype(int)).argmax()
                        cluster_to_label[k] = most_common_label
                    else:
                        cluster_to_label[k] = -1

                best_cluster_to_label = cluster_to_label

        self.centroids = best_centroids
        self.cluster_to_label = best_cluster_to_label

        pred_labels = np.array([self.cluster_to_label[label] for label in best_labels])
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
        distances = np.linalg.norm(test_data[:, np.newaxis] - self.centroids, axis=2)

        # Assign each test point to the nearest centroid
        test_cluster_labels = np.argmin(distances, axis=1)

        # Map each test point's cluster to the correct class label
        test_labels = np.array([self.cluster_to_label[label] for label in test_cluster_labels])

        return test_labels