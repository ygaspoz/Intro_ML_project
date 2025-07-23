# Module to train a kNN classifier.
#
# Author(s):
# - Leonardo Matteo Bolognese <leonardo.bolognese@epfl.ch>
# Version: 20250723

"""Module to train a kNN classifier."""

import numpy as np
from ..helpers.knn_helper import *


class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=3, task_kind="classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        self.x_train = None
        self.mean = None
        self.std = None
        self.y_train = None

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.x_train, self.mean, self.std = normalize(training_data)
        self.y_train = training_labels
        pred_labels = self.predict(training_data)

        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        normalized, _, _ = normalize(test_data, mean=self.mean, std=self.std)
        test_labels = np.apply_along_axis(knn_helper, 1, normalized, self.x_train, self.y_train, self.k)

        return test_labels
