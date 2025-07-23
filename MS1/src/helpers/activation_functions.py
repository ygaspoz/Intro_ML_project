# Activation function for multi-class logistic regression.
#
# Author(s):
# - Yann Gaspoz <yann.gaspoz@epfl.ch>
# Version: 20250723

"""Activation function for multi-class logistic regression."""

import numpy as np

def f_softmax(data: np.array, w: np.array) -> np.array:
    """
    Softmax function for multi-class logistic regression.

    Arguments:
        data (array): of shape (N,D) where N is the number of samples and D the number of features
        w (array): Weights of shape (D,C) where D is the number of features and C the number of classes
    Returns:
        (array): of shape (N, C) with N being the predictions for each class for each sample between 0 and 1
    """
    calc = np.exp(data @ w)
    return calc / np.sum(calc, axis=1, keepdims=True)