from .activation_functions import f_softmax
import numpy as np


def gradient_logistic_multi(data: np.array, labels: np.array, w: np.array):
    """
    Compute the gradient of the entropy for multi-class logistic regression.

    Args:
        data (array): Input data of shape (N, D)
        labels (array): Labels of shape  (N, C)  (in one-hot representation)
        w (array): Weights of shape (D, C)
    Returns:
        grad (np.array): Gradients of shape (D, C)
    """
    return data.T @ (f_softmax(data, w) - labels)