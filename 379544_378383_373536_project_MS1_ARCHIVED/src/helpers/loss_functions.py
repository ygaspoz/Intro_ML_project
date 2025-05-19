from .activation_functions import f_softmax
import numpy as np

def loss_logistic_multi(data: np.array, labels: np.array, w: np.array):
    """
    Loss function for multi class logistic regression, i.e., multi-class entropy.

    Args:
        data (array): Input data of shape (N, D)
        labels (array): Labels of shape  (N, C)  (in one-hot representation)
        w (array): Weights of shape (D, C)
    Returns:
        float: Loss value
    """
    return -np.sum(labels * np.log(f_softmax(data, w)))