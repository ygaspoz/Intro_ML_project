# Helper function to normalize data.
#
# Author(s):
# - Leonardo Matteo Bolognese <leonardo.bolognese@epfl.ch>
# Version: 20250723

"""Helper function to normalize data."""

import numpy as np


def normalize(data, mean=None, std=None):
    """
    Normalize the data to have zero mean and unit variance.
    Args:
        data (np.ndarray): The input data to be normalized.
        mean (np.ndarray): The mean of the data to be normalized
        std (np.ndarray): The standard deviation of the data to be normalized
    Returns:
        np.ndarray: The normalized data.
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)

    std_safe = np.where(std == 0, 1, std)

    return (data - mean) / std_safe, mean, std
