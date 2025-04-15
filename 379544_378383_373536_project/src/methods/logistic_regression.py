import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, accuracy_fn
from ..helpers.gradient_functions import gradient_logistic_multi
from ..helpers.loss_functions import loss_logistic_multi
from ..helpers.activation_functions import f_softmax


def init_weights(D, C):
    """
    Initialize the weights of the model.

    Arguments:
        D (int): number of features
        C (int): number of classes
    Returns:
        (array): random weights of shape (D,C)
    """
    return np.random.normal(0, 0.01, (D, C))


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500, verbose=0):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        assert lr > 0, "Learning rate should be positive"
        assert max_iters > 0, "Number of iterations should be positive"
        self.weights = None
        self.lr = lr
        self.max_iters = max_iters
        self.verbose = verbose

    def fit(self, training_data: np.array, training_labels: np.array):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        D = training_data.shape[1]
        C = get_n_classes(training_labels)

        weights = init_weights(D, C)
        for iteration in range(self.max_iters):
            gradient = gradient_logistic_multi(training_data, label_to_onehot(training_labels, C), weights)
            weights -= self.lr * gradient

            pred_labels = onehot_to_label(f_softmax(training_data, weights))
            if accuracy_fn(pred_labels, training_labels) == 1:
                break
            if self.verbose != 0 and iteration % self.verbose == 0:
                print(f"Iteration {iteration}, accuracy: {accuracy_fn(pred_labels, training_labels)}, loss: {loss_logistic_multi(training_data, label_to_onehot(training_labels, C), weights)}")
        self.weights = weights

        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        pred_labels = onehot_to_label(f_softmax(test_data, self.weights))
        return pred_labels
