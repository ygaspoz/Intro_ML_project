import numpy as np
from ..methods.logistic_regression import LogisticRegression
from ..helpers.load_dataset import load
from ..utils import accuracy_fn, macrof1_fn, mse_fn

def run(model, x_train, y_train, x_test, y_test):
    """
    Run the model with given training and testing data.

    Args:
        model (LogisticRegression): The logistic regression model.
        x_train (np.array): Training data.
        y_train (np.array): Training labels.
        x_test (np.array): Testing data.
        y_test (np.array): Testing labels.

    Returns:
        tuple: Predictions for training and testing data.
    """
    preds_train = model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc_train = accuracy_fn(preds_train, y_train)
    acc_test = accuracy_fn(preds, y_test)
    macrof1_train = macrof1_fn(preds_train, y_train)
    macrof1_test = macrof1_fn(preds, y_test)
    mse_train = mse_fn(preds_train, y_train)
    mse_test = mse_fn(preds, y_test)
    return acc_train, acc_test, macrof1_train, macrof1_test, mse_train, mse_test


def evaluate_hyper_parameter(parameter_type, start, end, step, default_secondary_param=1):
    """
    Evaluate hyper-parameters for logistic regression.

    Args:
        parameter_type (str): The type of hyper-parameter to evaluate.
        start (float): The starting value of the hyper-parameter.
        end (float): The ending value of the hyper-parameter.
        step (float): The step size for the hyper-parameter.

    Returns:
        dict: A dictionary containing the evaluated hyper-parameters and their corresponding scores.
    """
    assert parameter_type in ['max_iters', 'lr'], "Invalid parameter type"
    assert start >= 0 and start <= end, "Invalid parameter start"
    assert step > 0, "Invalid parameter step"

    # Assuming `model` is an instance of the LogisticRegression class
    # and has methods to set the hyper-parameters and evaluate the model.
    parameter_values = np.arange(start, end + step, step)
    x_train, x_test, y_train, y_test = load('../data')

    if parameter_type == 'max_iters':
        results = {param: [] for param in parameter_values}
        for param in parameter_values:
            model = LogisticRegression(lr=default_secondary_param, max_iters=int(param))
            acc_train, acc_test, macrof1_train, macrof1_test, mse_train, mse_test = run(model, x_train, y_train, x_test,
                                                                                        y_test)
            results[param].append({
                'train_accuracy': acc_train,
                'test_accuracy': acc_test,
                'train_macrof1': macrof1_train,
                'test_macrof1': macrof1_test,
                'train_mse': mse_train,
                'test_mse': mse_test
            })
    else:
        results = {param: [] for param in parameter_values}
        for param in parameter_values:
            model = LogisticRegression(lr=float(param), max_iters=default_secondary_param)
            acc_train, acc_test, macrof1_train, macrof1_test, mse_train, mse_test = run(model, x_train, y_train, x_test,
                                                                                        y_test)
            results[param].append({
                'train_accuracy': acc_train,
                'test_accuracy': acc_test,
                'train_macrof1': macrof1_train,
                'test_macrof1': macrof1_test,
                'train_mse': mse_train,
                'test_mse': mse_test
            })
    return results