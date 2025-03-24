import argparse

import numpy as np

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, train_test_split
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data

    # EXTRACTED FEATURES DATASET
    num_indices = [0, 3, 4, 7, 9, 11]
    cat_indices = [1, 2, 5, 6, 8, 10, 12]

    if args.data_type == "features":
        feature_data = np.load(os.path.join(args.data_path, "features.npz"), allow_pickle=True)
        xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
        ytrain, ytest = feature_data["ytrain"], feature_data["ytest"]

    # ORIGINAL IMAGE DATASET (MS2)
    elif args.data_type == "original":
        data_dir = os.path.join(args.data_path, "dog-small-64")
        xtrain, xtest, ytrain, ytest = load_data(data_dir)

    ## 2. Then we must prepare it. This is where you can create a validation set, normalize, add bias, etc.
    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)

        # Normalize only the numerical features
        means = np.mean(xtrain[:, num_indices], axis=0, keepdims=True)
        stds = np.std(xtrain[:, num_indices], axis=0, keepdims=True)
        xtrain[:, num_indices] = normalize_fn(xtrain[:, num_indices], means, stds)
        xval[:, num_indices] = normalize_fn(xval[:, num_indices], means, stds)
        xtest[:, num_indices] = normalize_fn(xtest[:, num_indices], means, stds)

        # Append bias term only to the numerical features
        xtrain_num = append_bias_term(xtrain[:, num_indices])
        xval_num = append_bias_term(xval[:, num_indices])
        xtest_num = append_bias_term(xtest[:, num_indices])

        # Concatenate the processed numerical features back with the categorical features
        xtrain = np.concatenate([xtrain_num, xtrain[:, cat_indices]], axis=1)
        xval = np.concatenate([xval_num, xval[:, cat_indices]], axis=1)
        xtest = np.concatenate([xtest_num, xtest[:, cat_indices]], axis=1)
    else:
        means = np.mean(xtrain[:, num_indices], axis=0, keepdims=True)
        stds = np.std(xtrain[:, num_indices], axis=0, keepdims=True)
        xtrain[:, num_indices] = normalize_fn(xtrain[:, num_indices], means, stds)
        xtest[:, num_indices] = normalize_fn(xtest[:, num_indices], means, stds)

        xtrain_num = append_bias_term(xtrain[:, num_indices])
        xtest_num = append_bias_term(xtest[:, num_indices])

        xtrain = np.concatenate([xtrain_num, xtrain[:, cat_indices]], axis=1)
        xtest = np.concatenate([xtest_num, xtest[:, cat_indices]], axis=1)


    ### WRITE YOUR CODE HERE to do any other data processing

    ## 3. Initialize the method you want to use.

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters, verbose=args.verbose)

    ## 4. Train and evaluate the method
    # Fit (:=train) the method on the training data for classification task
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    # Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / knn / logistic_regression / kmeans / nn (MS2)",
    )
    parser.add_argument(
        "--data_path", default="data", type=str, help="path to your dataset"
    )
    parser.add_argument(
        "--data_type", default="features", type=str, help="features/original(MS2)"
    )
    parser.add_argument(
        "--K", type=int, default=1, help="number of neighboring datapoints used for knn"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, otherwise use a validation set",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print more information during training",
    )

    # MS2 arguments
    parser.add_argument(
        "--nn_type",
        default="cnn",
        help="which network to use, can be 'Transformer' or 'cnn'",
    )
    parser.add_argument(
        "--nn_batch_size", type=int, default=64, help="batch size for NN training"
    )

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
