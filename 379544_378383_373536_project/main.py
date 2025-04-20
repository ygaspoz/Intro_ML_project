import argparse

import numpy as np

from src.tuning.tune_knn import tune_knn
from src.helpers.kmeans_helper import elbow_plot, tune_kmeans
from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.knn import KNN
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, train_test_split, plot_confusion_matrix
import os
import time

np.random.seed(100)

# If the user does not specify a tuned K for KMeans, this default will be used when tuning.
IDEAL_K_KMeans = 4
# If the user does not specify a k for KNN, this default will be used
IDEAL_K_KNN = 3

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
        xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)

    # Normalize the data
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

    if args.method == "kmeans" and args.K == 1:
        args.K = IDEAL_K_KMeans  # Default value for KMeans if not specified

    if args.method == "kmeans" and args.tune_kmeans:
        best_k, best_f1 = tune_kmeans(xtrain, ytrain, args.K, args.max_iters)
        print(f"Tuned KMeans: best k = {best_k}, best macro-F1 = {best_f1:.6f}")
        method_obj = KMeans(k=best_k, max_iters=args.max_iters)


    if args.method == "kmeans" and args.elbow:
        elbow_plot(xtrain, ytrain, args.K, args.max_iters)
        return

    if args.method == "knn" and args.K == 1:
        args.K = IDEAL_K_KNN

    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("NN not implemented in MS1.")

    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)
    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters, verbose=args.verbose)
    elif args.method == "kmeans":
        method_obj = KMeans(k=args.K, max_iters=args.max_iters, n_init=args.n_init)
    elif args.method == "knn":
        if args.tune_knn:
            print("Tuning k for KNN...")
            k_values = list(range(1, 22, 2)) # Only odd values
            method_obj = tune_knn(xtrain, ytrain, xtest, ytest, k_values)
            print("Best k for KNN: ", method_obj.k)
        else:
            method_obj = KNN(k=args.K)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method
    start_time_train = time.time()
    preds_train = method_obj.fit(xtrain, ytrain)
    end_time_train = time.time()

    start_time_predict = time.time()
    preds = method_obj.predict(xtest)
    end_time_predict = time.time()

    acc = accuracy_fn(preds_train, ytrain)
    macrof1_train = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1_train:.6f} - time = {end_time_train - start_time_train:.3f}s")

    acc = accuracy_fn(preds, ytest)
    macrof1_score = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1_score:.6f} - time = {end_time_predict - start_time_predict:.3f}s")

    classes = np.unique(ytrain)
    ytest = np.array(ytest, dtype=int)
    preds = np.array(preds, dtype=int)
    plot_confusion_matrix(ytest, preds, classes=classes)

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
        "--K", type=int, default=1,
        help="number of neighboring datapoints used for knn OR number of clusters for kmeans"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=700,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, otherwise use a validation set",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
        help="print stats every N iterations (0 for no verbose output)",
    )
    parser.add_argument(
        "--elbow",
        action="store_true",
        help="plot inertia vs. k for KMeans and exit if set"
    )
    parser.add_argument(
        "--tune_kmeans",
        action="store_true",
        help="automatically tune K for KMeans via macro F1 on validation"
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=10,
        help="number of random initializations for KMeans (default: 10)"
    )
    parser.add_argument(
        "--tune_knn",
        action="store_true",
        help="automatically tune k for KNN using validation accuracy"
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

    args = parser.parse_args()

    main(args)
