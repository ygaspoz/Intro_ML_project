import argparse

import numpy as np
from torchinfo import summary
import torch
import matplotlib.pyplot as plt

from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes, train_test_split


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """

    if args.check_gpu:
        if torch.cuda.is_available():
            print("CUDA is available! Running on GPU ...")
        else:
            print("CUDA is not available! Running on CPU ...")


    ## 1. First, we load our data
    xtrain, xtest, ytrain, y_test = load_data()

    ## 2. Prepare data based on model type
    if args.nn_type == "mlp":
        # Flatten images for MLP
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)
    elif args.nn_type == "cnn":
        if len(xtrain.shape) == 3:  # If shape is (N, H, W)
            xtrain = xtrain[:, np.newaxis, :, :]  # Add channel dimension
            xtest = xtest[:, np.newaxis, :, :]
        # With:
        if len(xtrain.shape) == 3:  # If shape is (N, H, W)
            xtrain = xtrain[:, np.newaxis, :, :]
            xtest = xtest[:, np.newaxis, :, :]
        elif len(xtrain.shape) == 4:  # If shape is (N, H, W, C)
            # Transpose from (N, H, W, C) to (N, C, H, W)
            xtrain = np.transpose(xtrain, (0, 3, 1, 2))
            xtest = np.transpose(xtest, (0, 3, 1, 2))

    # Continue with data preparation...
    means = np.mean(xtrain.reshape(xtrain.shape[0], -1), axis=0, keepdims=True)
    stds = np.std(xtrain.reshape(xtrain.shape[0], -1), axis=0, keepdims=True)

    # Normalize differently based on model type
    if args.nn_type == "mlp":
        xtrain = normalize_fn(xtrain, means, stds)
        xtest = normalize_fn(xtest, means, stds)
        xtrain = append_bias_term(xtrain)
        xtest = append_bias_term(xtest)
    else:
        # For CNN, reshape means/stds and normalize while preserving dimensions
        flat_xtrain = xtrain.reshape(xtrain.shape[0], -1)
        flat_xtrain = normalize_fn(flat_xtrain, means, stds)
        xtrain = flat_xtrain.reshape(xtrain.shape)

        flat_xtest = xtest.reshape(xtest.shape[0], -1)
        flat_xtest = normalize_fn(flat_xtest, means, stds)
        xtest = flat_xtest.reshape(xtest.shape)

    # Prepare the model (and data) for Pytorch
    # Note: you might need to reshape the data depending on the network you use!
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        model = MLP(input_size=xtrain.shape[1], n_classes=n_classes)
    elif args.nn_type == "cnn":
        model = CNN(n_classes=n_classes, input_channels=3)

    model.to(args.device)
    summary(model)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr,epochs=args.max_iters, batch_size=args.nn_batch_size,
                         device=args.device, verbose=args.verbose)


    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    if args.test:
        true_labels = y_test     # from your load_data() unpacking
    else:
        true_labels = y_test      # from your train_test_split()

    acc = accuracy_fn(preds, true_labels)
    macrof1 = macrof1_fn(preds, true_labels)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    if args.grid_search_lr_batch:
        import seaborn as sns
        import matplotlib.pyplot as plt

        learning_rates = [1e-4, 1e-3, 1e-2, 1e-1]
        batch_sizes = [16, 32, 64, 128]
        results = np.zeros((len(batch_sizes), len(learning_rates)))  # Rows = batch size, Cols = learning rate

        for i, bs in enumerate(batch_sizes):
            for j, lr in enumerate(learning_rates):
                print(f"\n Training with batch size = {bs}, learning rate = {lr}")

                if args.nn_type == "mlp":
                    model = MLP(input_size=xtrain.shape[1], n_classes=n_classes)
                else:
                    model = CNN(n_classes=n_classes, input_channels=3)

                model.to(args.device)
                trainer = Trainer(model, lr=lr, epochs=args.max_iters, batch_size=bs,
                                  device=args.device, verbose=False)

                trainer.fit(xtrain, ytrain)
                preds_val = trainer.predict(xtest)
                acc_val = accuracy_fn(preds_val, y_test)

                results[i, j] = acc_val
                print(f"Validation Accuracy = {acc_val:.2f}%")

        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(results, annot=True, fmt=".1f", cmap="viridis",
                    xticklabels=learning_rates, yticklabels=batch_sizes)
        plt.xlabel("Learning Rate")
        plt.ylabel("Batch Size")
        plt.title("Validation Accuracy (%) — Grid Search")
        plt.tight_layout()
        plt.show()

        # Print best combo
        best_idx = np.unravel_index(np.argmax(results), results.shape)
        best_bs = batch_sizes[best_idx[0]]
        best_lr = learning_rates[best_idx[1]]
        best_acc = results[best_idx]
        print(f"\n Best combination: Batch Size = {best_bs}, Learning Rate = {best_lr} → Accuracy = {best_acc:.2f}%")


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")
    parser.add_argument('--check_gpu', action="store_true", default=False, help="whether to check the GPU")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps', with multiple GPUs, add :NUM")


    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--verbose', action="store_true", help="print training progress per epoch")
    parser.add_argument('--grid_search_lr_batch', action="store_true", help="Grid search over learning rate and batch size")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
