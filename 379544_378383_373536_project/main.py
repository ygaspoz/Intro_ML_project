import argparse

import numpy as np
from torchinfo import summary
import torch
import matplotlib.pyplot as plt
from src.helpers.training_summary import *
import time
from src.data import load_data
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes, train_test_split, augment_data
from src.helpers.tuning import grid_search_batch_lr
import os


def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """

    # Enable GPU
    if args.check_optimisation:
        print("Checking for GPU optimisation...")
        if torch.cuda.is_available():
            print("CUDA is available!")
            num_gpus = torch.cuda.device_count()
            for gpu in range(num_gpus):
                name = torch.cuda.get_device_name(gpu)
                print(f"To use: {name}, please set the flag --device to cuda:{gpu}")
        else:
            print("CUDA is not available!")
        print("Checking for CPU optimisation...")
        num_cpu_cores = os.cpu_count()
        print(f"Maximum number of CPU cores: {num_cpu_cores}, use --workers {num_cpu_cores} to use the maximum number of CPU cores for data loading.")
        exit()

    if args.verbose:
        print(format_args_to_markdown_table(args))

    ## 1. First, we load our data
    xtrain, xtest, ytrain, y_test = load_data()
    if not args.test:
        xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=0.1, random_state=42)

    if args.augment_data:  #
        print(f"Original training data shape: {xtrain.shape}, labels: {ytrain.shape}")
        xtrain_aug, ytrain_aug = augment_data(xtrain, ytrain,
                                              allow_rotations=args.aug_allow_rotation,
                                              flip_horizontal=args.aug_flip_h,
                                              flip_vertical=args.aug_flip_v)
        print(f"Augmented training data shape: {xtrain_aug.shape}, labels: {ytrain_aug.shape}")
        xtrain = xtrain_aug
        ytrain = ytrain_aug

        indices = np.arange(xtrain.shape[0])
        np.random.shuffle(indices)
        xtrain = xtrain[indices]
        ytrain = ytrain[indices]

    ## 2. Prepare data based on model type
    if args.nn_type == "mlp":
        # Flatten images for MLP
        xtrain = xtrain.reshape(xtrain.shape[0], -1)
        xtest = xtest.reshape(xtest.shape[0], -1)
    elif args.nn_type == "cnn":
        if len(xtrain.shape) == 3:  # If shape is (N, H, W)
            xtrain = xtrain[:, np.newaxis, :, :]  # Add channel dimension
            xtest = xtest[:, np.newaxis, :, :]
        if len(xtrain.shape) == 3:  # If shape is (N, H, W)
            xtrain = xtrain[:, np.newaxis, :, :]
            xtest = xtest[:, np.newaxis, :, :]
        elif len(xtrain.shape) == 4:  # If shape is (N, H, W, C)
            xtrain = np.transpose(xtrain, (0, 3, 1, 2))
            xtest = np.transpose(xtest, (0, 3, 1, 2))

    # Continue with data preparation...
    if args.nn_type == "mlp":  #
        means = np.mean(xtrain, axis=0, keepdims=True)  #
        stds = np.std(xtrain, axis=0, keepdims=True)  #
    else:
        temp_xtrain_flat = xtrain.reshape(xtrain.shape[0], -1)
        means = np.mean(temp_xtrain_flat, axis=0, keepdims=True)
        stds = np.std(temp_xtrain_flat, axis=0, keepdims=True)

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
    n_classes = get_n_classes(ytrain)

    if args.nn_type == "mlp":
        model = MLP(input_size=xtrain.shape[1], n_classes=n_classes, dimensions=args.dimensions)
    elif args.nn_type == "cnn":
        model = CNN(n_classes=n_classes, input_channels=3)

    model.to(args.device)
    summary(model)

    # Trainer object
    method_obj = Trainer(model, lr=args.lr,epochs=args.max_iters, batch_size=args.nn_batch_size,
                         device=args.device, verbose=args.verbose)


    ## 4. Train and evaluate the method
    if args.grid_search_lr_batch:
        grid_search_batch_lr(xtrain, xtest, ytrain, y_test, args, n_classes)

    # Fit (:=train) the method on the training data
    start_time = time.time()
    preds_train = method_obj.fit(xtrain, ytrain)

    # Predict on unseen data
    preds = method_obj.predict(xtest)

    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f} - Training time = {time.time()-start_time:.2f}s")


    ## As there are no test dataset labels, check your model accuracy on validation dataset.
    # You can check your model performance on test set by submitting your test set predictions on the AIcrowd competition.
    acc = accuracy_fn(preds, y_test)
    macrof1 = macrof1_fn(preds, y_test)
    print(f"Validation set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")


    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    # Feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--data', default="dataset", type=str, help="path to your dataset")
    parser.add_argument('--nn_type', default="mlp",
                        help="which network architecture to use, it can be 'mlp' | 'transformer' | 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, help="batch size for NN training")
    parser.add_argument('--check_optimisation', action="store_true", default=False, help="Check what optimisations are possible, run this flag by itself")
    parser.add_argument('--device', type=str, default="cpu",
                        help="Device to use for the training, it can be 'cpu' | 'cuda' | 'mps', with multiple GPUs, add :NUM")
    parser.add_argument('--workers', type=int, default=0, help="number of workers to use for data loading, use optimisation flag to check the max")


    parser.add_argument('--lr', type=float, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true",
                        help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--verbose', action="store_true", help="print training progress per epoch")
    parser.add_argument('--grid_search_lr_batch', action="store_true", help="Grid search over learning rate and batch size")

    parser.add_argument('--augment_data', action="store_true", help="Enable data augmentation for training")
    parser.add_argument('--aug_allow_rotation', action="store_true", default=False,
                        help="Enable 90/180/270 deg rotations for augmentation")
    parser.add_argument('--aug_flip_h', action="store_true", default=True,
                        help="Enable horizontal flip for augmentation")
    parser.add_argument('--aug_flip_v', action="store_true", default=False,
                        help="Enable vertical flip for augmentation")
    parser.add_argument('--dimensions', type=int, nargs='+', default=[512, 256, 128],
                    help="List of hidden layer sizes for MLP, e.g., --dimensions 512 256 128")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()

    default_values = {
        "mlp": {"lr": 1e-3, "nn_batch_size": 128, "max_iters": 70},
        "cnn": {"lr": 1e-3, "nn_batch_size": 200, "max_iters": 200}
    }

    if args.lr == None:
        args.lr = default_values[args.nn_type]["lr"]

    if args.nn_batch_size == None:
        args.nn_batch_size = default_values[args.nn_type]["nn_batch_size"]

    if args.max_iters == None:
        args.max_iters = default_values[args.nn_type]["max_iters"]

    main(args)
