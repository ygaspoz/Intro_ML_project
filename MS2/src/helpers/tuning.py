# Tuning file for MS2 project.
#
# Author(s):
# - Yann Gaspoz <yann.gaspoz@epfl.ch>
# - Ilias Bouraoui <ilias.bouraoui@epfl.ch>
# - Leonardo Matteo Bolognese <leonardo.bolognese@epfl.ch>
# Version: 20250723

"""Tuning file for MS2 project."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import accuracy_fn

def grid_search_batch_lr(xtrain, xtest, ytrain, y_test, args, n_classes):
    """
        Performs a grid search over combinations of learning rates, batch sizes, and MLP hidden layer configurations
        (if applicable) to find the best-performing training setup for a given model architecture.
        At the end, it visualizes the validation accuracy as a heatmap and prints the best configuration.

        Parameters:
            xtrain (np.ndarray): Training features.
            xtest (np.ndarray): Test/validation features.
            ytrain (np.ndarray): Training labels.
            y_test (np.ndarray): Test/validation labels.
            args (argparse.Namespace): Parsed command-line arguments including model settings.
            n_classes (int): Number of output classes for classification.
    """
    learning_rates = [1e-5, 1e-4, 1e-3]
    batch_sizes = [16, 32, 64, 128]
    hidden_layer_configs = [[512, 256, 128], [256, 128], [128]]

    results = np.zeros((len(batch_sizes), len(learning_rates)))
    best_acc = 0
    best_config = {}

    for hidden_layers in hidden_layer_configs:
        print(f"\nTesting hidden layer configuration: {hidden_layers}")
        for i, bs in enumerate(batch_sizes):
            for j, lr in enumerate(learning_rates):
                print(f"\n Training with batch size = {bs}, learning rate = {lr}")

                if args.nn_type == "mlp":
                    model = MLP(input_size=xtrain.shape[1],
                                n_classes=n_classes,
                                dimensions=hidden_layers)
                else:
                    model = CNN(n_classes=n_classes, input_channels=3)

                model.to(args.device)
                trainer = Trainer(model, lr=lr, epochs=args.max_iters, batch_size=bs,
                                  device=args.device, verbose=False)

                trainer.fit(xtrain, ytrain)
                preds_val = trainer.predict(xtest)
                acc_val = accuracy_fn(preds_val, y_test)

                print(f"Validation Accuracy = {acc_val:.2f}%")

                if acc_val > best_acc:
                    best_acc = acc_val
                    best_config = {
                        "hidden_layers": hidden_layers,
                        "batch_size": bs,
                        "learning_rate": lr
                    }

                results[i, j] = acc_val

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(results, annot=True, fmt=".1f", cmap="viridis",
                xticklabels=learning_rates, yticklabels=batch_sizes)
    plt.xlabel("Learning Rate")
    plt.ylabel("Batch Size")
    plt.title("Validation Accuracy (%) — Grid Search")
    plt.tight_layout()
    plt.show()

    print(f"\nBest configuration:\n{best_config}\nBest accuracy: {best_acc:.2f}%")
