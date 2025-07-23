# Helper functions for KMeans clustering
#
# Author(s):
# - Ilias Bouraoui <ilias.bouraoui@epfl.ch>
# Version: 20250723

"""Helper functions for KMeans clustering."""

import numpy as np

from matplotlib import pyplot as plt
from ..methods.kmeans import KMeans
from ..utils import macrof1_fn

def elbow_plot(X, y, max_k, max_iters):
    """
    Plot inertia vs. k for k = 1..max_k using standard KMeans.
    """
    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(k=k, max_iters=max_iters)
        km.fit(X, y)
        inertias.append(km.sum_of_squared_distances)
    plt.plot(range(1, max_k + 1), inertias, marker='o')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Inertia (sum of squared distances)')
    plt.title('Elbow Method for KMeans')
    plt.grid(True)
    plt.show()

# Tune k by maximizing macro‑F1 and plot the tuning curve
def tune_kmeans(X, y, max_k, max_iters):
    """
    Tune KMeans’ k by maximizing macro‑F1 on a validation split.
    Tests k=1…max_k, plots F1 vs k, and returns the best k and its F1.
    """
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, val_idx = idx[:split], idx[split:]
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    best_k = 1
    best_f1 = -1.0
    f1s = []

    # Evaluate each k
    for k in range(1, max_k + 1):
        km = KMeans(k=k, max_iters=max_iters)
        km.fit(X_tr, y_tr)
        preds = km.predict(X_val)
        f1 = macrof1_fn(preds, y_val)
        f1s.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_k = k

    # Plot the tuning curve
    plt.figure()
    plt.plot(range(1, max_k + 1), f1s, marker='o')
    plt.xlabel("Number of clusters $k$")
    plt.ylabel("Macro‑F1 score")
    plt.title("Validation Macro‑F1 vs Number of clusters $k$")
    plt.grid(True)
    plt.show()

    return best_k, best_f1