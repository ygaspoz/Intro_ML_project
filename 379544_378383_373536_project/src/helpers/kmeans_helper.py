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

# Tune k by maximizing macro-F1
def tune_kmeans(X, y, max_k, max_iters):
    """
    Tune KMeans’ k by maximizing macro‑F1 on a validation split.
    Tests k=1…max_k and returns the best k and its F1.
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
    for k in range(1, max_k + 1):
        km = KMeans(k=k, max_iters=max_iters)
        km.fit(X_tr, y_tr)
        preds = km.predict(X_val)
        f1 = macrof1_fn(preds, y_val)
        if f1 > best_f1:
            best_f1 = f1
            best_k = k
    return best_k, best_f1