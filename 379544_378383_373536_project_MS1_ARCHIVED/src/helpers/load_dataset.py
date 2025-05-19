import numpy as np
import os
from ..utils import normalize_fn, append_bias_term

np.random.seed(100)

def load(data_path):
    num_indices = [0, 3, 4, 7, 9, 11]
    cat_indices = [1, 2, 5, 6, 8, 10, 12]

    feature_data = np.load(os.path.join(data_path, "features.npz"), allow_pickle=True)
    xtrain, xtest = feature_data["xtrain"], feature_data["xtest"]
    ytrain, ytest = feature_data["ytrain"], feature_data["ytest"]



    ## 2. Then we must prepare it. This is where you can create a validation set, normalize, add bias, etc.

    # Normalize the data
    means = np.mean(xtrain[:, num_indices], axis=0, keepdims=True)
    stds = np.std(xtrain[:, num_indices], axis=0, keepdims=True)
    xtrain[:, num_indices] = normalize_fn(xtrain[:, num_indices], means, stds)
    xtest[:, num_indices] = normalize_fn(xtest[:, num_indices], means, stds)

    xtrain_num = append_bias_term(xtrain[:, num_indices])
    xtest_num = append_bias_term(xtest[:, num_indices])

    xtrain = np.concatenate([xtrain_num, xtrain[:, cat_indices]], axis=1)
    xtest = np.concatenate([xtest_num, xtest[:, cat_indices]], axis=1)
    return xtrain, xtest, ytrain, ytest