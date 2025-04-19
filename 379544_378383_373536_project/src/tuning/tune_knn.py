from ..methods.knn import *

def tune_knn(x_train, y_train, x_val, y_val, k_values):
    """
    Tune the K value for KNN using validation data.

    Args:
        x_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        x_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation labels
        k_values (list): List of k values to try

    Returns:
        best_k (int): Best k value based on validation accuracy
        best_accuracy (float): Corresponding validation accuracy
    """
    best_k = None
    best_accuracy = 0.0

    for k in k_values:
        knn = KNN(k=k)
        knn.fit(x_train, y_train)
        predictions = knn.predict(x_val)
        accuracy = np.mean(predictions == y_val) * 100.0
        print(k, " ", accuracy)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    return KNN(best_k)