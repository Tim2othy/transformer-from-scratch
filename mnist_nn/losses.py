import numpy as np


# loss function and its derivative
def mse(y_true, y_pred):

    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-15))


def cross_entropy_prime(y_true, y_pred):
    # For softmax + cross-entropy, the gradient simplifies to (y_pred - y_true)
    # This is numerically stable and works with one-hot encoded labels
    return (y_pred - y_true) / y_true.shape[0]
