from cvxopt import matrix as cvxopt_matrix
import numpy as np


def generate_softmarginsvm(y, X, X_dash, C):
    """
    :param y: target value
    :param X: Dataset
    :param X_dash: (y.reshape(-1,1) * 1.) * X
    :param C: regularization parameter
    :return: Q, q, G, h, A, b
    """
    # Initializing values and computing H. Note the 1. to force to float type
    m, n = X.shape
    y = y.reshape(-1, 1) * 1.
    H = np.dot(X_dash, X_dash.T) * 1.

    # Converting into cvxopt format
    Q = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    return Q, q, G, h, A, b
