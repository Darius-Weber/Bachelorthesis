from cvxopt import matrix as cvxopt_matrix
import numpy as np

def generate_softmarginsvm(y, X, C):
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
    X_dash = y * X  # X_dash * X_dash^T = Q
    H = np.dot(X_dash, X_dash.T) * 1.

    # Converting into cvxopt format
    Q = cvxopt_matrix(H)
    q = cvxopt_matrix(-np.ones((m, 1)))
    G = cvxopt_matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
    h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = cvxopt_matrix(y.reshape(1, -1))
    b = cvxopt_matrix(np.zeros(1))
    return Q, q, G, h, A, b, X_dash

def generate_markowitz_portfolio_optimization(returns, r_min, pbar):
    cov = np.array(np.cov(returns.T))
    N = returns.shape[1]
    S = cvxopt_matrix(np.linalg.cholesky(cov)) # Decomposition of covariance matrix
    Q = cvxopt_matrix(cov)
    q = cvxopt_matrix(np.zeros((N, 1)))
    G = cvxopt_matrix(np.concatenate((-pbar, -np.identity(N)), 0))
    A = cvxopt_matrix(1.0, (1, N))
    b = cvxopt_matrix(1.0)
    h = cvxopt_matrix(np.concatenate((-np.ones((1, 1)) * r_min, np.zeros((N, 1))), 0))
    return Q, q, G, h, A, b, S