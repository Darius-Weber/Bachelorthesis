from cvxopt import matrix as cvxopt_matrix, spmatrix, blas, lapack
import numpy as np

def split_positive_semidefinite(Q, tol=1e-10):
    # Perform SVD of Q with full precision
    U, Sigma, Vt = np.linalg.svd(Q, full_matrices=False)

    # Construct S = U @ sqrt(diag(Sigma)) @ Vt
    sqrt_sigma = np.sqrt(np.maximum(Sigma, 0))  # Ensure Sigma is non-negative
    S = U @ np.diag(sqrt_sigma)

    # Round elements close to zero
    S[np.abs(S) < tol] = 0
    return S

# Define a function to generate steep U-shaped concave data for 'y'
def generate_u_shaped_data(u):
    # Generate random parameters for frequency and phase shift
    freq = np.random.rand() * 10
    phase = np.random.rand() * 2 * np.pi

    # Define the quadratic polynomial component for the U shape
    poly_component = u ** 2

    # Generate a small sinusoidal component
    sinusoidal_component = 0.1 * np.sin(freq * u + phase)

    # Generate a small noise term
    noise = np.random.normal(loc=0.0, scale=0.2, size=u.shape)

    # Combine the components to create the U-shaped data
    return poly_component + sinusoidal_component + noise


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
    S = cvxopt_matrix(split_positive_semidefinite(cov)) # Decomposition of covariance matrix
    Q = cvxopt_matrix(cov)
    q = cvxopt_matrix(np.zeros((N, 1)))
    G = cvxopt_matrix(np.concatenate((-pbar, -np.identity(N)), 0))
    A = cvxopt_matrix(1.0, (1, N))
    b = cvxopt_matrix(1.0)
    h = cvxopt_matrix(np.concatenate((-np.ones((1, 1)) * r_min, np.zeros((N, 1))), 0))
    return Q, q, G, h, A, b, S

def generate_convex_function_to_data(u_values):
    # Set the seed for the random number generator
    np.random.seed()

    # Generate 'y' values based on 'u'
    y_values = generate_u_shaped_data(u_values)

    # Convert arrays to cvxopt.matrix format
    u = cvxopt_matrix(u_values)
    y = cvxopt_matrix(y_values)

    m = len(u)

    # minimize    (1/2) * || yhat - y ||_2^2
    # subject to  yhat[j] >= yhat[i] + g[i]' * (u[j] - u[i]), j, i = 0,...,m-1
    #
    # Variables  yhat (m), g (m).

    nvars = 2 * m
    # Construct S as an identity matrix of size m
    Q = cvxopt_matrix(0.0, (nvars, nvars))

    # Set the diagonal elements to the desired values
    for i in range(m):
        Q[i, i] = 1.0
    S = split_positive_semidefinite(Q)
    q = cvxopt_matrix(0.0, (nvars, 1))
    q[:m] = -y

    # m blocks (i = 0,...,m-1) of linear inequalities
    #
    #     yhat[i] + g[i]' * (u[j] - u[i]) <= yhat[j], j = 0,...,m-1.

    G = spmatrix([], [], [], (m ** 2, nvars))
    I = spmatrix(1.0, range(m), range(m))
    for i in range(m):
        # coefficients of yhat[i]
        G[list(range(i * m, (i + 1) * m)), i] = 1.0

        # coefficients of g[i]
        G[list(range(i * m, (i + 1) * m)), m + i] = u - u[i]

        # coefficients of yhat[j]
        G[list(range(i * m, (i + 1) * m)), list(range(m))] -= I
    G = cvxopt_matrix(G)
    h = cvxopt_matrix(0.0, (m ** 2, 1))
    A = spmatrix([], [], [], (0, q.size[0]))
    A = cvxopt_matrix(A)
    b = cvxopt_matrix(0.0, (0, 1))
    #dummy equality constraint
    return Q, q, G, h, A, b, S

def generate_regressor_selection(alpha_index, m, n):

    # Data for QP
    #
    #     minimize    (1/2) ||A*x - b|_2^2
    #     subject to  -y <= x <= y
    #                 sum(y) <= alpha
    # Generate random A matrix
    A_obj = cvxopt_matrix(np.random.randn(m, n))

    # Generate random b vector
    b_obj = cvxopt_matrix(np.random.randn(m, 1))

    m, n = A_obj.size
    Q = cvxopt_matrix(0.0, (2 * n, 2 * n))
    Q[:n, :n] = A_obj.T * A_obj

    S = split_positive_semidefinite(np.array(Q))

    q = cvxopt_matrix(0.0, (2 * n, 1))
    q[:n] = -A_obj.T * b_obj
    I = cvxopt_matrix(0.0, (n, n))
    I[::n + 1] = 1.0
    G = cvxopt_matrix([[I, -I, cvxopt_matrix(0.0, (1, n))], [-I, -I, cvxopt_matrix(1.0, (1, n))]])
    h = cvxopt_matrix(0.0, (2 * n + 1, 1))

    xln = cvxopt_matrix(0.0, (n, 1))
    xln[:m] = b_obj
    lapack.gels(+A_obj, xln)

    nopts = 100
    alphas = blas.asum(xln) / (nopts - 1) * cvxopt_matrix(range(1, nopts), tc='d')
    alpha = alphas[alpha_index]  # Any alpha from alphas can be selected

    h[-1] = alpha
    A = spmatrix([], [], [], (0, q.size[0]))
    A = cvxopt_matrix(A)
    b = cvxopt_matrix(0.0, (0, 1))
    return Q, q, G, h, A, b, S