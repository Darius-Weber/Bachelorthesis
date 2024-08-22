from cvxopt import matrix
import numpy as np
import sys
import os

# Add the parent directory of 'solver' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solver.qp import qp 

def is_positive_semidefinite(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    return np.all(eigenvalues >= -1e-10)

def split_positive_semidefinite(Q, tol=1e-10):
    # Perform SVD of Q with full precision
    U, Sigma, Vt = np.linalg.svd(Q, full_matrices=False)

    # Construct S = U @ sqrt(diag(Sigma)) @ Vt
    sqrt_sigma = np.sqrt(np.maximum(Sigma, 0))  # Ensure Sigma is non-negative
    S = U @ np.diag(sqrt_sigma)

    # Round elements close to zero due to floating point errors
    S[np.abs(S) < tol] = 0
    return S

# Define S with nice integer values
S3 = np.array([[2, 0], [1, 3]])
# Compute Q
Q2 = S3 @ S3.T
A2 = np.array([[1, 1]])  # Equality constraint
b2 = np.array([1])        # Ensure that b2 is chosen such that it allows a positive solution
G2 = np.array([[-1, -1]])  # Inequality constraint to ensure positivity
h2 = np.array([[0]])       # Ensure h2 is chosen such that it allows a positive solution
q2 = np.array([3, 0])   # Objective function linear part
print(G2)
print(h2)
print(q2)

if is_positive_semidefinite(Q2):
    print("Q is positive semidefinite")
    S2 = split_positive_semidefinite(Q2)
    S = matrix(S2)
    print(S2)
    print(S2@S2.T)

Q = matrix(Q2, tc='d')  # Ensure Q is of type 'd' (double)
A = matrix(A2, tc='d')  # Ensure A is of type 'd' and floating-point
b = matrix(b2, tc='d')  # Ensure b is of type 'd'
G = matrix(G2, tc='d')  # Ensure G is of type 'd'
h = matrix(h2, tc='d')  # Ensure h is of type 'd' and floating-point
q = matrix(q2, (2, 1), tc='d')  # Ensure q is a column vector of type 'd'

sol = qp(Q, q, G, h, A, b, callback=lambda res: res)
print(sol['x'])