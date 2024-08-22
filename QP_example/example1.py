from cvxopt import matrix
import numpy as np
import sys
import os

# Add the parent directory of 'solver' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from solver.qp import qp 

S2 = np.array([[2, 0], [1, 3]])
# Compute Q from S*S^T
Q2 = S2 @ S2.T # Objective function quadratic part
A2 = np.array([[1, 1]])  # Equality constraint
b2 = np.array([1])        
G2 = np.array([[-1, -1]])  # Inequality constraint
h2 = np.array([[0]])       
q2 = np.array([3, 0])   # Objective function linear part

Q = matrix(Q2, tc='d') 
A = matrix(A2, tc='d')  
b = matrix(b2, tc='d')  
G = matrix(G2, tc='d')  
h = matrix(h2, tc='d')  
q = matrix(q2, (2, 1), tc='d')  

sol = qp(Q, q, G, h, A, b, callback=lambda res: res)
print(sol['x'])