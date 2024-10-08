# Modified code from: Stephen P Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.
# Figure 6.7, page 311.
# Sparse regressor selection.
#
# The problem data are different from the book.

from cvxopt import blas, lapack, matrix, mul
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import qp
import cvxopt
import numpy as np
from pickle import load
try: import pylab
except ImportError: pylab_installed = False
else: pylab_installed = True

qp.options['show_progress'] = True
# Set dimensions
m = 10  
n = 20 

# Generate random A matrix
A = cvxopt.matrix(np.random.randn(m, n))

# Generate random b vector
b = cvxopt.matrix(np.random.randn(m, 1))
m, n = A.size

# set x[k] to zero if abs(x[k]) <= tol * max(abs(x)).
tol = 1e-1

# Data for QP
#
#     minimize    (1/2) ||A*x - b|_2^2
#     subject to  -y <= x <= y
#                 sum(y) <= alpha

P = matrix(0.0, (2*n,2*n))
P[:n,:n] = A.T*A

q = matrix(0.0, (2*n,1))
q[:n] = -A.T*b
I = matrix(0.0, (n,n))
I[::n+1] = 1.0
G = matrix([[I, -I, matrix(0.0, (1,n))], [-I, -I, matrix(1.0, (1,n))]])
h = matrix(0.0, (2*n+1,1))



# Least-norm solution
xln = matrix(0.0, (n,1))
xln[:m] = b
lapack.gels(+A, xln)


nopts = 100
res = [ blas.nrm2(b) ]
card = [ 0 ]
alphas = blas.asum(xln)/(nopts-1) * matrix(range(1,nopts), tc='d')
for alpha in alphas:

    #    minimize    ||A*x-b||_2
    #    subject to  ||x||_1 <= alpha

    h[-1] = alpha
    
    x = qp.qp(P, q, G, h)['x'][:n]
    xmax = max(abs(x))
    I = [ k for k in range(n) if abs(x[k]) > tol*xmax ]
    if len(I) <= m:
        xs = +b
        lapack.gels(A[:,I], xs)
        x[:] = 0.0
        x[I] = xs[:len(I)]
        res += [ blas.nrm2(A*x-b) ]
        card += [ len(I) ]

# Eliminate duplicate cardinalities and make staircase plot.
res2, card2 = [], []
for c in range(m+1):
    r = [ res[k] for k in range(len(res)) if card[k] == c ]
    if r:
        res2 += [ min(r), min(r) ]
        card2 += [ c, c+1 ]

if pylab_installed:
    pylab.figure(1, facecolor='w')
    pylab.plot( res2[::2], card2[::2], 'go' )
    pylab.plot( res2, card2, 'g-')

    pylab.xlabel(r'$||\boldsymbol{A}\boldsymbol{x}-\boldsymbol{b}||_2$')
    pylab.ylabel(r'$\mathbf{card}(\boldsymbol{x})$')
    pylab.title('Regressor selection problem')
    pylab.show()