# Modified code from: Stephen P Boyd and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.
# Figure 6.24, page 339.
# Fitting a convex function to given data
# The problem data are different from the book.

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solver import qp
from cvxopt import matrix, spmatrix


# Set the seed for the random number generator
np.random.seed()

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


qp.options['show_progress'] = True
#Number of Datapoints
NUM = 100
# Generate 'u' values (from -2 to 2) for the independent variable
u_values = np.linspace(-2.0, 2.0, num=NUM).reshape(-1, 1)

# Generate 'y' values based on 'u'
y_values = generate_u_shaped_data(u_values)

u = matrix(u_values)
y = matrix(y_values)

m = len(u)

# minimize    (1/2) * || yhat - y ||_2^2
# subject to  yhat[j] >= yhat[i] + g[i]' * (u[j] - u[i]), j, i = 0,...,m-1
#
# Variables  yhat (m), g (m).

nvars = 2*m
# Construct S as an identity matrix of size m
P = matrix(0.0, (nvars, nvars))

# Set the diagonal elements to the desired values
for i in range(m):
    P[i, i] = 1.0
q = matrix(0.0, (nvars,1))
q[:m] = -y

# m blocks (i = 0,...,m-1) of linear inequalities
#
#     yhat[i] + g[i]' * (u[j] - u[i]) <= yhat[j], j = 0,...,m-1.

G = spmatrix([],[],[], (m**2, nvars))
I = spmatrix(1.0, range(m), range(m))
for i in range(m):
    # coefficients of yhat[i]
    G[list(range(i*m, (i+1)*m)), i] = 1.0

    # coefficients of g[i]
    G[list(range(i*m, (i+1)*m)), m+i] = u - u[i]

    # coefficients of yhat[j]
    G[list(range(i*m, (i+1)*m)), list(range(m))] -= I

h = matrix(0.0, (m**2,1))
A = spmatrix([], [], [], (0, q.size[0]))
b = matrix(0.0, (0,1))
res = qp.qp(P, q, G, h, A, b, callback=lambda res: res)
c=1
showiter = 3
test = 0
for sol in res['intermediate']:

    if (c-1)%showiter==0 or (c-1)==len(res['intermediate'])-1:
        test+=1
        yhat = np.array(sol['x'][:m]).flatten()
        g = np.array(sol['x'][m:]).flatten()

        nopts = 1000
        ts = np.linspace(min(u_values), max(u_values), nopts)
        f = [max(yhat + g * (t - u_values.flatten())) for t in ts]

        try: import pylab
        except ImportError: pass
        else:
            pylab.figure(1, facecolor='w')
            pylab.plot(u_values, y_values, 'wo', markeredgecolor='b')
            pylab.plot(ts, f, '-g')
            pylab.axis([min(u_values)-1, max(u_values)+1, min(y_values)-1, max(y_values)+1])
            pylab.title("Fitting a convex function to given data")
            pylab.show()
            
    c+=1
print(test)