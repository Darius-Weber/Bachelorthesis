from cvxopt import matrix as cvxopt_matrix, uniform as cvxopt_uniform
from cvxopt import solvers as cvxopt_solvers
import numpy as np
from matplotlib import pyplot as plt
#https://xavierbourretsicotte.github.io/SVM_implementation.html
#Dataset:
#https://goelhardik.github.io/2016/11/28/svm-cvxopt/

DIM = 2
COLORS = ['red', 'blue']

# 2-D mean of ones
M1 = np.ones((DIM,))
# 2-D mean of threes
M2 = 3 * np.ones((DIM,))
# 2-D covariance of 0.3
C1 = np.diag(0.3 * np.ones((DIM,)))
# 2-D covariance of 0.2
C2 = np.diag(0.2 * np.ones((DIM,)))

def generate_gaussian(m, c, num):
    return np.random.multivariate_normal(m, c, num)

NUM = 3
# generate 50 points from gaussian 1
x1 = generate_gaussian(M1, C1, NUM)
# labels
y1 = np.ones((x1.shape[0],))
# generate 50 points from gaussian 2
x2 = generate_gaussian(M2, C2, NUM)
y2 = -np.ones((x2.shape[0],))
# join
X = np.concatenate((x1, x2), axis = 0)
y = np.concatenate((y1, y2), axis = 0)


#Initializing values and computing H. Note the 1. to force to float type
C = 10
m,n = X.shape
y = y.reshape(-1,1) * 1.
X_dash = y * X
H = np.dot(X_dash , X_dash.T) * 1.

#Converting into cvxopt format - as previously
Q = cvxopt_matrix(H)
q = cvxopt_matrix(-np.ones((m, 1)))
G = cvxopt_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
A = cvxopt_matrix(y.reshape(1, -1))
b = cvxopt_matrix(np.zeros(1))
# print(np.allclose(np.array(Q), np.array(np.dot(X_dash, X_dash.T))))
#Run solver
sol = cvxopt_solvers.qp(Q, q, G, h, A, b)
alphas = np.array(sol['x'])
print("alpha", alphas)
print(Q)
print(A)
print(G)
print(h)
print(b)
#==================Computing and printing parameters===============================#
w = ((y * alphas).T @ X).reshape(-1,1)
S = (alphas > 1e-4).flatten()
b = y[S] - np.dot(X[S], w)

#Display results
print('Alphas = ',alphas[alphas > 1e-4])
print('w = ', w.flatten())
print('b = ', b[0])

# Getting the separate data points
x_pos = X[y.flatten() == 1]
x_neg = X[y.flatten() == -1]

# Generating a grid for contour plotting
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
x1, x2 = np.meshgrid(x1_range, x2_range)
X_range = np.c_[x1.ravel(), x2.ravel()]

# Defining the decision boundary and margins
f = lambda x: np.dot(x.flatten(), w.flatten()) + sum(b)/len(b)
y_range = np.array([f(x) for x in X_range]).reshape(-1)

fig, ax = plt.subplots()

# Plotting the decision boundary and margins
ax.contour(x1, x2, y_range.reshape(x1.shape), colors='k', levels=[-1, 0, 1], alpha=1,
           linestyles=['--', '-', '--'])

# Plotting the data points
ax.scatter(x_pos[:, 0], x_pos[:, 1], marker='o', color='b', label='Positive +1')
ax.scatter(x_neg[:, 0], x_neg[:, 1], marker='x', color='r', label='Negative -1')

# Support Vectors
ax.scatter(X[S][:, 0], X[S][:, 1], s=100, facecolors='none', edgecolors='k', marker='o')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Soft Margin SVM')
plt.legend()
plt.grid(True)
plt.show()
