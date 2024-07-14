import cvxopt as opt
from cvxopt import solvers as cvxopt_solvers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#https://github.com/aghanhussain/Markowitz-Portfolio-Optimization-with-Python/blob/master/Markowitz-Portfolio-Optimization-with-Python.ipynb

def split_positive_semidefinite(Q, tol=1e-10):
    # Perform SVD of Q with full precision
    U, Sigma, Vt = np.linalg.svd(Q, full_matrices=False)

    # Construct S = U @ sqrt(diag(Sigma)) @ Vt
    sqrt_sigma = np.sqrt(np.maximum(Sigma, 0))  # Ensure Sigma is non-negative
    S = U @ np.diag(sqrt_sigma)

    # Round elements close to zero due to floating point errors
    S[np.abs(S) < tol] = 0
    return S
def calculate_frontier(returns, opt_mus_n):
    '''
    returns optimal portfolio weights and corresponding sigmas for a desired optimal portfolio return
    Params:
    - returns: T x N matrix of observed data
    '''
    returns = pd.DataFrame(returns)
    cov = np.array(np.cov(returns.T))
    N = returns.shape[1]
    pbar = np.matrix(returns.mean())
    S = split_positive_semidefinite(cov)
    # print(np.allclose(np.array(cov), np.array(np.dot(S, S.T))))
    #print(cov.size)
    # define list of optimal / desired mus for which we'd like to find the optimal sigmas
    optimal_mus = []
    r_min = pbar.mean()  # minimum expected return
    for i in range(opt_mus_n):
        optimal_mus.append(r_min)
        r_min += (pbar.mean() / 100)
    # constraint matrices for quadratic programming
    P = opt.matrix(cov)
    # print(np.allclose(np.array(P), np.array(np.dot(S, S.T))))
    q = opt.matrix(np.zeros((N, 1)))
    G = opt.matrix(np.concatenate((-pbar, -np.identity(N)), 0))
    A = opt.matrix(1.0, (1, N))
    b = opt.matrix(1.0)
    print(P)
    print(G)
    print(A)
    # hide optimization
    opt.solvers.options['show_progress'] = True
    # calculate portfolio weights, every weight vector is of size Nx1
    # find optimal weights with qp(P, q, G, h, A, b)
    try:
        optimal_weights = [
            cvxopt_solvers.qp(P, q, G, opt.matrix(np.concatenate((-np.ones((1, 1)) * mu, np.zeros((N, 1))), 0)), A, b)['x'] for mu
            in optimal_mus]
    except Exception:
        print('Optimization failed. No optimal solution. Please change opt_mus_n.')
        exit(1)

    # find optimal sigma
    # \sigma = w^T * Cov * w
    optimal_sigmas = [np.sqrt(np.matrix(w).T * cov.T.dot(np.matrix(w)))[0, 0] for w in optimal_weights]

    return optimal_weights, optimal_mus, optimal_sigmas


def create_random_weights(n_assets):
    '''
    returns randomly choosen portfolio weights that sum to one
    '''
    w = np.random.rand(n_assets)
    return w / w.sum()


def evaluate_random_portfolio(returns):
    '''
    returns the mean and standard deviation of returns for a random portfolio
    '''
    # in case a resampler is used
    returns = pd.DataFrame(returns)

    # calculate from covariance, asset returns and weights
    cov = np.matrix(returns.cov())
    R = np.matrix(returns.mean())
    w = np.matrix(create_random_weights(returns.shape[1]))

    # calculate expected portfolio return and risk
    mu = w * R.T
    sigma = np.sqrt(w * cov * w.T)

    return mu, sigma
def create_random_portfolios(returns, n_portfolios=1500):
    '''
    plots randomly created portfolios
    '''
    # calculate mean and std for every portfolio
    pf_mus, pf_sigmas = np.column_stack([evaluate_random_portfolio(returns) for _ in range(n_portfolios)])

    return pf_mus, pf_sigmas


np.random.seed(1)
n_obs = 252 #252 trading days
n_assets = 4 # number of assets
opt_mus_n = 35  #when error.

artificial_returns = np.random.randn(n_obs, n_assets) + 0.05
pf_mus, pf_sigmas = create_random_portfolios(artificial_returns, n_portfolios=3000)
optimal_weights, optimal_mus, optimal_sigmas = calculate_frontier(artificial_returns, opt_mus_n)
plt.plot(pf_sigmas, pf_mus, 'o', markersize=5, label='Available Market Portfolio')
plt.plot(optimal_sigmas, optimal_mus, 'y-o', color='orange', markersize=8, label='Efficient Frontier')
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier and Available Portfolios')
plt.legend(loc='best')
plt.show()