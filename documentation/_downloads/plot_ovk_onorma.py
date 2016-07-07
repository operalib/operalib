"""
======================================================
Online Learning with Operator-Valued kernels
======================================================

An example to illustrate online learning with operator-valued
kernels.
"""

import operalib as ovk
import numpy as np
import matplotlib.pyplot as plt

import time

np.random.seed(0)

n = 1000
d = 20
p = 4
X = np.random.rand(n, d)


def phi(X):
    """Generate data according to Evgeniou, C. A. Micchelli, and M. Pontil.

    'Learning multiple tasks with kernel methods.' 2005.
    """
    return np.hstack((X[:, 0:1] ** 2,
                      X[:, 3:4] ** 2,
                      X[:, 0:1] * X[:, 1:2],
                      X[:, 2:3] * X[:, 4:5],
                      X[:, 1:2],
                      X[:, 3:4],
                      np.ones((n, 1))))


print('Generating Data')
y = np.dot(phi(X), np.random.multivariate_normal(np.zeros(7),
                                                 np.diag([0.5, 0.25, 0.1, 0.05,
                                                          0.15, 0.1, 0.15]),
                                                 p).T)

# Link components to a common mean.
y = .5 * y + 0.5 * np.mean(y, axis=1).reshape(-1, 1)

est = ovk.ONORMA('DGauss', A=.8 * np.eye(p) + .2 * np.ones((p, p)), gamma=.25,
                 learning_rate=ovk.InvScaling(1., 0.5), lbda=0.00001)

print('Fitting Joint...')
start = time.time()
err = np.empty(n)
err[0] = np.linalg.norm(y[0, :]) ** 2
est.partial_fit(X[0, :].reshape(1, -1), y[0, :])
for t in range(1, n):
    err[t] = np.linalg.norm(est.predict(X[t, :].reshape(1, -1)) - y[t, :]) ** 2
    est.partial_fit(X[t, :].reshape(1, -1), y[t, :])
print('Joint training time:', time.time() - start)
print('Joint MSE:', err[-1])

err_c = np.cumsum(err) / (np.arange(n) + 1)
plt.semilogy(np.linspace(0, 100, err_c.size), err_c, label='Joint')

est = ovk.ONORMA('DGauss', A=1. * np.eye(p) + .0 * np.ones((p, p)), gamma=.25,
                 learning_rate=ovk.InvScaling(1., 0.5), lbda=0.00001)

print('Fitting Independant...')
start = time.time()
err = np.empty(n)
err[0] = np.linalg.norm(y[0, :]) ** 2
est.partial_fit(X[0, :].reshape(1, -1), y[0, :])
for t in range(n):
    err[t] = np.linalg.norm(est.predict(X[t, :].reshape(1, -1)) - y[t, :]) ** 2
    est.partial_fit(X[t, :].reshape(1, -1), y[t, :])
print('Independant training time:', time.time() - start)
print('Independant MSE:', err[-1])

err_c = np.cumsum(err) / (np.arange(n) + 1)
plt.semilogy(np.linspace(0, 100, err_c.size), err_c, label='Independant')

plt.ylim(2.5e-2, 1.5e-1)
plt.title('Online learning with ONORMA')
plt.xlabel('Size of the Training set (%)')
plt.ylabel('MSE')
plt.legend()
plt.show()
