"""OVK learning unit tests.

The :mod:`sklearn.tests` unit tests for ovk
kernels.
"""

from sklearn.utils.estimator_checks import check_estimator
from sklearn import ovk
from scipy.optimize import check_grad
from numpy.random import RandomState, rand, randn
from numpy import sort, pi, sin, cos, array, dot, eye
from numpy.linalg import norm, cholesky

seed = 0
rng = RandomState(seed)
X = sort(200 * rng.rand(1000, 1) - 100, axis=0)
y = array([pi * sin(X).ravel(), pi * cos(X).ravel()]).T
Tr = 2 * rand(2, 2) - 1
Tr = dot(Tr, Tr.T)
Tr = Tr / norm(Tr, 2)
U = cholesky(Tr)
y = dot(y, U)

Sigma = 2 * rand(2, 2) - 1
Sigma = dot(Sigma, Sigma.T)
Sigma = 1. * Sigma / norm(Sigma, 2)
Cov = cholesky(Sigma)
y += dot(randn(y.shape[0], y.shape[1]), Cov.T)


def _test_valid_estimator():
    check_estimator(ovk.Ridge)


def _test_ridge_grad_id():
    K = ovk.DecomposableKernel(A=eye(2))
    risk = ovk.KernelRidgeRisk(0.01)
    check_grad(lambda *args: risk.functional_grad_val(*args)[0],
               lambda *args: risk.functional_grad_val(*args)[1],
               rand(X.shape[0] * y.shape[1]),
               y.ravel(), K(X, X))


def _test_ridge_grad_cov():
    K = ovk.DecomposableKernel(A=eye(2))
    risk = ovk.KernelRidgeRisk(0.01)
    check_grad(lambda *args: risk.functional_grad_val(*args)[0],
               lambda *args: risk.functional_grad_val(*args)[1],
               rand(X.shape[0] * y.shape[1]),
               y.ravel(), K(X, X))
