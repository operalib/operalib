"""OVK learning, unit tests.

The :mod:`sklearn.tests.test_KenrelRidge` tests OVK ridge regression estimator.
"""

from sklearn.utils.estimator_checks import check_estimator
from scipy.optimize import check_grad
from numpy.random import RandomState, rand, randn
from numpy import sort, pi, sin, cos, array, dot, eye, arange, newaxis, cov
from numpy.linalg import norm, cholesky

import operalib as ovk

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

X_test = arange(-100.0, 100.0, .5)[:, newaxis]
y_t = dot(array([pi * sin(X_test).ravel(),
                 pi * cos(X_test).ravel()]).T, U)


def test_valid_estimator():
    """Test whether ovk.Ridge is a valid sklearn estimator."""
    check_estimator(ovk.Ridge)


def test_ridge_grad_id():
    """Test ovk.KernelRidgeRisk gradient with finite differences."""
    K = ovk.DecomposableKernel(A=eye(2))
    risk = ovk.KernelRidgeRisk(0.01)
    assert check_grad(lambda *args: risk.functional_grad_val(*args)[0],
                      lambda *args: risk.functional_grad_val(*args)[1],
                      rand(X.shape[0] * y.shape[1]),
                      y.ravel(), K(X, X)) < 1e-3


def test_ridge_grad_cov():
    """Test ovk.KernelRidgeRisk gradient with finite differences."""
    K = ovk.DecomposableKernel(A=eye(2))
    risk = ovk.KernelRidgeRisk(0.01)
    assert check_grad(lambda *args: risk.functional_grad_val(*args)[0],
                      lambda *args: risk.functional_grad_val(*args)[1],
                      rand(X.shape[0] * y.shape[1]),
                      y.ravel(), K(X, X)) < 1e-3


def test_learn_periodic_id():
    """Test ovk periodic estimator fit, predict. A=Id."""
    regr_1 = ovk.Ridge('DPeriodic', lbda=0.01, period=2 * pi, theta=.99)
    regr_1.fit(X, y)
    assert regr_1.score(X_test, y_t) > 0.9


def test_learn_periodic_cov():
    """Test ovk periodic estimator fit, predict. A=cov(y.T)."""
    A = cov(y.T)
    regr_1 = ovk.Ridge('DPeriodic', lbda=0.01, period=2 * pi, theta=.99, A=A)
    regr_1.fit(X, y)
    assert regr_1.score(X_test, y_t) > 0.9


def test_learn_gauss_cov():
    """Test ovk gaussian estimator fit, predict. A=cov(y.T)."""
    A = cov(y.T)
    regr_1 = ovk.Ridge('DGauss', lbda=0.01, gamma=0.1, A=A)
    regr_1.fit(X, y)
    assert regr_1.score(X_test, y_t) > 0.8
