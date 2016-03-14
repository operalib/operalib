"""OVK learning, unit tests.

The :mod:`sklearn.tests.test_risk` tests risk module.
"""

from scipy.optimize import check_grad

import operalib as ovk
from numpy import eye, dot, sort, array, arange, pi, cos, sin, newaxis
from numpy.testing import assert_allclose
from numpy.linalg import cholesky, norm
from numpy.random import randn, rand, seed

seed(0)
X = sort(200 * rand(1000, 1) - 100, axis=0)
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


def test_ridge_grad_id():
    """Test ovk.KernelRidgeRisk gradient with finite differences."""
    K = ovk.DecomposableKernel(A=eye(2))
    risk = ovk.KernelRidgeRisk(0.01)
    assert check_grad(lambda *args: risk.functional_grad_val(*args)[0],
                      lambda *args: risk.functional_grad_val(*args)[1],
                      randn(X.shape[0] * y.shape[1]),
                      y.ravel(), K(X, X)) < 1e-3


def test_ridge_grad_cov():
    """Test ovk.KernelRidgeRisk gradient with finite differences."""
    K = ovk.DecomposableKernel(A=eye(2))
    risk = ovk.KernelRidgeRisk(0.01)
    assert check_grad(lambda *args: risk.functional_grad_val(*args)[0],
                      lambda *args: risk.functional_grad_val(*args)[1],
                      randn(X.shape[0] * y.shape[1]),
                      y.ravel(), K(X, X)) < 1e-3


def test_grad_val():
    """Test whether ovk.KernelRidgeRisk gradient and val are consistents."""
    K = ovk.DecomposableKernel(A=eye(2))
    risk = ovk.KernelRidgeRisk(0.01)
    C = randn(X.shape[0] * y.shape[1])
    v = risk(C, y.ravel(), K(X, X))
    g = risk.functional_grad(C, y.ravel(), K(X, X))
    vg = risk.functional_grad_val(C, y.ravel(), K(X, X))
    assert_allclose(v, vg[0])
    assert_allclose(g, vg[1])
