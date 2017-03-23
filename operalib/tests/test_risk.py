"""OVK learning, unit tests.

The :mod:`sklearn.tests.test_risk` tests risk module.
"""

from scipy.optimize import check_grad
from scipy.sparse import coo_matrix

import operalib as ovk
from numpy import eye, dot, sort, array, arange, pi, cos, sin, newaxis, ones
from numpy.testing import assert_allclose
from numpy.linalg import cholesky, norm
from numpy.random import randn, rand, seed, randint

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
    """Test ovk.OVKRidgeRisk gradient with finite differences."""
    K = ovk.DecomposableKernel(A=eye(2))
    risk = ovk.OVKRidgeRisk(0.01)
    assert check_grad(lambda *args: risk.functional_grad_val(*args)[0],
                      lambda *args: risk.functional_grad_val(*args)[1],
                      randn(X.shape[0] * y.shape[1]),
                      y.ravel(), K(X, X)) < 1e-3


def test_ridge_grad_cov():
    """Test ovk.OVKRidgeRisk gradient with finite differences."""
    K = ovk.DecomposableKernel(A=eye(2))
    risk = ovk.OVKRidgeRisk(0.01)
    assert check_grad(lambda *args: risk.functional_grad_val(*args)[0],
                      lambda *args: risk.functional_grad_val(*args)[1],
                      randn(X.shape[0] * y.shape[1]),
                      y.ravel(), K(X, X)) < 1e-3


def test_grad_val():
    """Test whether ovk.OVKRidgeRisk gradient and val are consistents."""
    K = ovk.DecomposableKernel(A=eye(2))
    risk = ovk.OVKRidgeRisk(0.01)
    C = randn(X.shape[0] * y.shape[1])
    v = risk(C, y.ravel(), K(X, X))
    g = risk.functional_grad(C, y.ravel(), K(X, X))
    vg = risk.functional_grad_val(C, y.ravel(), K(X, X))
    assert_allclose(v, vg[0])
    assert_allclose(g, vg[1])


def test_rff_ridge_grad_id():
    """Test ovk.ORFFidgeRisk gradient with finite differences."""
    K = ovk.DecomposableKernel(A=eye(2))
    risk = ovk.ORFFRidgeRisk(0.01)
    D = 100
    assert check_grad(lambda *args: risk.functional_grad_val(*args)[0],
                      lambda *args: risk.functional_grad_val(*args)[1],
                      randn(D * y.shape[1]),
                      y.ravel(), K.get_orff_map(X, D), K) < 1e-3


def test_rff_ridge_grad_cov():
    """Test ovk.ORFFRidgeRisk gradient with finite differences."""
    K = ovk.DecomposableKernel(A=eye(2))
    risk = ovk.ORFFRidgeRisk(0.01)
    D = 100
    assert check_grad(lambda *args: risk.functional_grad_val(*args)[0],
                      lambda *args: risk.functional_grad_val(*args)[1],
                      randn(D * y.shape[1]),
                      y.ravel(), K.get_orff_map(X, D), K) < 1e-3


def test_rff_grad_val_least_squares():
    """Test whether ovk.ORFFRidgeRisk least square gradient and val are
    consistents."""
    K = ovk.DecomposableKernel(A=eye(2))
    risk = ovk.ORFFRidgeRisk(0.01)
    D = 100
    C = randn(D * y.shape[1])
    v = risk(C, y.ravel(), K.get_orff_map(X, D), K)
    g = risk.functional_grad(C, y.ravel(), K.get_orff_map(X, D), K)
    vg = risk.functional_grad_val(C, y.ravel(), K.get_orff_map(X, D), K)
    assert_allclose(v, vg[0])
    assert_allclose(g, vg[1])


def one_hot(X, k):
    m = len(X)
    return coo_matrix((ones(m), (arange(m), X)), shape=(m, k)).todense()


def test_rff_ridge_hinge_grad():
    """Test ovk.ORFFRidgeRisk gradient with finite differences."""
    K = ovk.DecomposableKernel(A=eye(3))
    risk = ovk.ORFFRidgeRisk(0.01, 'Hinge')
    D = 100
    y = one_hot(randint(0, 3, X.shape[0]), 3)
    # print(risk.functional_grad_val(rand(D * y.shape[1]), y.ravel(), K.get_orff_map(X, D), K))
    vl = check_grad(lambda *args: risk.functional_grad_val(*args)[0],
                    lambda *args: risk.functional_grad_val(*args)[1],
                    rand(D * y.shape[1]),
                    y.ravel(), K.get_orff_map(X, D), K)
    print(vl)
    assert vl < 1e-3


def test_rff_grad_val_hinge():
    """Test whether ovk.ORFFRidgeRisk hinge gradient and val are
    consistents."""
    K = ovk.DecomposableKernel(A=eye(3))
    risk = ovk.ORFFRidgeRisk(0.01, 'Hinge')
    y = one_hot(randint(0, 3, X.shape[0]), 3)
    D = 100
    C = randn(D * y.shape[1])
    v = risk(C, y.ravel(), K.get_orff_map(X, D), K)
    g = risk.functional_grad(C, y.ravel(), K.get_orff_map(X, D), K)
    vg = risk.functional_grad_val(C, y.ravel(), K.get_orff_map(X, D), K)
    assert_allclose(v, vg[0])
    assert_allclose(g, vg[1])

