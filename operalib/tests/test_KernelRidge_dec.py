"""OVK learning, unit tests.

The :mod:`sklearn.tests.test_KernelRidge` tests OVK ridge regression estimator.
"""

from sklearn.utils.estimator_checks import check_estimator
from numpy.random import rand, randn, seed
from numpy import sort, pi, sin, cos, array, dot, arange, newaxis, cov
from numpy.linalg import norm, cholesky
from distutils.version import LooseVersion
from warnings import warn

import operalib as ovk

seed(0)
X = sort(200 * rand(100, 1) - 100, axis=0)
y = array([pi * sin(X).ravel(), pi * cos(X).ravel()]).T
Tr = 2 * rand(2, 2) - 1
Tr = dot(Tr, Tr.T)
Tr = Tr / norm(Tr, 2)
U = cholesky(Tr)
y = dot(y, U)

Sigma = 2 * rand(2, 2) - 1
Sigma = dot(Sigma, Sigma.T)
Sigma = .1 * Sigma / norm(Sigma, 2)
Cov = cholesky(Sigma)
y += dot(randn(y.shape[0], y.shape[1]), Cov.T)

X_test = arange(-100.0, 100.0, .5)[:, newaxis]
y_t = dot(array([pi * sin(X_test).ravel(),
                 pi * cos(X_test).ravel()]).T, U)


def test_valid_estimator():
    """Test whether ovk.Ridge is a valid sklearn estimator."""
    from sklearn import __version__
    # Adding patch revision number cause crash
    if LooseVersion(__version__) >= LooseVersion('0.18'):
        check_estimator(ovk.Ridge)
    else:
        warn('sklearn\'s check_estimator seems to be broken in __version__ <='
             ' 0.17.x... skipping')


def test_learn_periodic_autocorr_id():
    """Test ovk periodic estimator fit, predict. A=Id, autocorrelation."""
    regr_1 = ovk.Ridge('DPeriodic', lbda=0.01, theta=.8,
                       period='autocorr', autocorr_params={'thres': 0.01,
                                                           'min_dist': 2})
    regr_1.fit(X, y)
    assert regr_1.score(X_test, y_t) > 0.5


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
    regr_1 = ovk.Ridge('DGauss', lbda=.01, gamma=5., A=A)
    regr_1.fit(X, y)
    assert regr_1.score(X_test, y_t) > 0.3
