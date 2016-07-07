"""OVK learning, unit tests.

The :mod:`sklearn.tests.test_KernelRidge` tests OVK ridge regression estimator.
"""
import operalib as ovk

from sklearn import __version__
from distutils.version import LooseVersion
if LooseVersion(__version__) < LooseVersion('0.18'):
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split


def test_learn_cf():
    """Test ovk curl-free estimator fit."""
    X, y = ovk.toy_data_curl_free_field(n=500)

    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=100)

    regr_1 = ovk.Ridge(kernel=ovk.RBFCurlFreeKernel(gamma=10.), lbda=0)
    regr_1.fit(Xtr, ytr)
    assert regr_1.score(Xte, yte) >= 0.8


def test_learn_df():
    """Test ovk curl-free estimator fit."""
    X, y = ovk.toy_data_div_free_field(n=500)

    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=100)

    regr_1 = ovk.Ridge(kernel=ovk.RBFDivFreeKernel(gamma=10.), lbda=0)
    regr_1.fit(Xtr, ytr)
    assert regr_1.score(Xte, yte) >= 0.8
