"""OVK learning, unit tests.

The :mod:`sklearn.tests.test_KenrelRidge` tests OVK ridge regression estimator.
"""
import operalib as ovk
from sklearn.cross_validation import train_test_split


def test_learn_cf():
    """Test ovk curl-free estimator fit."""
    X, y = ovk.generate_2D_curl_free_field(n=500)

    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=100)

    regr_1 = ovk.Ridge(
        kernel=ovk.RBFCurlFreeKernel, lbda=0, kernel_params={'gamma': 10.})
    regr_1.fit(Xtr, ytr)
    assert regr_1.score(Xte, yte) >= 0.8


def test_learn_df():
    """Test ovk curl-free estimator fit."""
    X, y = ovk.generate_2D_div_free_field(n=5000)

    Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=100)

    regr_1 = ovk.Ridge(
        kernel=ovk.RBFDivFreeKernel, lbda=0, kernel_params={'gamma': 10.})
    regr_1.fit(Xtr, ytr)
    assert regr_1.score(Xte, yte) >= 0.8
