"""OVK learning, unit tests.

The :mod:`sklearn.tests.test_QuantileRegression` tests OVK quantile regression
estimator.
"""

import operalib as ovk

from sklearn.utils.estimator_checks import check_estimator
import numpy as np


def test_valid_estimator():
    """Test whether ovk.Quantile is a valid sklearn estimator."""
    check_estimator(ovk.Quantile)


def test_learn_quantile():
    """Test OVK quantile estimator fit, predict."""
    probs = np.linspace(0.1, 0.9, 5)  # Quantile levels of interest
    x_train, y_train, z_train = ovk.toy_data_quantile(50)
    x_test, y_test, z_test = ovk.toy_data_quantile(1000, probs=probs)

    # Joint quantile regression
    lbda = 1e-2
    gamma = 1e1
    joint = ovk.Quantile(probs=probs, kernel='DGauss', lbda=lbda,
                         gamma=gamma, gamma_quantile=1e-2)
    # Independent quantile regression
    ind = ovk.Quantile(probs=probs, kernel='DGauss', lbda=lbda,
                       gamma=gamma, gamma_quantile=np.inf)
    # Independent quantile regression (with non-crossing constraints)
    nc = ovk.Quantile(probs=probs, kernel='DGauss', lbda=lbda,
                      gamma=gamma, gamma_quantile=np.inf, nc_const=True)

    # Fit on training data
    for reg in [joint, ind, nc]:
        reg.fit(x_train, y_train)
        assert reg.score(x_test, y_test) > 0.7
