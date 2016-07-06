"""OVK learning, unit tests.

The :mod:`sklearn.tests.test_learningrate` tests the different learning rates.
"""

import operalib as ovk


def test_constant():
    """Test whether constant learning rate."""
    eta = ovk.Constant(1)
    assert eta(10) == 1


def test_invscaling():
    """Test whether inverse scaling learning rate."""
    eta = ovk.InvScaling(1., 2.)
    assert eta(10) == 1. / 10. ** 2.
