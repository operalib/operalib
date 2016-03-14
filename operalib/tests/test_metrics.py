"""OVK learning, unit tests.

The :mod:`sklearn.tests.test_metrics` tests scalar kernel metrics.
"""

from operalib.metrics import first_periodic_kernel
from numpy.random import rand, seed
from numpy import sort, pi
from numpy.testing import assert_allclose

seed(0)
X = sort(2 * rand(1000, 1) - 1, axis=0)


def test_first_periodic_kernel_default():
    """Test whether default params are 1. / X.shape[0], 2 * pi."""
    assert_allclose(first_periodic_kernel(X, X),
                    first_periodic_kernel(X, X, .8, 2. * pi))
