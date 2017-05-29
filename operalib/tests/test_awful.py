"""OVK learning, unit tests.

The :mod:`sklearn.tests.awful` tests awful dataset.
"""
from numpy import isnan

import operalib as ovk


def test_awful():
    """Test awful function."""
    _, targets = ovk.toy_data_curl_free_field(n_samples=500)
    targets = ovk.datasets.awful(targets)
    assert isnan(targets).any()
