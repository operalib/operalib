"""OVK learning, unit tests.

The :mod:`sklearn.tests.awful` tests awful dataset.
"""
from numpy import isnan

import operalib as ovk


def test_awful():
    """Test awful function."""
    _, y = ovk.toy_data_curl_free_field(n=500)
    y = ovk.datasets.awful(y)
    assert isnan(y).any()
