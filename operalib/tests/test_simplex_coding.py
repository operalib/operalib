"""OVK learning, unit tests.

The :mod:`sklearn.tests.test_simplex_coding` tests the simplex coding
estimator.
"""
from numpy import zeros, array

import operalib as ovk

from sklearn.utils.testing import assert_array_almost_equal, assert_array_equal


def test_simplexcoding():
    """Test SimplexCoding."""
    # one-class case
    inp = ["pos", "pos", "pos", "pos"]
    scode = ovk.preprocessing.SimplexCoding()
    expected = zeros((4, 1))
    got = scode.fit_transform(inp)
    assert_array_equal(scode.binarizer_.classes_, ["pos"])
    assert_array_equal(expected, got)
    assert_array_equal(scode.inverse_transform(got), inp)

    # two-class case
    inp = ["neg", "pos", "pos", "neg"]
    expected = array([[0], [1], [1], [0]])
    got = scode.fit_transform(inp)
    assert_array_equal(scode.binarizer_.classes_, ["neg", "pos"])
    assert_array_equal(expected, got)
    assert_array_equal(scode.inverse_transform(got), inp)

    # multi-class case
    inp = ["spam", "ham", "eggs", "ham", "0"]
    expected = array([[-1./3., -0.471, -0.816],
                      [-1./3., -0.471, 0.816],
                      [-1./3., 0.943, 0.],
                      [-1./3., -0.471, 0.816],
                      [1., 0., 0.]])
    got = scode.fit_transform(inp)
    assert_array_equal(scode.binarizer_.classes_, ['0', 'eggs', 'ham', 'spam'])
    assert_array_almost_equal(expected, got, decimal=1e-2)
    assert_array_equal(scode.inverse_transform(got), inp)
