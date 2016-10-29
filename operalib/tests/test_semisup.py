"""OVK learning, unit tests.

The :mod:`sklearn.tests.test_semisuo` tests semisup module.
"""

import operalib as ovk
import numpy as np


def test_semisup_linop():
    """Test ovk.semisup.SemisupLinop."""
    np.random.seed()

    n = 100
    p = 5
    lbda2 = .1

    # supervised indices
    B = np.random.randint(2, size=(n)).astype(np.bool)

    # Graph Laplacian
    n_unsup = np.sum(~B)
    L = np.random.randn(n_unsup, n_unsup)
    L = np.dot(L, L.T)

    U, J = ovk.ridge._SemisupLinop(lbda2, B, L, p).gen()

    y = np.random.randn(n, p)
    # lbda2 * np.dot(L, y[~B, :]).ravel()
    # print(np.concatenate((y[B, :].ravel(),
    #                       lbda2 * np.dot(L, y[~B, :]).ravel())))

    res = np.empty((n, p))
    res[B, :] = y[B, :]
    res[~B] = 0
    assert np.allclose(J * y.ravel(), res.ravel())

    res = np.empty((n, p))
    res[B, :] = y[B, :]
    res[~B] = lbda2 * np.dot(L, y[~B, :])
    assert np.allclose(U * y.ravel(), res.ravel())
