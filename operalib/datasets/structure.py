""" Synthetic datasets for strutured learning."""

from sklearn.utils import check_random_state
from numpy import zeros, diag, dot, hstack, ones


def _phi(X):
    return hstack((X[:, 0:1] ** 2,
                   X[:, 3:4] ** 2,
                   X[:, 0:1] * X[:, 1:2],
                   X[:, 2:3] * X[:, 4:5],
                   X[:, 1:2],
                   X[:, 3:4],
                   ones((X.shape[0], 1))))


def toy_data_multitask(n_samples, input_dim, output_dim, random_state=None):
    """Generate data according to Evgeniou, C. A. Micchelli, and M. Pontil.

    'Learning multiple tasks with kernel methods.' 2005.

    Parameters
    ----------

    """
    rs = check_random_state(random_state)
    X = rs.rand(n_samples, input_dim)
    w = rs.multivariate_normal(zeros(7),
                               diag([0.5, 0.25, 0.1, 0.05, 0.15, 0.1,
                                     0.15]),
                               output_dim)
    y = dot(_phi(X), w.T)

    return X, y
