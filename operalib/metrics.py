"""
:mod:`operalib.metrics` define upplementary metrics not
available in scikit-learn

"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
# License: MIT

from numpy import log, pi, sqrt, exp, cos
from sklearn.metrics.pairwise import check_pairwise_arrays, euclidean_distances


def first_periodic_kernel(X, Y=None, gamma=None, period=None):
    # TODO: Add mathematical form of the kernel in the docstring
    """Compute the first periodic kernel between *X* and *Y*.

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)

    Y : array of shape (n_samples_Y, n_features)

    gamma : float, default None
        If None, default to 1.0 / n_samples_X

    period : float, default None
        If None, default to 2 * pi.

        This parameter should not be default as
        wrong estimation lead to poor learning score.

    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 0.8

    if period is None:
        period = 2. * pi

    a = -log(gamma) / period
    b = 2 * pi / period
    c = sqrt(pi / a) * (exp(- b ** 2 / (4 * a)) + 1)
    K = euclidean_distances(X, Y, squared=True)

    # TODO: Optimize to avoid temporary
    return exp(-a * K) * (1 + cos(b * sqrt(K))) / c
