"""
:mod:`operalib.learningrate` implements the learning rate for (Stochastic)
gradient descent algorithms
"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
# License: MIT


def Constant(eta=1.):
    r"""Constant learnin rate.

    .. math::
        t \mapsto eta

    Parameters
    ----------
    eta : float default 1.

    Returns
    -------
    eta : Callable
    """
    return lambda t: float(eta)


def InvScaling(eta=1., power=1.):
    r"""Constant learnin rate.

    .. math::
        t \mapsto eta0 * t^{-power}

    Parameters
    ----------
    eta0 : float, default 1.

    power : float, default 1.

    Returns
    -------
    eta : Callable
    """
    return lambda t: float(eta) / t ** power
