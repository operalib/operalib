"""
:mod:`operalib.learningrate` implements the learning rate for (Stochastic)
gradient descent algorithms
"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
# License: MIT


class Constant(object):
    r"""Constant learning rate.

    .. math::
        t \mapsto eta

    Attributes
    ----------
    eta : float default 1.
    """

    def __init__(self, eta=1):
        r"""Initialize constant learning rate.

        Parameters
        ----------
        eta : float default 1.
        """
        self.eta = eta

    def __call__(self, t):
        r"""Return learning rate at time t.

        Returns
        -------
        self.eta : float
        """
        return self.eta


class InvScaling(object):
    r"""Inverse scaling learnin rate.

    .. math::
        t \mapsto eta0 * t^{-power}

    Parameters
    ----------
    eta0 : float, default 1.

    power : float, default 1.

    Returns
    -------
    InvScaling : Callable
    """

    def __init__(self, eta=1, power=1.):
        r"""Initialize inverse scaling learnin rate.

        Parameters
        ----------
        eta0 : float, default 1.

        power : float, default 1.
        """
        self.eta = eta
        self.power = power

    def __call__(self, t):
        r"""Return learning rate at time t.

        Returns
        -------
        self.eta / t ** self.power : float
        """
        return float(self.eta) / t ** self.power
