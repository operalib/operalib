"""
:mod:`operalib.risk` implements risk model and their gradients.
"""
# Authors: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
#         Maxime Sangnier <maxime.sangnier@gmail.com>
# License: MIT

from numpy.linalg import norm
from numpy import inner


class KernelRidgeRisk(object):
    """Define Kernel ridge risk and its gradient."""

    def __init__(self, lbda):
        """Initialize Empirical kernel ridge risk.

        Parameters
        ----------
        lbda : {float}
            Small positive values of lbda improve the conditioning of the
            problem and reduce the variance of the estimates.  Lbda corresponds
            to ``(2*C)^-1`` in other linear models such as LogisticRegression
            or LinearSVC.
        """
        self.lbda = lbda

    def __call__(self, coefs, ground_truth, Gram):
        """Compute the Empirical OVK ridge risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the coefs

        Returns
        -------
        float : Empirical OVK ridge risk
        """
        pred = Gram * coefs
        res = pred - ground_truth
        np = ground_truth.size
        reg = inner(coefs, pred)
        return norm(res) ** 2 / (2 * np) + self.lbda * reg / (2 * np)

    def functional_grad(self, coefs, ground_truth, Gram):
        """Compute the gradient of the Empirical OVK ridge risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the coefs

        Returns
        -------
        {vector-like} : gradient of the Empirical OVK ridge risk
        """
        pred = Gram * coefs
        res = pred - ground_truth
        np = ground_truth.size
        return Gram * res / np + self.lbda * pred / np

    def functional_grad_val(self, coefs, ground_truth, Gram):
        """Compute the gradient and value of the Empirical OVK ridge risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the coefs

        Returns
        -------
        Tuple{float, vector-like} : Empirical OVK ridge risk and its gradient
        returned as a tuple.
        """
        pred = Gram * coefs
        res = pred - ground_truth
        np = ground_truth.size
        reg = inner(coefs, pred)
        return (norm(res) ** 2 / (2 * np) + self.lbda * reg / (2 * np), Gram *
                res / np + self.lbda * pred / np)
