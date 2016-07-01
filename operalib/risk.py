"""
:mod:`operalib.risk` implements risk model and their gradients.
"""
# Authors: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
#	   Maxime Sangnier <maxime.sangnier@gmail.com>
# License: MIT

from numpy.linalg import norm
from numpy import inner, reshape, sum, fmax


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

    def __call__(self, dual_coefs, ground_truth, Gram):
        """Compute the Empirical OVK ridge risk.

        Parameters
        ----------
        dual_coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the dual_coefs

        Returns
        -------
        float : Empirical OVK ridge risk
        """
        pred = Gram * dual_coefs
        res = pred - ground_truth
        np = ground_truth.size
        reg = inner(dual_coefs, pred)
        return norm(res) ** 2 / (2 * np) + self.lbda * reg / (2 * np)

    def functional_grad(self, dual_coefs, ground_truth, Gram):
        """Compute the gradient of the Empirical OVK ridge risk.

        Parameters
        ----------
        dual_coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the dual_coefs

        Returns
        -------
        {vector-like} : gradient of the Empirical OVK ridge risk
        """
        pred = Gram * dual_coefs
        res = pred - ground_truth
        np = ground_truth.size
        return Gram * res / np + self.lbda * pred / np

    def functional_grad_val(self, dual_coefs, ground_truth, Gram):
        """Compute the gradient and value of the Empirical OVK ridge risk.

        Parameters
        ----------
        dual_coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the dual_coefs

        Returns
        -------
        Tuple{float, vector-like} : Empirical OVK ridge risk and its gradient
        returned as a tuple.
        """
        pred = Gram * dual_coefs
        res = pred - ground_truth
        np = ground_truth.size
        reg = inner(dual_coefs, pred)
        return (norm(res) ** 2 / (2 * np) + self.lbda * reg / (2 * np), Gram *
                res / np + self.lbda * pred / np)


#class KernelQuantileRisk(object):
#    """Define Kernel quantile regression risk and its gradient."""
#
#    def __init__(self, lbda, probs):
#        """Initialize Empirical kernel quantile regression risk.
#
#        Parameters
#        ----------
#        lbda : {float}
#            Small positive values of lbda improve the conditioning of the
#            problem and reduce the variance of the estimates.  Lbda corresponds
#            to ``(2*C)^-1`` in other linear models such as LogisticRegression
#            or LinearSVC.
#	probs : {array}
#            Array of quantile levels of interest.
#        """
#        self.lbda = lbda
#	self.probs = probs
#
#    def __call__(self, dual_coefs, intercept, ground_truth, Gram):
#        """Compute the Empirical OVK risk.
#
#        Parameters
#        ----------
#        dual_coefs : {vector-like}, shape = [n_samples1 * n_targets]
#            Coefficient to optimise
#
#	intercept : {float}
#	    Intercept
#
#        ground_truth : {vector-like}
#            Targets samples
#
#        Gram : {LinearOperator}
#            Gram matrix acting on the dual_coefs
#
#        Returns
#        -------
#        float : Empirical OVK ridge risk
#        """
#        pred = Gram * dual_coefs
#        reg = inner(dual_coefs, pred)
#	pred = reshape(pred, (dual_coefs.shape[0]/intercept.shape[0], intercept.shape[0]))
#	pred += intercept
#	res = reshape(pred - ground_truth.T, -1)
#        np = ground_truth.size
#        return pinball(res, self.probs) / np + self.lbda * reg / (2 * np)
#
#
#TO DO
#
#    def functional_grad(self, dual_coefs, ground_truth, Gram):
#        """Compute the gradient of the Empirical OVK ridge risk.
#
#        Parameters
#        ----------
#        dual_coefs : {vector-like}, shape = [n_samples1 * n_targets]
#            Coefficient to optimise
#
#        ground_truth : {vector-like}
#            Targets samples
#
#        Gram : {LinearOperator}
#            Gram matrix acting on the dual_coefs
#
#        Returns
#        -------
#        {vector-like} : gradient of the Empirical OVK ridge risk
#        """
#        pred = Gram * dual_coefs
#        res = pred - ground_truth
#        np = ground_truth.size
#        return Gram * res / np + self.lbda * pred / np
#
#    def functional_grad_val(self, dual_coefs, ground_truth, Gram):
#        """Compute the gradient and value of the Empirical OVK ridge risk.
#
#        Parameters
#        ----------
#        dual_coefs : {vector-like}, shape = [n_samples1 * n_targets]
#            Coefficient to optimise
#
#        ground_truth : {vector-like}
#            Targets samples
#
#        Gram : {LinearOperator}
#            Gram matrix acting on the dual_coefs
#
#        Returns
#        -------
#        Tuple{float, vector-like} : Empirical OVK ridge risk and its gradient
#        returned as a tuple.
#        """
#        pred = Gram * dual_coefs
#        res = pred - ground_truth
#        np = ground_truth.size
#        reg = inner(dual_coefs, pred)
#        return (norm(res) ** 2 / (2 * np) + self.lbda * reg / (2 * np), Gram *
#                res / np + self.lbda * pred / np)
#
#def pinball(res, probs):
#    n = res.shape[0] / probs.shape[0]
#    r = reshape(res, (n, probs.shape[0])).T
#    l = sum([sum(t*fmax(t*q, (t-1)*q)) for (t, q) in zip(probs, r)])
#    return l
#
