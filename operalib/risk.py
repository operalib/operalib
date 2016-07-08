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

    def __call__(self, coefs, ground_truth, Gram, weight, zeronan):
        """Compute the Empirical OVK ridge risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the coefs

        weight: {LinearOperator}

        zeronan: {LinearOperator}

        Returns
        -------
        float : Empirical OVK ridge risk
        """
        np = ground_truth.size
        pred = Gram * coefs
        wpred = weight * pred  # sup x identity | unsup x lbda_m x L
        res = zeronan * (wpred - ground_truth)
        wip = wpred - zeronan * wpred  # only unsup part of wpred
        reg = inner(coefs, pred)  # reg in rkhs
        lap = inner(wip, pred)  # Laplacian part x lambda_m

        obj = norm(zeronan * res) ** 2 / (2 * np)  # Loss
        obj += self.lbda * reg / (2 * np)  # Regulariation
        obj += lap / (2 * np)  # Laplacian regularization
        return obj

    def functional_grad(self, coefs, ground_truth, Gram, weight, zeronan):
        """Compute the gradient of the Empirical OVK ridge risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the coefs

        L : array, shape = [n_samples_miss, n_samples_miss]
            Graph Laplacian of data with missing targets (semi-supervised
            learning).

        Returns
        -------
        {vector-like} : gradient of the Empirical OVK ridge risk
        """
        np = ground_truth.size
        pred = Gram * coefs
        res = weight * pred - zeronan * ground_truth
        return Gram * res / np + self.lbda * pred / np

    def functional_grad_val(self, coefs, ground_truth, Gram, weight, zeronan):
        """Compute the gradient and value of the Empirical OVK ridge risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        Gram : {LinearOperator}
            Gram matrix acting on the coefs

        L : array, shape = [n_samples_miss, n_samples_miss]
            Graph Laplacian of data with missing targets (semi-supervised
            learning).

        Returns
        -------
        Tuple{float, vector-like} : Empirical OVK ridge risk and its gradient
        returned as a tuple.
        """
        np = ground_truth.size
        pred = Gram * coefs
        wpred = weight * pred  # sup x identity | unsup x lbda_m x L
        res = wpred - zeronan * ground_truth
        wip = wpred - zeronan * wpred  # only unsup part of wpred
        reg = inner(coefs, pred)  # reg in rkhs
        lap = inner(wip, pred)  # Laplacian part x lambda_m

        obj = norm(zeronan * res) ** 2 / (2 * np)  # Loss
        obj += self.lbda * reg / (2 * np)  # Regulariation
        obj += lap / (2 * np)  # Laplacian regularization
        return obj, Gram * res / np + self.lbda * pred / np
