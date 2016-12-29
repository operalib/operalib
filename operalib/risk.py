"""
:mod:`operalib.risk` implements risk model and their gradients.
"""
# Authors: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
#         Maxime Sangnier <maxime.sangnier@gmail.com>
# License: MIT

from numpy import inner, dot
from numpy.linalg import norm
from numpy.ma import masked_invalid

from .kernels import DecomposableKernel


class ORFFRidgeRisk(object):
    """Define ORFF ridge risk and its gradient."""

    def __init__(self, lbda):
        """Initialize Empirical ORFF ridge risk.

        Parameters
        ----------
        lbda : {float}
            Small positive values of lbda improve the conditioning of the
            problem and reduce the variance of the estimates.  Lbda corresponds
            to ``(2*C)^-1`` in other linear models such as LogisticRegression
            or LinearSVC.
        """
        self.lbda = lbda

    def __call__(self, coefs, ground_truth, phix, ker):
        """Compute the Empirical ORFF ridge risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        phix : {LinearOperator}
            X ORFF mapping operator acting on the coefs

        Returns
        -------
        float : Empirical ORFF ridge risk
        """
        pred = phix * coefs
        res = pred - ground_truth
        np = ground_truth.size
        if isinstance(ker, DecomposableKernel):
            J = dot(ker.B_, ker.B_.T)
            reg = inner(coefs, dot(coefs.reshape((-1, ker.r)), J).ravel())
        else:
            raise('Unsupported kernel')
        return norm(res) ** 2 / (2 * np) + self.lbda * reg / (2 * np)

    def functional_grad(self, coefs, ground_truth, phix, ker):
        """Compute the gradient of the Empirical ORFF ridge risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        phix : {LinearOperator}
            X ORFF mapping operator acting on the coefs

        Returns
        -------
        {vector-like} : gradient of the Empirical ORFF ridge risk
        """
        pred = phix * coefs
        res = pred - ground_truth
        np = ground_truth.size
        if isinstance(ker, DecomposableKernel):
            J = dot(ker.B_, ker.B_.T)
            reg_grad = dot(coefs.reshape((-1, ker.r)), J).ravel()
        else:
            raise('Unsupported kernel')
        return phix.rmatvec(res) / np + self.lbda * reg_grad / np

    def functional_grad_val(self, coefs, ground_truth, phix, ker):
        """Compute the gradient and value of the Empirical ORFF ridge risk.

        Parameters
        ----------
        coefs : {vector-like}, shape = [n_samples1 * n_targets]
            Coefficient to optimise

        ground_truth : {vector-like}
            Targets samples

        phix : {LinearOperator}
            X ORFF mapping operator acting on the coefs

        Returns
        -------
        Tuple{float, vector-like} : Empirical ORFF ridge risk and its gradient
        returned as a tuple.
        """
        pred = phix * coefs
        res = pred - ground_truth
        np = ground_truth.size
        if isinstance(ker, DecomposableKernel):
            J = dot(ker.B_, ker.B_.T)
            reg_grad = dot(coefs.reshape((-1, ker.r)), J).ravel()
            reg = inner(coefs, reg_grad)
        else:
            raise('Unsupported kernel')
        return (norm(res) ** 2 / (2 * np) + self.lbda * reg / (2 * np),
                phix.rmatvec(res) / np + self.lbda * coefs / np)


class OVKRidgeRisk(object):
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

    def __call__(self, coefs, ground_truth, Gram,
                 weight=None, zeronan=None):
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
        reg = inner(coefs, pred)  # reg in rkhs
        vgt = masked_invalid(ground_truth)
        vgt[vgt.mask] = pred[vgt.mask]
        if weight is None or zeronan is None:
            obj = norm(pred - vgt) ** 2 / (2 * np)
        else:
            wpred = weight * pred  # sup x identity | unsup x lbda_m x L
            res = zeronan * (wpred - vgt)
            wip = wpred - zeronan * wpred  # only unsup part of wpred
            lap = inner(wip, pred)  # Laplacian part x lambda_m

            obj = norm(zeronan * res) ** 2 / (2 * np)  # Loss
            obj += lap / (2 * np)  # Laplacian regularization
        obj += self.lbda * reg / (2 * np)  # Regulariation
        return obj

    def functional_grad(self, coefs, ground_truth, Gram,
                        weight=None, zeronan=None):
        """Compute the gradient of the Empirical OVK ridge risk.

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
        {vector-like} : gradient of the Empirical OVK ridge risk
        """
        np = ground_truth.size
        pred = Gram * coefs
        vgt = masked_invalid(ground_truth)
        vgt[vgt.mask] = pred[vgt.mask]
        if weight is None or zeronan is None:
            res = pred - vgt
        else:
            res = weight * pred - zeronan * vgt
        return Gram * res / np + self.lbda * pred / np

    def functional_grad_val(self, coefs, ground_truth, Gram,
                            weight=None, zeronan=None):
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
        vgt = masked_invalid(ground_truth)
        vgt[vgt.mask] = pred[vgt.mask]
        reg = inner(coefs, pred)  # reg in rkhs
        if weight is None or zeronan is None:
            res = pred - vgt
            obj = norm(res) ** 2 / (2 * np)
        else:
            wpred = weight * pred  # sup x identity | unsup x lbda_m x L
            res = wpred - zeronan * vgt
            wip = wpred - zeronan * wpred  # only unsup part of wpred
            lap = inner(wip, pred)  # Laplacian part x lambda_m

            obj = norm(zeronan * res) ** 2 / (2 * np)  # Loss
            obj += lap / (2 * np)  # Laplacian regularization
        obj += self.lbda * reg / (2 * np)  # Regulariation
        return obj, Gram * res / np + self.lbda * pred / np
