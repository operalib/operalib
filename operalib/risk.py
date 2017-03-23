"""
:mod:`operalib.risk` implements risk model and their gradients.
"""
# Authors: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
#         Maxime Sangnier <maxime.sangnier@gmail.com>
# License: MIT

from numpy import inner, maximum, sum, multiply, subtract, where
from numpy.linalg import norm
from numpy.ma import masked_invalid


class ORFFHingeLoss(object):
    """Define Hinge loss for ORFF models and its gradient."""

    def __init__(self):
        """Initialize Hhnge ORFF loss.

        Parameters
        ==========
        """
        pass

    def __call__(self, coefs, ground_truth, phix, ker):
        """Compute the hinge loss value for ORFF models.


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
        float : Empirical ORFF hinge loss

        """
        pred = phix * coefs
        return maximum(pred.reshape((-1, ker.p)).T -
                       sum(multiply(ground_truth, pred).reshape((-1, ker.p)),
                           axis=1).flat + 1 - ground_truth.reshape((-1,
                                                                    ker.p)).T,
                       0).sum(axis=0).mean()

    def functional_grad(self, coefs, ground_truth, phix, ker):
        pred = phix * coefs
        mask = (pred.reshape((-1, ker.p)).T -
                sum(multiply(ground_truth, pred).reshape((-1, ker.p)),
                    axis=1).flat + 1 - ground_truth.reshape((-1,
                                                             ker.p)).T > 0).T
        temp = multiply(mask.sum(axis=1).flat,
                        ground_truth.reshape((-1, ker.p)).T).T.flat
        mask = multiply(mask, 1 - ground_truth.reshape((-1, ker.p))).flat
        temp = phix.rmatvec(subtract(mask, temp))
        return (temp / ground_truth.size * ker.p)

    def functional_grad_val(self, coefs, ground_truth, phix, ker):
        pred = phix * coefs
        mrg = (pred.reshape((-1, ker.p)).T -
               sum(multiply(ground_truth, pred).reshape((-1, ker.p)),
                   axis=1).flat + 1 - ground_truth.reshape((-1, ker.p)).T).T
        val = maximum(mrg, 0).sum(axis=1).mean()
        mrg = mrg > 0
        temp = multiply(mrg.sum(axis=1).flat,
                        ground_truth.reshape((-1, ker.p)).T).T.flat
        mrg = multiply(mrg, 1 - ground_truth.reshape((-1, ker.p))).flat
        temp = phix.rmatvec(subtract(mrg, temp))
        return (val, temp / ground_truth.size * ker.p)


class ORFFLSLoss(object):
    """Define Least squares loss for ORFF models and its gradient."""

    def __init__(self):
        """Initialize Least squares ORFF loss.

        Parameters
        ==========
        """
        pass

    def __call__(self, coefs, ground_truth, phix, ker):
        """Compute the Least squares loss value for ORFF models.


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
        float : Empirical ORFF least square loss

        """
        pred = phix * coefs
        res = pred - ground_truth
        return norm(res) ** 2 / (2 * ground_truth.size)

    def functional_grad(self, coefs, ground_truth, phix, ker):
        """Compute the Least squares loss gradient for ORFF models.


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
        float : Empirical ORFF least square loss gradient

        """
        pred = phix * coefs
        res = pred - ground_truth
        return phix.rmatvec(res) / ground_truth.size

    def functional_grad_val(self, coefs, ground_truth, phix, ker):
        """Compute the Least squares loss gradient and value for ORFF models.


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
        tuple : Empirical ORFF least square loss value and gradient

        """
        pred = phix * coefs
        res = pred - ground_truth
        return norm(res) ** 2 / (2 * ground_truth.size), \
            phix.rmatvec(res) / ground_truth.size


class ORFFRidgeRisk(object):
    """Define Ridge risk for ORFF models and its gradient."""

    def __init__(self, lbda, loss='LS'):
        """Initialize Empirical ORFF ridge risk.

        Parameters
        ----------
        lbda : {float}
            Small positive values of lbda improve the conditioning of the
            problem and reduce the variance of the estimates. Lbda corresponds
            to ``(2*C)^-1`` in other linear models such as LogisticRegression
            or LinearSVC.
        """
        self.lbda = lbda
        if loss is 'LS':
            self.loss = ORFFLSLoss()
        elif loss is 'Hinge':
            self.loss = ORFFHingeLoss()
        else:
            raise NotImplementedError('unsupported loss')

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
        return (self.loss(coefs, ground_truth, phix, ker) +
                self.lbda * norm(coefs) ** 2 / (2 * ground_truth.size))

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
        return (self.loss.functional_grad(coefs, ground_truth, phix, ker) +
                self.lbda * coefs / ground_truth.size)

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
        val_loss, grad_loss = self.loss.functional_grad_val(coefs,
                                                            ground_truth,
                                                            phix, ker)
        return (val_loss +
                self.lbda * norm(coefs) ** 2 / (2 * ground_truth.size),
                grad_loss + self.lbda * coefs / ground_truth.size)


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
        vgt[where(vgt.mask)] = pred[where(vgt.mask)]
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
        vgt[where(vgt.mask)] = pred[where(vgt.mask)]
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
        vgt[where(vgt.mask)] = pred[where(vgt.mask)]
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
