"""
:mod:`operalib.ridge` implements Operator-Valued Kernel ridge
regression.
"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
#         Maxime Sangnier <maxime.sangnier@gmail.com>
# License: MIT

from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator
from numpy import (reshape, eye, zeros, empty, dot, all, isnan, diag, number,
                   issubdtype)

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import rbf_kernel

from .metrics import first_periodic_kernel
from .kernels import DecomposableKernel, RBFCurlFreeKernel
from .risk import OVKRidgeRisk
from .signal import get_period

# When adding a new kernel, update this table and the _get_kernel_map method
PAIRWISE_KERNEL_FUNCTIONS = {
    'DGauss': DecomposableKernel,
    'DPeriodic': DecomposableKernel,
    'CurlF': RBFCurlFreeKernel}


def _graph_Laplacian(similarities):
    return diag(similarities.sum(axis=1)) - similarities


class _SemisupLinop:

    def __init__(self, lbda2, is_sup, L, p):
        self.lbda2 = lbda2
        self.is_sup = is_sup.ravel()
        self.L = L
        self.p = p
        self.ns = is_sup.shape[0] - L.shape[0]
        self.ls = L.shape[0]

    def _dot_U(self, vec):
        mat = vec.reshape((self.ns + self.ls, self.p))
        res = empty((self.ns + self.ls, self.p))
        res[self.is_sup, :] = mat[self.is_sup, :]
        res[~self.is_sup, :] = self.lbda2 * dot(self.L, mat[~self.is_sup, :])

        return res.ravel()

    def _dot_JT(self, vec):
        mat = vec.reshape((self.ns + self.ls, self.p))
        res = empty((self.ns + self.ls, self.p))
        res[self.is_sup, :] = mat[self.is_sup, :]
        res[~self.is_sup, :] = 0

        return res.ravel()

    def gen(self):
        shape_U = ((self.ns + self.ls) * self.p, (self.ns + self.ls) * self.p)
        shape_JT = ((self.ns + self.ls) * self.p, (self.ns + self.ls) * self.p)

        # return U, JT
        return (LinearOperator(shape_U,
                               dtype=self.L.dtype,
                               matvec=lambda b: self._dot_U(b),
                               rmatvec=lambda b: self._dot_U(b)),
                LinearOperator(shape_JT,
                               dtype=self.L.dtype,
                               matvec=lambda b: self._dot_JT(b),
                               rmatvec=lambda b: self._dot_JT(b)))


class OVKRidge(BaseEstimator, RegressorMixin):
    """Operator-Valued kernel ridge regression.

    Operator-Valued kernel ridge regression (OVKRR) combines ridge regression
    (linear least squares with l2-norm regularization) with the (OV)kernel
    trick. It learns a linear function in the space induced by the
    respective kernel and the data. For non-linear kernels, this corresponds to
    a non-linear function in the original space.

    The form of the model learned by OVKRR is identical to support vector
    regression (SVR). However, different loss functions are used: OVKRR uses
    squared error loss while support vector regression uses epsilon-insensitive
    loss, both combined with l2 regularization. In contrast to SVR, fitting a
    OVKRR model can be done in closed-form and is typically faster for
    medium-sized datasets. On the other  hand, the learned model is non-sparse
    and thus slower than SVR, which learns a sparse model for epsilon > 0, at
    prediction-time.

    Optimization problem solved for learning:
    .. math::
        \min_{h \in \mathcal H}~ \frac{\lambda}{2} \|h\|_{\mathcal H}^2 +
        \frac{1}{np} \sum_{i=1}^n \|y_j - h(x_i)\|_{\mathcal Y}^2 +
        \frac{\lambda_m}{2} \sum_{i=1}^n W_{ij}
        \|h(x_i) - h(x_j)\|_{\mathcal Y}^2

    Attributes
    ----------
    dual_coef_ : array, shape = [n_samples x n_targest]
        Weight vector(s) in kernel space

    linop_ : callable
        Callable which associate to the training points X the Gram matrix (the
        Gram matrix being a LinearOperator)

    A_ : array, shape = [n_targets, n_targets]
        Set when Linear operator used by the decomposable kernel is default or
        None.

    L_ : array, shape = [n_samples_miss, n_samples_miss]
        Graph Laplacian of data with missing targets (semi-supervised
        learning).

    period_ : float
        Set when period used by the First periodic kernel is 'autocorr'.

    solver_res_ : any
        Raw results returned by the solver.

    References
    ----------
    * Micchelli, Charles A., and Massimiliano Pontil.
      "On learning vector-valued functions." Neural computation
      17.1 (2005): 177-204.

    * Alvarez, Mauricio A., Lorenzo Rosasco, and Neil D. Lawrence.
      "Kernels for vector-valued functions: A review." arXiv preprint
      arXiv:1106.6251 (2011). APA

    * Brouard Celine, d'Alche-Buc Florence and Szafranski Marie.
      "Input Output Kernel Regression," Hal preprint
      hal-01216708 (2015).


    See also
    --------
    sklearn.OVKRidge
        Linear ridge regression.
    sklearn.KernelOVKRidge
        Kernel ridge regression.
    sklearn.SVR
        Support Vector Regression implemented using libsvm.

    Examples
    --------
    >>> import operalib as ovk
    >>> import numpy as np
    >>> n_samples, n_features, n_targets = 10, 5, 5
    >>> rng = np.random.RandomState(0)
    >>> y = rng.randn(n_samples, n_targets)
    >>> X = rng.randn(n_samples, n_features)
    >>> clf = ovk.OVKRidge('DGauss', lbda=1.0)
    >>> clf.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    OVKRidge(...)
    """

    def __init__(self,
                 ovkernel='DGauss', lbda=1e-5, lbda_m=0.,
                 A=None, gamma=None, gamma_m=None, theta=0.7,
                 period='autocorr', autocorr_params=None,
                 solver='L-BFGS-B', solver_params=None):
        """Initialize OVK ridge regression model.

        Parameters
        ----------

        ovkernel : {string, callable}, default='DGauss'
            Kernel mapping used internally. A callable should accept two
            arguments, and should return a LinearOperator.

        lbda : {float}, default=1e-5
            Small positive values of lbda improve the conditioning of the
            problem and reduce the variance of the estimates.  Lbda corresponds
            to ``(2*C)^-1`` in other linear models such as LogisticRegression
            or LinearSVC.

        lbda_m : {float}, default=0.
            Regularization parameter for quadratic penalty on data with missing
            targets.

        A : {LinearOperator, array-like, sparse matrix}, default=None
            Linear operator used by the decomposable kernel. If default is
            None, wich is set to identity matrix of size y.shape[1] when
            fitting.

        gamma : {float}, default=None.
            Gamma parameter for the Decomposable Gaussian kernel.
            Ignored by other kernels.

        gamma_m : {float}, default=None.
            Gamma parameter for the graph Laplacian inducing penalty on data
            with missing targets.

        theta : {float}, default=.7
            Theta parameter for the Decomposable First Periodic kernel.
            Ignored by other kernels.

        period : {float}, default=default_period
            Period parameter for the First periodic kernel. If optional modules
            have been imported then default_period is 2 * pi. Otherwise it uses
            autocorrelation methods to determine the period.

        solver : {callable}, default=scipy.optimize.fmin_l_bfgs_b
            Solver able to find the minimum of the ridge problem.
            scipy.optimize.fmin_l_bfgs_b(*solver_params)[0] must return the
            optimal solution.

        autocorr_params : {mapping of string to any}
            Additional parameters (keyword arguments) for the period detection
            for periodic kernels. If None, parameter choice is left to the
            period detection method.

        solver_params : {mapping of string to any}, optional
            Additional parameters (keyword arguments) for solver function
            passed as callable object.
        """
        self.ovkernel = ovkernel
        self.lbda = lbda
        self.lbda_m = lbda_m
        self.A = A
        self.gamma = gamma
        self.gamma_m = gamma_m
        self.theta = theta
        self.period = period
        self.autocorr_params = autocorr_params
        self.solver = solver
        self.solver_params = solver_params

    def _validate_params(self):
        # check on self.ovkernel is performed in method __get_kernel
        if self.lbda < 0:
            raise ValueError('lbda must be positive')
        if self.lbda_m < 0:
            raise ValueError('lbda_m must be positive')
        # if self.A < 0: # Check whether A is S PD would be really expensive
        #     raise ValueError('A must be a symmetric positive operator')
        if self.gamma is not None:
            if self.gamma < 0:
                raise ValueError('sigma must be positive or default (None)')
        if self.theta < 0:
            raise ValueError('theta must be positive')
        if isinstance(self.period, (int, float)):
            if self.period < 0:
                raise ValueError('period must be positive')
        # TODO, add supported solver check

    def _default_decomposable_op(self, y):
        # TODO: check NaN values (semi-sup learning)
        if self.A is not None:
            return self.A
        elif y.ndim == 2:
            return eye(y.shape[1])
        else:
            return eye(1)

    def _default_period(self, X, y):
        if self.period is 'autocorr':
            autocorr_params = self.autocorr_params or {}
            return get_period(X, y, **autocorr_params)
        elif isinstance(self.period, (int, float)):
            return self.period
        else:
            raise ValueError('period must be a positive number or a valid '
                             'string')

    def _get_kernel_map(self, X, y):
        # When adding a new kernel, update this table and the _get_kernel_map
        # method
        if callable(self.ovkernel):
            ovkernel = self.ovkernel
        elif type(self.ovkernel) is str:
            # 1) check string and assign the right parameters
            if self.ovkernel == 'DGauss':
                self.A_ = self._default_decomposable_op(y)
                kernel_params = {'A': self.A_, 'scalar_kernel': rbf_kernel,
                                 'scalar_kernel_params': {'gamma': self.gamma}}
            elif self.ovkernel == 'DPeriodic':
                self.A_ = self._default_decomposable_op(y)
                self.period_ = self._default_period(X, y)
                kernel_params = {'A': self.A_,
                                 'scalar_kernel': first_periodic_kernel,
                                 'scalar_kernel_params': {'gamma': self.theta,
                                                          'period':
                                                          self.period_}, }
            elif self.ovkernel == 'CurlF':
                kernel_params = {'gamma': self.gamma}
            else:
                raise NotImplementedError('unsupported kernel')
            # 2) Uses lookup table to select the right kernel from string
            ovkernel = PAIRWISE_KERNEL_FUNCTIONS[self.ovkernel](
                **kernel_params)
        else:
            raise NotImplementedError('unsupported kernel')
        return ovkernel(X)

    def _decision_function(self, X):
        pred = self.linop_(X) * self.dual_coefs_

        return reshape(pred, (X.shape[0], self.linop_.p)) \
            if self.linop_.p > 1 else pred

    def fit(self, X, y):
        """Fit OVK ridge regression model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data.

        y : {array-like}, shape = [n_samples] or [n_samples, n_targets]
            Target values. numpy.NaN for missing targets (semi-supervised
            learning).

        Returns
        -------
        self : returns an instance of self.
        """
        X = check_array(X, force_all_finite=True, accept_sparse=False,
                        ensure_2d=True)
        y = check_array(y, force_all_finite=False, accept_sparse=False,
                        ensure_2d=False)
        if y.ndim == 1:
            y = check_array(y, force_all_finite=True, accept_sparse=False,
                            ensure_2d=False)
        self._validate_params()

        solver_params = self.solver_params or {}

        self.linop_ = self._get_kernel_map(X, y)
        Gram = self.linop_(X)
        risk = OVKRidgeRisk(self.lbda)

        if not issubdtype(y.dtype, number):
            raise ValueError("Unknown label type: %r" % y.dtype)
        if y.ndim > 1:
            is_sup = ~all(isnan(y), axis=1)
        else:
            is_sup = ~isnan(y)

        if sum(~is_sup) > 0:
            self.L_ = _graph_Laplacian(rbf_kernel(X[~is_sup, :],
                                                  gamma=self.gamma_m))
        else:
            self.L_ = empty((0, 0))

        p = y.shape[1] if y.ndim > 1 else 1
        weight, zeronan = _SemisupLinop(self.lbda_m, is_sup, self.L_, p).gen()

        self.solver_res_ = minimize(risk.functional_grad_val,
                                    zeros(Gram.shape[1]),
                                    args=(y.ravel(), Gram, weight, zeronan),
                                    method=self.solver,
                                    jac=True,
                                    options=solver_params)
        self.dual_coefs_ = self.solver_res_.x
        return self

    def predict(self, X):
        """Predict using the OVK ridge model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : {array}, shape = [n_samples] or [n_samples, n_targets]
            Returns predicted values.
        """
        check_is_fitted(self, ['dual_coefs_', 'linop_'], all_or_any=all)
        X = check_array(X, force_all_finite=True, accept_sparse=False,
                        ensure_2d=True)
        return self._decision_function(X)
