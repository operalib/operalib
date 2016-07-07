"""
:mod:`operalib.ridge` implements Operator-Valued Kernel ridge
regression.
"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
# License: MIT

from scipy.optimize import fmin_l_bfgs_b
from numpy import reshape, eye, zeros

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from sklearn.metrics.pairwise import rbf_kernel

from .metrics import first_periodic_kernel
from .kernels import DecomposableKernel
from .risk import KernelRidgeRisk
from .signal import get_period

# When adding a new kernel, update this table and the _get_kernel_map method
PAIRWISE_KERNEL_FUNCTIONS = {
    'DGauss': DecomposableKernel,
    'DPeriodic': DecomposableKernel, }


class Ridge(BaseEstimator, RegressorMixin):
    """Operator-Valued kernel ridge regression.

    Operator-Valued kernel ridge regression (OVKRR) combines ridge regression
    (linear least squares with l2-norm regularization) with the (OV)kernel
    trick. It thus learns a linear function in the space induced by the
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

    Attributes
    ----------
    dual_coef_ : array, shape = [n_features] or [n_targets, n_features]
        Weight vector(s) in kernel space

    linop_ : callable
        Callable which associate to the training points X the Gram matrix (the
        Gram matrix being a LinearOperator)

    A_ : array, shape = [n_targets, n_targets]
        Set when Linear operator used by the decomposable kernel is default or
        None.

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


    See also
    --------
    sklearn.Ridge
        Linear ridge regression.
    sklearn.KernelRidge
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
    >>> clf = ovk.Ridge('DGauss', lbda=1.0)
    >>> clf.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    Ridge(A=None, autocorr_params=None, gamma=None, kernel='DGauss',
       lbda=1.0, period='autocorr',
       solver=<function fmin_l_bfgs_b at ...>, solver_params=None,
       theta=0.7)
    """

    def __init__(self,
                 kernel='DGauss', lbda=1e-5,
                 A=None, gamma=None, theta=0.7, period='autocorr',
                 autocorr_params=None,
                 solver=fmin_l_bfgs_b, solver_params=None):
        """Initialize OVK ridge regression model.

        Parameters
        ----------

        kernel : {string, callable}, default='DGauss'
            Kernel mapping used internally. A callable should accept two
            arguments, and should return a LinearOperator.

        lbda : {float}, default=1e-5
            Small positive values of lbda improve the conditioning of the
            problem and reduce the variance of the estimates.  Lbda corresponds
            to ``(2*C)^-1`` in other linear models such as LogisticRegression
            or LinearSVC.

        A : {LinearOperator, array-like, sparse matrix}, default=None
            Linear operator used by the decomposable kernel. If default is
            None, wich is set to identity matrix of size y.shape[1] when
            fitting.

        gamma : {float}, default=None.
            Gamma parameter for the Decomposable Gaussian kernel.
            Ignored by other kernels.

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
        self.kernel = kernel
        self.lbda = lbda
        self.A = A
        self.gamma = gamma
        self.theta = theta
        self.period = period
        self.autocorr_params = autocorr_params
        self.solver = solver
        self.solver_params = solver_params

    def _validate_params(self):
        # check on self.kernel is performed in method __get_kernel
        if self.lbda < 0:
            raise ValueError('lbda must be positive')
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
        if callable(self.kernel):
            ov_kernel = self.kernel
        elif type(self.kernel) is str:
            # 1) check string and assign the right parameters
            if self.kernel == 'DGauss':
                self.A_ = self._default_decomposable_op(y)
                kernel_params = {'A': self.A_, 'scalar_kernel': rbf_kernel,
                                 'scalar_kernel_params': {'gamma': self.gamma}}
            elif self.kernel == 'DPeriodic':
                self.A_ = self._default_decomposable_op(y)
                self.period_ = self._default_period(X, y)
                kernel_params = {'A': self.A_,
                                 'scalar_kernel': first_periodic_kernel,
                                 'scalar_kernel_params': {'gamma': self.theta,
                                                          'period':
                                                          self.period_}, }
            else:
                raise NotImplemented('unsupported kernel')
            # 2) Uses lookup table to select the right kernel from string
            ov_kernel = PAIRWISE_KERNEL_FUNCTIONS[self.kernel](**kernel_params)
        else:
            raise NotImplemented('unsupported kernel')
        return ov_kernel(X)

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
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        X, y = check_X_y(X, y, ['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True)
        self._validate_params()

        solver_params = self.solver_params or {}

        self.linop_ = self._get_kernel_map(X, y)
        Gram = self.linop_(X)
        risk = KernelRidgeRisk(self.lbda)
        self.solver_res_ = fmin_l_bfgs_b(risk.functional_grad_val,
                                         zeros(Gram.shape[1]),
                                         args=(y.ravel(), Gram),
                                         *solver_params)
        self.dual_coefs_ = self.solver_res_[0]
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
        X = check_array(X)
        return self._decision_function(X)
