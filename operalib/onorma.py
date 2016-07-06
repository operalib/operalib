"""
:mod:`operalib.ridge` implements Operator-valued Naive Online
Regularised Risk Minimization Algorithm (ONORMA)
"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
# License: MIT

from numpy import reshape, eye, zeros, ravel, vstack

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_X_y, check_array, shuffle
from sklearn.utils.validation import check_is_fitted

from .metrics import first_periodic_kernel
from .kernels import DecomposableKernel, DotProductKernel
from .learningrate import Constant


PAIRWISE_KERNEL_FUNCTIONS = {
    'DGauss': DecomposableKernel,
    'DPeriodic': DecomposableKernel,
    'DotProduct': DotProductKernel}


class ONORMA(BaseEstimator, RegressorMixin):
    """Operator-Valued Operator-valued Naive Online Regularised Risk
    Minimization Algorithm .

    Operator-Valued kernel Operator-valued Naive Online Regularised Risk
    Minimization Algorithm (ONORMA) hat extends the standard kernel-based
    online learning algorithm NORMA from scalar-valued to operator-valued
    setting

    Attributes
    ----------
    coef_ : array, shape = [n_features] or [n_targets, n_features]
        Weight vector(s) in kernel space

    linop_ : callable
        Callable which associate to the training points X the Gram matrix (the
        Gram matrix being a LinearOperator)

    A_ : array, shape = [n_targets, n_targets]
        Set when Linear operator used by the decomposable kernel is default or
        None.

    T_ : integer
        Total number of iterations

    n_ : integer
        Total number of datapoints

    p_ : integer
        Dimensionality of the outputs


    References
    ----------
    * Audiffren, Julien, and Hachem Kadri.
      "Online learning with multiple operator-valued kernels."
      arXiv preprint arXiv:1311.0222 (2013).

    * Kivinen, Jyrki, Alexander J. Smola, and Robert C. Williamson.
      "Online learning with kernels."
      IEEE transactions on signal processing 52.8 (2004): 2165-2176.


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
    >>> clf = ovk.ONORMA('DGauss', lbda=1.0)
    >>> clf.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ONORMA(A=None, T=None, gamma=None, kernel='DGauss', kernel_params=None,
        lbda=1.0, learning_rate=None, mu=0.2, random_state=0, shuffle=True,
        truncation=None)
    """

    def __init__(self, kernel='DGauss', lbda=1e-5,
                 T=None, A=None, learning_rate=None, truncation=None,
                 gamma=None, mu=0.2, kernel_params=None, shuffle=True,
                 random_state=0):
        """Initialize ONORMA.

        Parameters
        ----------

        kernel : {string, callable}, default='DGauss'
            Kernel mapping used internally. A callable should accept two
            arguments and the keyword arguments passed to this object as
            kernel_params, and should return a LinearOperator.

        lbda : {float}, default=1e-5
            Small positive values of lbda improve the conditioning of the
            problem and reduce the variance of the estimates.  Lbda corresponds
            to ``(2*C)^-1`` in other linear models such as LogisticRegression
            or LinearSVC.

        T : {integer}, default=None
            Number of iterations.

        A : {LinearOperator, array-like, sparse matrix}, default=None
            Linear operator used by the decomposable kernel. If default is
            None, wich is set to identity matrix of size y.shape[1] when
            fitting.

        learning_rate : {Callable}
            Learning rate, a function that return the step size at given step

        truncation : learning_rate : {Callable}
            TODO

        gamma : {float}, default=None.
            Gamma parameter for the Decomposable Gaussian kernel.
            Ignored by other kernels.

        kernel_params : {mapping of string to any}, optional
            Additional parameters (keyword arguments) for kernel function
            passed as callable object.
        """
        self.kernel = kernel
        self.lbda = lbda
        self.T = T
        self.A = A
        self.learning_rate = learning_rate
        self.truncation = truncation
        self.gamma = gamma
        self.mu = mu
        self.kernel_params = kernel_params
        self.shuffle = shuffle
        self.random_state = random_state

    def _validate_params(self):
        # check on self.kernel is performed in method __get_kernel
        if self.lbda < 0:
            raise ValueError('lbda must be a positive scalar')
        if self.mu < 0 or self.mu > 1:
            raise ValueError('mu must be a scalar between 0. and 1.')
        if self.T is not None:
            if self.T <= 0:
                raise ValueError('T must be a positive integer')
        # if self.A < 0: # Check whether A is S PD would be really expensive
        #     raise ValueError('A must be a symmetric positive operator')
        if self.gamma is not None:
            if self.gamma < 0:
                raise ValueError('gamma must be positive or default (None)')

    def _default_decomposable_op(self, y):
        if self.A is not None:
            return self.A
        elif y.ndim == 2:
            return eye(y.shape[1])
        else:
            return eye(1)

    def _get_kernel_map(self, X, y):
        # When adding a new kernel, update this table and the _get_kernel_map
        # method
        if callable(self.kernel):
            kernel_params = self.kernel_params or {}
            ov_kernel = self.kernel(**kernel_params)
        elif type(self.kernel) is str:
            # 1) check string and assign the right parameters
            if self.kernel == 'DGauss':
                self.A_ = self._default_decomposable_op(y)
                kernel_params = {'A': self.A_, 'scalar_kernel': rbf_kernel,
                                 'scalar_kernel_params': {'gamma': self.gamma}}
            elif self.kernel == 'DotProduct':
                kernel_params = {'mu': self.mu, 'p': y.shape[1]}
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
        self.linop_ = self._get_kernel_map(self.X_seen_, self.y_seen_)
        pred = self.linop_(X) * self.coefs_[:self.t_ * self.p_]

        return reshape(pred, (X.shape[0], -1)) if self.linop_.p > 1 else pred

    def predict(self, X):
        """Predict using ONORMA model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : {array}, shape = [n_samples] or [n_samples, n_targets]
            Returns predicted values.
        """
        check_is_fitted(self, ['coefs_', 'n_', 'p_'], all_or_any=all)
        X = check_array(X)
        return self._decision_function(X)

    def partial_fit(self, X, y, n=None, p=None):
        """Partial fit of ONORMA model.

        This method is usefull for online learning for instance.
        Must call

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training data.

        y : {array-like}, shape = [n_samples] or [n_samples, n_targets]
            Target values.

        n : {integer}
            Total number of data.
            This argument is required for the first call to partial_fit and can
            be omitted in the subsequent calls.

        p : {integer}
            Dimensionality of the outputs.
            This argument is required for the first call to partial_fit and can
            be omitted in the subsequent calls.

        Returns
        -------
        self : returns an instance of self.
        """
        if (n is not None) and (p is not None):
            X, y = check_X_y(X, y, ['csr', 'csc', 'coo'],
                             y_numeric=True, multi_output=True)
            self._validate_params()

            self.n_ = n
            self.d_ = X.shape[1]
            self.p_ = y.shape[1]

            if self.T is None:
                self.T_ = n
            else:
                self.T_ = self.T
            if self.learning_rate is None:
                self.learning_rate_ = Constant(1.)
            else:
                self.learning_rate_ = self.learning_rate

            self.coefs_ = zeros(self.T_ * p)

            self.t_ = 0
            eta_t = self.learning_rate_(self.t_ + 1)

            self.coefs_[self.t_ * self.p_:(self.t_ + 1) * self.p_] -= ravel(
                eta_t * (0 - y))
            self.X_seen_ = X
            self.y_seen_ = y

            self.t_ += 1

            return self

        eta_t = self.learning_rate_(self.t_ + 1)

        self.coefs_[self.t_ * self.p_:(self.t_ + 1) * self.p_] -= ravel(
            eta_t * (self._decision_function(X) - y))
        self.coefs_[:self.t_ * self.p_] *= (1. - eta_t * self.lbda)

        self.X_seen_ = vstack((self.X_seen_, X))
        self.y_seen_ = vstack((self.y_seen_, y))

        self.t_ += 1

        return self

    def fit(self, X, y):
        """Fit ONORMA model.

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

        n = X.shape[0]
        p = y.shape[1]

        self.partial_fit(X[0, :].reshape(1, -1),
                         y[0, :].reshape(1, -1), n, p)
        for self.t_ in range(1, self.T_):
            idx = self.t_ % self.n_
            self.partial_fit(X[idx, :].reshape(1, -1),
                             y[idx, :].reshape(1, -1))

        return self
