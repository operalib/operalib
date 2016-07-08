"""
:mod:`operalib.ridge` implements Operator-valued Naive Online
Regularised Risk Minimization Algorithm (ONORMA)
"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
# License: MIT

import dill

from numpy import eye, empty, ravel, vstack, zeros

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted

from .metrics import first_periodic_kernel
from .kernels import DecomposableKernel, DotProductKernel
from .learningrate import Constant, InvScaling

# When adding a new kernel, update this table and the _get_kernel_map
# method
PAIRWISE_KERNEL_FUNCTIONS = {
    'DGauss': DecomposableKernel,
    'DPeriodic': DecomposableKernel,
    'DotProduct': DotProductKernel}

# When adding a new learning rate, update this table and the _get_learning_rate
# method
LEARNING_RATE_FUNCTIONS = {
    'constant': Constant,
    'invscaling': InvScaling}


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
    >>> clf = ovk.ONORMA('DGauss', lbda=1.)
    >>> clf.fit(X, y)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    ONORMA(A=None, T=None, eta=1.0, gamma=None, kernel='DGauss', lbda=1.0,
        learning_rate='invscaling', mu=0.2, power=0.5, random_state=0,
        shuffle=True, truncation=None)
    """

    def __init__(self, kernel='DGauss', lbda=1e-5,
                 T=None, A=None, learning_rate='invscaling', truncation=None,
                 gamma=None, mu=0.2, eta=1., power=.5,
                 shuffle=True, random_state=0):
        """Initialize ONORMA.

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
        """
        self.kernel = kernel
        self.lbda = lbda
        self.T = T
        self.A = A
        self.learning_rate = learning_rate
        self.truncation = truncation
        self.gamma = gamma
        self.mu = mu
        self.shuffle = shuffle
        self.random_state = random_state
        self.eta = eta
        self.power = power

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
            ov_kernel = self.kernel
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
        return ov_kernel

    def _get_learning_rate(self):
        if callable(self.learning_rate):
            return self.learning_rate
        elif type(self.learning_rate) is str:
            # 1) check string and assign the right parameters
            if self.learning_rate == 'constant':
                lr_params = {'eta': self.eta}
            elif self.learning_rate == 'invscaling':
                lr_params = {'eta': self.eta, 'power': self.power}
            else:
                raise NotImplemented('unsupported kernel')
            lr = LEARNING_RATE_FUNCTIONS[self.learning_rate](**lr_params)
        else:
            raise NotImplemented('unsupported learning rate')
        return lr

    def _decision_function(self, X):
        self.linop_ = self.ov_kernel_(self.X_seen_)
        pred = self.linop_(X) * self.coefs_[:self.t_ * self.p_]

        return pred.reshape(X.shape[0], -1) if self.linop_.p > 1 else pred

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
        check_is_fitted(self, ['coefs_', 't_', 'p_',
                               'X_seen_', 'y_seen_'], all_or_any=all)
        X = check_array(X)
        return self._decision_function(X)

    def partial_fit(self, X, y):
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
        n = 1 if X.ndim <= 1 else X.shape[0]
        Xt = X.reshape(n, -1) if X.ndim <= 1 else X
        yt = y.reshape(n, -1) if y.ndim <= 1 else y
        init = not (hasattr(self, 'coefs_') and hasattr(self, 't_'))
        if hasattr(self, 't_'):
            init = self.t_ == 0
        if init:
            Xtt = Xt[0, :].reshape(1, -1)
            ytt = yt[0, :].reshape(1, -1)
            self.d_ = Xtt.shape[1]
            self.p_ = ytt.shape[1]

            self.learning_rate_ = self._get_learning_rate()

            self.coefs_ = empty(self.p_)
            eta_t = self.learning_rate_(1)
            self.coefs_[:self.p_] = -ravel(eta_t * (0 - ytt))
            self.X_seen_ = Xtt
            self.y_seen_ = ytt

            self.ov_kernel_ = self._get_kernel_map(self.X_seen_, self.y_seen_)

            self.t_ = 1

        # Reshape if self.coefs_ has not been preallocated
        self.coefs_.resize((self.t_ + (n - 1 if init else n)) * self.p_)
        for idx in range(1 if init else 0, n):
            Xtt = Xt[idx, :].reshape(1, -1)
            ytt = yt[idx, :].reshape(1, -1)
            eta_t = self.learning_rate_(self.t_ + 1)

            # Update weights
            self.coefs_[self.t_ * self.p_:(self.t_ + 1) * self.p_] = -ravel(
                eta_t * (self._decision_function(Xtt) - ytt))
            self.coefs_[:self.t_ * self.p_] *= (1. - eta_t * self.lbda)

            # Update seen data
            self.X_seen_ = vstack((self.X_seen_, Xtt))
            self.y_seen_ = vstack((self.y_seen_, ytt))

            # prepare next step
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
        X, y = check_X_y(X, y, None, y_numeric=True, multi_output=True)
        self._validate_params()
        self.T_ = X.shape[0] if self.T is None else self.T

        self.t_ = 0
        if y.ndim > 1:
            self.coefs_ = zeros(self.T_ * y.shape[1])
            for i in range(self.T_):
                idx = i % X.shape[0]
                self.partial_fit(X[idx, :], y[idx, :])
        else:
            self.coefs_ = zeros(self.T_)
            for i in range(self.T_):
                idx = i % X.shape[0]
                self.partial_fit(X[idx, :], y[idx])
        return self
