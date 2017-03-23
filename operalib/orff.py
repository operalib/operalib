"""
:mod:`operalib.orff` implements Operator-Valued Random Fourier Features
regression.
"""

from scipy.optimize import minimize
from numpy import eye, zeros

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import rbf_kernel

from .kernels import DecomposableKernel, RBFCurlFreeKernel
from .risk import ORFFRidgeRisk

# When adding a new kernel, update this table and the _get_kernel_map method
PAIRWISE_KERNEL_FUNCTIONS = {
    'DGauss': DecomposableKernel,
    'DSkewed_chi2': DecomposableKernel,
    'DPeriodic': DecomposableKernel,
    'CurlF': RBFCurlFreeKernel}


class ORFFRidge(BaseEstimator, RegressorMixin):

    def __init__(self, ovkernel='DGauss', loss='LS', lbda=1e-5,
                 A=None, gamma=1., D=1000, skew=0.,
                 solver='L-BFGS-B', solver_params=None):
        self.ovkernel = ovkernel
        self.loss = loss
        self.lbda = lbda
        self.A = A
        self.gamma = gamma
        self.skew = skew
        self.D = D
        self.solver = solver
        self.solver_params = solver_params

    def _validate_params(self):
        # check on self.ovkernel is performed in method __get_kernel
        if self.D < 0:
            raise ValueError('lbda must be a positive integer')
        if self.lbda < 0:
            raise ValueError('lbda must be a positive float')
        # if self.A < 0: # Check whether A is S PD would be really expensive
        #     raise ValueError('A must be a symmetric positive operator')
        if self.gamma is not None:
            if self.gamma < 0:
                raise ValueError('sigma must be positive or default (None)')
        if self.skew is not None:
            if self.skew < 0:
                raise ValueError('sigma must be positive or default (None)')
        # TODO, add supported solver check

    def _get_kernel(self, X, y):
        # When adding a new kernel, update this table and the _get_kernel_map
        # method
        if callable(self.ovkernel):
            ov_kernel = self.ovkernel
        elif isinstance(self.ovkernel, str):
            # 1) check string and assign the right parameters
            if self.ovkernel == 'DGauss':
                self.A_ = self._default_decomposable_op(y)
                kernel_params = {'A': self.A_, 'scalar_kernel': rbf_kernel,
                                 'scalar_kernel_params': {'gamma': self.gamma}}
            elif self.ovkernel == 'DSkewed_chi2':
                self.A_ = self._default_decomposable_op(y)
                kernel_params = {'A': self.A_, 'scalar_kernel': 'skewed_chi2',
                                 'scalar_kernel_params': {'skew': self.skew}}
            elif self.ovkernel == 'CurlF':
                kernel_params = {'gamma': self.gamma}
            else:
                raise NotImplementedError('unsupported kernel')
            # 2) Uses lookup table to select the right kernel from string
            ov_kernel = \
                PAIRWISE_KERNEL_FUNCTIONS[self.ovkernel](**kernel_params)
        else:
            raise NotImplementedError('unsupported kernel')
        return ov_kernel

    def _default_decomposable_op(self, y):
        if self.A is not None:
            return self.A
        elif y.ndim == 2:
            return eye(y.shape[1])
        else:
            return eye(1)

    def _decision_function(self, X):
        pred = self.linop_.get_orff_map(X) * self.coefs_

        return pred.reshape((X.shape[0], self.p)) \
            if self.p > 1 else pred

    def fit(self, X, y):
        """Fit ORFF ridge regression model.

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
        self.p = y.shape[1] if y.ndim > 1 else 1

        solver_params = self.solver_params or {}

        self.linop_ = self._get_kernel(X, y)
        self.phix_ = self.linop_.get_orff_map(X, self.D)
        risk = ORFFRidgeRisk(self.lbda, self.loss)
        self.solver_res_ = minimize(risk.functional_grad_val,
                                    zeros(self.phix_.shape[1],
                                          dtype=X.dtype),
                                    args=(y.ravel(), self.phix_, self.linop_),
                                    method=self.solver,
                                    jac=True, options=solver_params)
        self.coefs_ = self.solver_res_.x
        return self

    def predict(self, X):
        """Predict using the ORFF ridge model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : {array}, shape = [n_samples] or [n_samples, n_targets]
            Returns predicted values.
        """
        check_is_fitted(self, ['coefs_', 'linop_'], all_or_any=all)
        X = check_array(X)
        return self._decision_function(X)
