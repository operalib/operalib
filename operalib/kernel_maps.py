"""
:mod:`operalib.kernel_maps` implement some Operator-Valued Kernel
maps associated to the operator-valued kernel models defined in
:mod:`operalib.kernels`.
"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
# License: MIT

from scipy.sparse.linalg import LinearOperator
from numpy import ravel, dot, reshape

from .kernels import DecomposableKernel


class DecomposableKernelMap(DecomposableKernel):
    r"""Decomposable Operator-Valued Kernel.

    Decomposable Operator-Valued Kernel map of the form:

    .. math::
        X \mapsto K_X(Y) = k_s(X, Y) A

    where A is a symmetric positive semidefinite operator acting on the
    outputs. This class just fixes the support data X to the kernel. Hence it
    naturally inherit from Decomposable kernels

    Attributes
    ----------

    n: {Int}
        Number of samples.

    d: {Int}
        Number of features

    X: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Support samples.

    Gs: {array-like, sparse matrix}, shape = [n, n]
        Gram matrix associated with the scalar kernel

    References
    ----------

    See also
    --------

    DecomposableKernel
        Decomposable Kernel

    Examples
    --------
    """

    def __init__(self, X, A, scalar_kernel, scalar_kernel_params):
        """Initialize the Decomposable Operator-Valued Kernel.

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples1, n_features]
            Support samples.

        A : {array, LinearOperator}, shape = [n_targets, n_targets]
            Linear operator acting on the outputs

        scalar_kernel : {callable}
            Callable which associate to the training points X the Gram matrix.

        scalar_kernel_params : {mapping of string to any}, optional
            Additional parameters (keyword arguments) for kernel function
            passed as callable object.
        """
        super(DecomposableKernelMap, self).__init__(A, scalar_kernel,
                                                    scalar_kernel_params)
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.X = X
        self.Gs_train = None

    def __mul__(self, Ky):
        """Syntaxic sugar.

           If Kx is a compatible decomposable kernel, returns

           .. math::
                K(X, Y) = K_X^T K_Y

        Parameters
        ----------
        Ky : {DecomposableKernelMap}
            Compatible kernel Map (e.g. same kernel but different support data
            X).

        Returns
        -------
        K(X, Y) : LinearOperator
            Returns K(X, Y).
        """
        # TODO: Check that Kx is compatible
        return self.__call__(Ky.X)

    def _Gram(self, X):
        kernel_params = self.scalar_kernel_params or {}
        if X is self.X:
            if self.Gs_train is None:
                self.Gs_train = self.scalar_kernel(self.X, **kernel_params)
            return self.Gs_train

        return self.scalar_kernel(X, self.X, **kernel_params)

    def _dot(self, Gs, c):
        return ravel(dot(dot(Gs, reshape(c, (self.n, self.p))), self.A))

    def __call__(self, Y):
        """Return the Gram matrix associated with the data Y.

        .. math::
               K(X, Y)

        Parameters
        ----------
        Y : {array-like, sparse matrix}, shape = [n_samples1, n_features]
            Samples.

        Returns
        -------
        K(X, Y) : LinearOperator
            Returns K(X, Y).
        """
        return LinearOperator(
            (Y.shape[0] * self.p, self.n * self.p),
            matvec=lambda b: self._dot(self._Gram(Y), b),
            rmatvec=lambda b: self._dot(self._Gram(Y), b))
