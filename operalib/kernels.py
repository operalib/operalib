"""
:mod:`operalib.kernels` implements some Operator-Valued Kernel
models.
"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
# License: MIT

from sklearn.metrics.pairwise import rbf_kernel


class DecomposableKernel(object):
    """Decomposable Operator-Valued Kernel.

    Decomposable Operator-Valued Kernel of the form:

    .. math::
        X, Y \mapsto K(X, Y) = k_s(X, Y) A

    where A is a symmetric positive semidefinite operator acting on the
    outputs.

    Attributes
    ----------
    A : {array, LinearOperator}, shape = [n_targets, n_targets]
        Linear operator acting on the outputs

    scalar_kernel : {callable}
        Callable which associate to the training points X the Gram matrix.

    scalar_kernel_params : {mapping of string to any}
        Additional parameters (keyword arguments) for kernel function passed as
        callable object.

    p : {Int}
        dimension of the targets (n_targets).

    References
    ----------

    See also
    --------

    DecomposableKernelMap
        Decomposable Kernel map

    Examples
    --------
    """

    def __init__(self, A, scalar_kernel=rbf_kernel, scalar_kernel_params=None):
        """Initialize the Decomposable Operator-Valued Kernel.

        Parameters
        ----------

        A : {array, LinearOperator}, shape = [n_targets, n_targets]
            Linear operator acting on the outputs

        scalar_kernel : {callable}
            Callable which associate to the training points X the Gram matrix.

        scalar_kernel_params : {mapping of string to any}, optional
            Additional parameters (keyword arguments) for kernel function
            passed as callable object.
        """
        self.A = A
        self.scalar_kernel = scalar_kernel
        self.scalar_kernel_params = scalar_kernel_params
        self.p = A.shape[0]

    def get_kernel_map(self, X):
        """Return the kernel map associated with the data X.

               K_x: Y -> K(X, Y)

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        K_x : DecomposableKernelMap, callable
            Returns K_x: Y -> K(X, Y).
        """
        from .kernel_maps import DecomposableKernelMap
        return DecomposableKernelMap(X, self.A,
                                     self.scalar_kernel,
                                     self.scalar_kernel_params)

    def __call__(self, X, Y=None):
        """Return the kernel map associated with the data X.

               K_x: Y -> K(X, Y) if Y is None
               K(X, Y) otherwise

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples1, n_features]
            Samples.

        Y : {array-like, sparse matrix}, shape = [n_samples2, n_features],
                                          default = None
            Samples.

        Returns
        -------
        K_x : DecomposableKernelMap, callable or LinearOperator
            Returns K_x: Y -> K(X, Y) if Y is None; K(X, Y) otherwise.
        """
        Kmap = self.get_kernel_map(X)
        if Y is None:
            return Kmap
        else:
            return Kmap(Y)
