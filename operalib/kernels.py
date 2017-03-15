"""
:mod:`operalib.kernels` implements some Operator-Valued Kernel
models.
"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
# License: MIT

from numpy import dot, diag, sqrt

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.kernel_approximation import RBFSampler, SkewedChi2Sampler
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import svd


class DotProductKernel(object):
    r"""
    Dot product Operator-Valued Kernel of the form:

    .. math::
        x, y \mapsto K(x, y) = \mu \langle x, y \rangle 1_p + (1-\mu) \langle
        x, y \rangle^2 I_p

    Attributes
    ----------
    mu : {array, LinearOperator}, shape = [n_targets, n_targets]
        Tradeoff between shared and independant components

    p : {Int}
        dimension of the targets (n_targets).

    References
    ----------

    See also
    --------

    DotProductKernelMap
        Dot Product Kernel Map

    Examples
    --------
    >>> import operalib as ovk
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> K = ovk.DotProductKernel(mu=.2, p=5)
    >>> # The kernel matrix as a linear operator
    >>> K(X, X)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    <500x500 _CustomLinearOperator with dtype=float64>
    """

    def __init__(self, mu, p):
        """Initialize the  Dot product Operator-Valued Kernel.

        Parameters
        ----------

        mu : {float}
            Tradeoff between shared and independant components.

        p : {integer}
            dimension of the targets (n_targets).
        """
        self.mu = mu
        self.p = p

    def get_kernel_map(self, X):
        r"""Return the kernel map associated with the data X.

        .. math::
               K_x: Y \mapsto K(X, Y)

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        K_x : DotProductKernelMap, callable

        .. math::
            K_x: Y \mapsto K(X, Y).
        """
        from .kernel_maps import DotProductKernelMap
        return DotProductKernelMap(X, self.mu, self.p)

    def __call__(self, X, Y=None):
        r"""Return the kernel map associated with the data X.

        .. math::
               K_x: \begin{cases}
               Y \mapsto K(X, Y) \enskip\text{if } Y \text{is None,} \\
               K(X, Y) \enskip\text{otherwise.}
               \end{cases}

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples1, n_features]
            Samples.

        Y : {array-like, sparse matrix}, shape = [n_samples2, n_features],
                                          default = None
            Samples.

        Returns
        -------
        K_x : DotProductKernelMap, callable or LinearOperator

            .. math::
               K_x: \begin{cases}
               Y \mapsto K(X, Y) \enskip\text{if } Y \text{is None,} \\
               K(X, Y) \enskip\text{otherwise}
               \end{cases}
        """
        Kmap = self.get_kernel_map(X)
        if Y is None:
            return Kmap
        else:
            return Kmap(Y)


class DecomposableKernel(object):
    r"""
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

    References
    ----------

    See also
    --------

    DecomposableKernelMap
        Decomposable Kernel map

    Examples
    --------
    >>> import operalib as ovk
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> K = ovk.DecomposableKernel(np.eye(2))
    >>> # The kernel matrix as a linear operator
    >>> K(X, X)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    <200x200 _CustomLinearOperator with dtype=float64>
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
        r"""Return the kernel map associated with the data X.

        .. math::
               K_x: Y \mapsto K(X, Y)

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        K_x : DecomposableKernelMap, callable

        .. math::
            K_x: Y \mapsto K(X, Y).
        """
        from .kernel_maps import DecomposableKernelMap
        return DecomposableKernelMap(X, self.A,
                                     self.scalar_kernel,
                                     self.scalar_kernel_params)

    def get_orff_map(self, X, D=100, eps=1e-5, random_state=0):
        r"""Return the Random Fourier Feature map associated with the data X.

        .. math::
               K_x: Y \mapsto \tilde{\Phi}(X)

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        \tilde{\Phi}(X) : Linear Operator, callable
        """
        u, s, v = svd(self.A, full_matrices=False, compute_uv=True)
        self.B_ = dot(diag(sqrt(s[s > eps])), v[s > eps, :])
        self.r = self.B_.shape[0]

        if (self.scalar_kernel is rbf_kernel) and not hasattr(self, 'Xb_'):
            if self.scalar_kernel_params is None:
                gamma = 1.
            else:
                gamma = self.scalar_kernel_params['gamma']
            self.phi_ = RBFSampler(gamma=gamma,
                                   n_components=D, random_state=random_state)
            self.phi_.fit(X)
            self.Xb_ = self.phi_.transform(X).astype(X.dtype)
        elif (self.scalar_kernel is 'skewed_chi2') and not hasattr(self,
                                                                   'Xb_'):
            if self.scalar_kernel_params is None:
                skew = 1.
            else:
                skew = self.scalar_kernel_params['skew']
            self.phi_ = SkewedChi2Sampler(skewedness=skew,
                                          n_components=D,
                                          random_state=random_state)
            self.phi_.fit(X)
            self.Xb_ = self.phi_.transform(X).astype(X.dtype)
        elif not hasattr(self, 'Xb_'):
            raise NotImplementedError('ORFF map for kernel is not '
                                      'implemented yet')

        D = self.phi_.n_components
        if X is self.Xb_:
            cshape = (D, self.r)
            rshape = (self.Xb_.shape[0], self.p)
            oshape = (self.Xb_.shape[0] * self.p, D * self.r)
            return LinearOperator(oshape,
                                  dtype=self.Xb_.dtype,
                                  matvec=lambda b: dot(dot(self.Xb_,
                                                           b.reshape(cshape)),
                                                       self.B_),
                                  rmatvec=lambda r: dot(Xb.T,
                                                        dot(r.reshape(rshape),
                                                            self.B_.T)))
        else:
            Xb = self.phi_.transform(X)
            cshape = (D, self.r)
            rshape = (X.shape[0], self.p)
            oshape = (Xb.shape[0] * self.p, D * self.r)
            return LinearOperator(oshape,
                                  dtype=self.Xb_.dtype,
                                  matvec=lambda b: dot(dot(Xb,
                                                           b.reshape(cshape)),
                                                       self.B_),
                                  rmatvec=lambda r: dot(Xb.T,
                                                        dot(r.reshape(rshape),
                                                            self.B_.T)))

    def __call__(self, X, Y=None):
        r"""Return the kernel map associated with the data X.

        .. math::
               K_x: \begin{cases}
               Y \mapsto K(X, Y) \enskip\text{if } Y \text{is None,} \\
               K(X, Y) \enskip\text{otherwise.}
               \end{cases}

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

            .. math::
               K_x: \begin{cases}
               Y \mapsto K(X, Y) \enskip\text{if } Y \text{is None,} \\
               K(X, Y) \enskip\text{otherwise}
               \end{cases}
        """
        Kmap = self.get_kernel_map(X)
        if Y is None:
            return Kmap
        else:
            return Kmap(Y)


class RBFCurlFreeKernel(object):
    r"""
    Curl-free Operator-Valued Kernel of the form:

    .. math::
        X \mapsto K_X(Y) = 2 \gamma exp(-\gamma||X - Y||^2)(I - 2\gamma(X - Y)
        (X - T)^T).

    Attributes
    ----------
    gamma : {float}
        RBF kernel parameter.

    References
    ----------

    See also
    --------

    RBFCurlFreeKernelMap
        Curl-free Kernel map

    Examples
    --------
    >>> import operalib as ovk
    >>> import numpy as np
    >>> X = np.random.randn(100, 2)
    >>> K = ovk.RBFCurlFreeKernel(1.)
    >>> # The kernel matrix as a linear operator
    >>> K(X, X)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    <200x200 _CustomLinearOperator with dtype=float64>
    """

    def __init__(self, gamma):
        """Initialize the Decomposable Operator-Valued Kernel.

        Parameters
        ----------
        gamma : {float}, shape = [n_targets, n_targets]
            RBF kernel parameter.
        """
        self.gamma = gamma

    def get_kernel_map(self, X):
        r"""Return the kernel map associated with the data X.

        .. math::
               K_x: Y \mapsto K(X, Y)

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        K_x : DecomposableKernelMap, callable

        .. math::
            K_x: Y \mapsto K(X, Y).
        """
        from .kernel_maps import RBFCurlFreeKernelMap
        return RBFCurlFreeKernelMap(X, self.gamma)

    def get_orff_map(self, X, D=100, random_state=0):
        r"""Return the Random Fourier Feature map associated with the data X.

        .. math::
               K_x: Y \mapsto \tilde{\Phi}(X)

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        \tilde{\Phi}(X) : Linear Operator, callable
        """
        self.r = 1
        if not hasattr(self, 'Xb_'):
            self.phi_ = RBFSampler(gamma=self.gamma,
                                   n_components=D, random_state=random_state)
            self.phi_.fit(X)
            self.Xb_ = self.phi_.transform(X)
            self.Xb_ = (self.Xb_.reshape((self.Xb_.shape[0],
                                          1, self.Xb_.shape[1])) *
                        self.phi_.random_weights_.reshape((1, -1,
                                                           self.Xb_.shape[1])))
            self.Xb_ = self.Xb_.reshape((-1, self.Xb_.shape[2]))

        D = self.phi_.n_components
        if X is self.Xb_:
            return LinearOperator(self.Xb_.shape,
                                  matvec=lambda b: dot(self.Xb_ * b),
                                  rmatvec=lambda r: dot(self.Xb_.T * r))
        else:
            Xb = self.phi_.transform(X)
            Xb = (Xb.reshape((Xb.shape[0], 1, Xb.shape[1])) *
                  self.phi_.random_weights_.reshape((1, -1, Xb.shape[1])))
            Xb = Xb.reshape((-1, Xb.shape[2]))
            return LinearOperator(Xb.shape,
                                  matvec=lambda b: dot(Xb, b),
                                  rmatvec=lambda r: dot(Xb.T, r))

    def __call__(self, X, Y=None):
        r"""Return the kernel map associated with the data X.

        .. math::
               K_x: \begin{cases}
               Y \mapsto K(X, Y) \enskip\text{if } Y \text{is None,} \\
               K(X, Y) \enskip\text{otherwise.}
               \end{cases}

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

        .. math::
            K_x: \begin{cases}
            Y \mapsto K(X, Y) \enskip\text{if } Y \text{is None,} \\
            K(X, Y) \enskip\text{otherwise}
            \end{cases}
        """
        Kmap = self.get_kernel_map(X)
        if Y is None:
            return Kmap
        else:
            return Kmap(Y)


class RBFDivFreeKernel(object):
    r"""
    Divergence-free Operator-Valued Kernel of the form:

    .. math::
        X \mapsto K_X(Y) = exp(-\gamma||X-Y||^2)A_{X,Y},

    where,

    .. math::
        A_{X,Y} = 2\gamma(X-Y)(X-T)^T+((d-1)-2\gamma||X-Y||^2 I).

    Attributes
    ----------
    gamma : {float}
        RBF kernel parameter.

    References
    ----------

    See also
    --------

    RBFDivFreeKernelMap
        Divergence-free Kernel map

    Examples
    --------
    >>> import operalib as ovk
    >>> import numpy as np
    >>> X = np.random.randn(100, 2)
    >>> K = ovk.RBFDivFreeKernel(1.)
    >>> # The kernel matrix as a linear operator
    >>> K(X, X)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
    <200x200 _CustomLinearOperator with dtype=float64>
    """

    def __init__(self, gamma):
        """Initialize the Decomposable Operator-Valued Kernel.

        Parameters
        ----------
        gamma : {float}, shape = [n_targets, n_targets]
            RBF kernel parameter.
        """
        self.gamma = gamma

    def get_kernel_map(self, X):
        r"""Return the kernel map associated with the data X.

        .. math::
               K_x: Y \mapsto K(X, Y)

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        K_x : DecomposableKernelMap, callable

        .. math::
            K_x: Y \mapsto K(X, Y).
        """
        from .kernel_maps import RBFDivFreeKernelMap
        return RBFDivFreeKernelMap(X, self.gamma)

    def get_orff_map(self, X, D=100, random_state=0):
        r"""Return the Random Fourier Feature map associated with the data X.

        .. math::
               K_x: Y \mapsto \tilde{\Phi}(X)

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        \tilde{\Phi}(X) : Linear Operator, callable
        """
        self.r = 1
        if not hasattr(self, 'Xb_'):
            self.phi_ = RBFSampler(gamma=self.gamma,
                                   n_components=D, random_state=random_state)
            self.phi_.fit(X)
            self.Xb_ = self.phi_.transform(X)
            self.Xb_ = (self.Xb_.reshape((self.Xb_.shape[0],
                                          1, self.Xb_.shape[1])) *
                        self.phi_.random_weights_.reshape((1, -1,
                                                           self.Xb_.shape[1])))
            self.Xb_ = self.Xb_.reshape((-1, self.Xb_.shape[2]))

        D = self.phi_.n_components
        if X is self.Xb_:
            return LinearOperator(self.Xb_.shape,
                                  matvec=lambda b: dot(self.Xb_ * b),
                                  rmatvec=lambda r: dot(self.Xb_.T * r))
        else:
            Xb = self.phi_.transform(X)
            # TODO:
            # w = self.phi_.random_weights_.reshape((1, -1, Xb.shape[1]))
            # wn = np.linalg.norm(w)
            # Xb = (Xb.reshape((Xb.shape[0], 1, Xb.shape[1])) *
            #       wn * np.eye()w np.dot(w.T, w) / wn)
            Xb = Xb.reshape((-1, Xb.shape[2]))
            return LinearOperator(Xb.shape,
                                  matvec=lambda b: dot(Xb, b),
                                  rmatvec=lambda r: dot(Xb.T, r))

    def __call__(self, X, Y=None):
        r"""Return the kernel map associated with the data X.

        .. math::
               K_x: \begin{cases}
               Y \mapsto K(X, Y) \enskip\text{if } Y \text{is None,} \\
               K(X, Y) \enskip\text{otherwise.}
               \end{cases}

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

        .. math::
            K_x: \begin{cases}
            Y \mapsto K(X, Y) \enskip\text{if } Y \text{is None,} \\
            K(X, Y) \enskip\text{otherwise}
            \end{cases}
        """
        Kmap = self.get_kernel_map(X)
        if Y is None:
            return Kmap
        else:
            return Kmap(Y)
