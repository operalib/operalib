"""OVK learning.

The :mod:`sklearn.ovk` module includes algorithms to learn with Operator-valued
kernels.
"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
# License: MIT


from .kernels import DecomposableKernel
from .risk import KernelRidgeRisk
from .ridge import Ridge
from .metrics import first_periodic_kernel

__all__ = ['DecomposableKernel',
           'KernelRidgeRisk',
           'first_periodic_kernel',
           'Ridge']
