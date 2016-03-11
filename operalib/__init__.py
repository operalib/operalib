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

__author__ = ['Romain Brault']
__copyright__ = 'Copyright 2015, Operalib'
__credits__ = ['Romain Brault', 'Florence D\'Alche Buc',
               'Markus Heinonen', 'Tristan Tchilinguirian',
               'Alexandre Gramfort']
__license__ = 'MIT'
__version__ = '0.0.1'  # Don't forget to change in setup.py
__maintainer__ = ['Romain Brault']
__email__ = ['romain.brault@telecom-paristech.fr']
__status__ = 'Prototype'

__all__ = ['DecomposableKernel',
           'KernelRidgeRisk',
           'first_periodic_kernel',
           'Ridge']
