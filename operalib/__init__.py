"""OVK learning.

The :mod:`operalib` module includes algorithms to learn with Operator-valued
kernels.
"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
# License: MIT

import sys
import re
import warnings

# Make sure that DeprecationWarning within this package always gets printed
warnings.filterwarnings('always', category=DeprecationWarning,
                        module='^{0}\.'.format(re.escape(__name__)))

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#

__author__ = ['Romain Brault']
__copyright__ = 'Copyright 2015, Operalib'
__credits__ = ['Romain Brault', 'Florence D\'Alche Buc',
               'Markus Heinonen', 'Tristan Tchilinguirian',
               'Alexandre Gramfort']
__license__ = 'MIT'
__version__ = '0.1b0'
__maintainer__ = ['Romain Brault']
__email__ = ['romain.brault@telecom-paristech.fr']
__status__ = 'Beta'

try:
    # This variable is injected in the __builtins__ by the build
    # process. It used to enable importing subpackages of sklearn when
    # the binaries are not built
    __OPERALIB_SETUP__
except NameError:
    __OPERALIB_SETUP__ = False

if __OPERALIB_SETUP__:
    sys.stderr.write('Partial import of operalib during the build process.\n')
    # We are not importing the rest of the scikit during the build
    # process, as it may not be compiled yet
else:
    from .kernels import DecomposableKernel, RBFCurlFreeKernel, \
        RBFDivFreeKernel
    from .risk import KernelRidgeRisk
    from .ridge import Ridge
    from .metrics import first_periodic_kernel
    from .datasets.vectorfield import generate_2D_curl_free_field, \
        generate_2D_curl_free_mesh, mesh2array, array2mesh, \
        generate_2D_div_free_field, generate_2D_div_free_mesh

    __all__ = ['DecomposableKernel', 'RBFCurlFreeKernel', 'RBFDivFreeKernel',
               'KernelRidgeRisk',
               'first_periodic_kernel',
               'Ridge',
               'generate_2D_curl_free_field', 'generate_2D_curl_free_mesh',
               'generate_2D_div_free_field', 'generate_2D_div_free_mesh',
               'mesh2array', 'array2mesh']
