import os

from operalib._build_utils import get_blas_info


def configuration(parent_package='', top_path=None):
    """Get BLAS configuration."""
    from numpy.distutils.misc_util import Configuration

    config = Configuration('operalib', parent_package, top_path)

    cblas_libs, blas_info = get_blas_info()

    if os.name == 'posix':
        cblas_libs.append('m')

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
