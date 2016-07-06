"""Synthetic dataset."""


from .vectorfield import generate_2D_curl_free_field, \
    generate_2D_curl_free_mesh, array2mesh, mesh2array, \
    generate_2D_div_free_mesh, generate_2D_div_free_field
from .quantile import generate_quantile_data

__all__ = ['generate_2D_curl_free_field', 'generate_2D_curl_free_mesh',
           'generate_2D_div_free_field', 'generate_2D_div_free_mesh',
           'array2mesh', 'mesh2array',
           'generate_quantile_data']
