"""Synthetic dataset."""


from .vectorfield import toy_data_curl_free_field, \
    toy_data_div_free_mesh, array2mesh, mesh2array, \
    toy_data_curl_free_mesh, toy_data_div_free_field
from .quantile import toy_data_quantile

__all__ = ['toy_data_curl_free_field', 'toy_data_div_free_mesh',
           'toy_data_div_free_field', 'toy_data_curl_free_mesh',
           'array2mesh', 'mesh2array',
           'toy_data_quantile']
