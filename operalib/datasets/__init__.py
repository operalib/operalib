"""Synthetic dataset."""


from .vectorfield import generate_2D_curl_free_field, \
    generate_2D_curl_free_mesh, array2mesh, mesh2array

__all__ = ['generate_2D_curl_free_field', 'generate_2D_curl_free_mesh',
           'array2mesh', 'mesh2array']
