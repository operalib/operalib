"""Synthetic dataset."""


from .vectorfield import generate_2D_curl_free_field, \
    generate_2D_curl_free_mesh

__all__ = ['generate_2D_curl_free_field', 'generate_2D_curl_free_mesh']
