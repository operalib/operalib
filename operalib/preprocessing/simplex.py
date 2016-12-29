"""Simplex coding module."""

from numpy import dot, array, vstack, hstack, ones, zeros, sqrt


def scode(n):
    """Simplex coding operator."""
    return _scode_i(n - 1)


def _scode_i(n):
    """Simplex coding operator (internal).

    https://papers.nips.cc/paper/4764-multiclass-learning-with-simplex-coding.pdf
    """
    if n > 1:
        C1 = vstack((ones((1, 1)), zeros((n - 1, 1))))
        C2 = vstack((-ones((1, n)) / n,
                     _scode_i(n - 1) * sqrt(1. - 1. / (n * n))))
        return hstack((C1, C2))
    if n == 1:
        return array([1, -1])
    if n < 1:
        raise "Dimension n should be at least one"""


def sencode(A):
    """Simplex coding encoder."""
    encoder = scode(A.shape[1]).T
    return dot(A, encoder)


def sdecode(A):
    """Simplex coding decoder."""
    decoder = scode(A.shape[1] + 1)
    return dot(A, decoder).argmax(axis=1)
