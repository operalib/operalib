"""Synthetic datasets for vector-field learning."""

from numpy import arange, sqrt, meshgrid, pi, exp, gradient, empty, floor


def _gaussian(X, Y, mean_X, mean_Y, scale=1):
    Xc = X - mean_X
    Yc = Y - mean_Y
    return pi ** 2 * exp(- scale / 2 * (Xc ** 2 + Yc ** 2)) / sqrt(scale)


def array2mesh(arr, side=None):
    if side is None:
        side = int(floor(sqrt(arr.shape[0])))
    X = arr[:, 0].reshape((side, side))
    y = arr[:, 1].reshape((side, side))

    return X, y


def mesh2array(X, Y):
    arr = empty((X.size, 2))
    arr[:, 0] = X.ravel()
    arr[:, 1] = Y.ravel()

    return arr


def toy_data_curl_free_mesh(n=1000, loc=25, space=0.5):
    """Curl-Free toy dataset.

    Generate a scalar field as mixture of five gaussians at location:
        - (0   , 0)
        - (0   , loc)
        - ( loc, 0)
        - (-loc, 0)
        - (0   , -loc)
    whith variance equal to 'space'. Then return the gradient of the field.

    Parameters
    ----------
    n : {integer}
        Number of samples to generate.

    loc: {float}
        Centers of the Gaussians.

    space: {float}
        Variance of the Gaussians.


    Returns
    -------
    X : {array}, shape = [n, n]
        Mesh, X coordinates.

    Y : {array}, shape = [n, n]
        Mesh, Y coordinates.

    U : {array}, shape = [n, n]
        Mesh, X velocity as (X, Y) coordinates

    V : {array}, shape = [n, n]
        Mesh, Y velocity as (X, Y) coordinates
    """
    xs = arange(-1, 1, 2. / sqrt(n))
    ys = arange(-1, 1, 2. / sqrt(n))
    X, Y = meshgrid(xs, ys)
    field = _gaussian(X, Y, -space, 0, loc) + \
        _gaussian(X, Y, space, 0, loc) - \
        _gaussian(X, Y, 0, space, loc) - \
        _gaussian(X, Y, 0, -space, loc)
    V, U = gradient(field)

    return X, Y, U, V


def toy_data_div_free_mesh(n=1000, loc=25, space=0.5):
    """Divergence-Free toy dataset.

    Generate a scalar field as mixture of five gaussians at location:
        - (0   , 0)
        - (0   , loc)
        - ( loc, 0)
        - (-loc, 0)
        - (0   , -loc)
    whith variance equal to 'space'. Then return the orthogonal of gradient of
    the field.

    Parameters
    ----------
    n : {integer}
        Number of samples to generate.

    loc: {float}
        Centers of the Gaussians.

    space: {float}
        Variance of the Gaussians.


    Returns
    -------
    X : {array}, shape = [n, n]
        Mesh, X coordinates.

    Y : {array}, shape = [n, n]
        Mesh, Y coordinates.

    U : {array}, shape = [n, n]
        Mesh, X velocity as (X, Y) coordinates

    V : {array}, shape = [n, n]
        Mesh, Y velocity as (X, Y) coordinates
    """
    xs = arange(-1, 1, 2. / sqrt(n))
    ys = arange(-1, 1, 2. / sqrt(n))
    X, Y = meshgrid(xs, ys)
    field = _gaussian(X, Y, -space, 0, loc) + \
        _gaussian(X, Y, space, 0, loc) - \
        _gaussian(X, Y, 0, space, loc) - \
        _gaussian(X, Y, 0, -space, loc)
    V, U = gradient(field)

    return X, Y, V, -U


def toy_data_curl_free_field(n=1000, loc=25, space=0.5):
    X, Y, U, V = toy_data_curl_free_mesh(n, loc, space)

    X = mesh2array(X, Y)
    y = mesh2array(U, V)
    return X, y


def toy_data_div_free_field(n=1000, loc=25, space=0.5):
    X, Y, U, V = toy_data_div_free_mesh(n, loc, space)

    X = mesh2array(X, Y)
    y = mesh2array(U, V)
    return X, y
