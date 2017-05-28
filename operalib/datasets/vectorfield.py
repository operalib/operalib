"""Synthetic datasets for vector-field learning."""

from numpy import arange, sqrt, meshgrid, pi, exp, gradient, empty, floor


def _gaussian(X, Y, mean_X, mean_Y, scale=1):
    Xc = X - mean_X
    Yc = Y - mean_Y
    return pi ** 2 * exp(- scale / 2 * (Xc ** 2 + Yc ** 2)) / sqrt(scale)


def array2mesh(arr, side=None):
    """Array to mesh converter."""
    if side is None:
        side = int(floor(sqrt(arr.shape[0])))
    X = arr[:, 0].reshape((side, side))
    y = arr[:, 1].reshape((side, side))

    return X, y


def mesh2array(X, Y):
    """Mesh to array converter."""
    arr = empty((X.size, 2))
    arr[:, 0] = X.ravel()
    arr[:, 1] = Y.ravel()

    return arr


def toy_data_curl_free_mesh(n_points=1000, loc=25, space=0.5):
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
    n_points : {integer}
        Number of samples to generate.

    loc: {float}
        Centers of the Gaussians.

    space: {float}
        Variance of the Gaussians.


    Returns
    -------
    X, Y : :rtype: (array, array), shape = [n, n]
            Mesh, X, Y coordinates.

    U, V : :rtype: (array, array), shape = [n, n]
           Mesh, (U, V) velocity at (X, Y) coordinates
    """
    x_grid = arange(-1, 1, 2. / sqrt(n_points))
    y_grid = arange(-1, 1, 2. / sqrt(n_points))
    x_mesh, y_mesh = meshgrid(x_grid, y_grid)
    field = _gaussian(x_mesh, y_mesh, -space, 0, loc) + \
        _gaussian(x_mesh, y_mesh, space, 0, loc) - \
        _gaussian(x_mesh, y_mesh, 0, space, loc) - \
        _gaussian(x_mesh, y_mesh, 0, -space, loc)

    return (x_mesh, y_mesh), gradient(field)


def toy_data_div_free_mesh(n_points=1000, loc=25, space=0.5):
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
    n_points : {integer}
        Number of samples to generate.

    loc: {float}
        Centers of the Gaussians.

    space: {float}
        Variance of the Gaussians.


    Returns
    -------
    X, Y : pair{array}, {array}}, shape = [n, n]
            Mesh, X, Y coordinates.

    U, V : pair{{array}, {array}}, shape = [n, n]
           Mesh, (U, V) velocity at (X, Y) coordinates
    """
    x_grid = arange(-1, 1, 2. / sqrt(n_points))
    y_grid = arange(-1, 1, 2. / sqrt(n_points))
    x_mesh, y_mesh = meshgrid(x_grid, y_grid)
    field = _gaussian(x_mesh, y_mesh, -space, 0, loc) + \
        _gaussian(x_mesh, y_mesh, space, 0, loc) - \
        _gaussian(x_mesh, y_mesh, 0, space, loc) - \
        _gaussian(x_mesh, y_mesh, 0, -space, loc)
    v_mesh, u_mesh = gradient(field)

    return (x_mesh, y_mesh), (v_mesh, -u_mesh)


def toy_data_curl_free_field(n_points=1000, loc=25, space=0.5):
    (X, Y), (U, V) = toy_data_curl_free_mesh(n_points, loc, space)

    X = mesh2array(X, Y)
    y = mesh2array(U, V)
    return X, y


def toy_data_div_free_field(n_points=1000, loc=25, space=0.5):
    (X, Y), (U, V) = toy_data_div_free_mesh(n_points, loc, space)

    X = mesh2array(X, Y)
    y = mesh2array(U, V)
    return X, y
