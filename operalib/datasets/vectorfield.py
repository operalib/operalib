"""Synthetic datasets for vector-field learning."""

from numpy import arange, sqrt, meshgrid, pi, exp, gradient, floor, hstack

# pylint: disable=C0103


def _gaussian(x_mesh, y_mesh, x_mesh_mean, y_mesh_mean, scale=1):
    """Generate a gaussian.

    Parameters
    ----------
    x_mesh : array, shape = [n_samples, n_samples]
        The inputs mesh x_axis.

    y_mesh : array, shape = [n_samples, n_samples]
        The inputs mesh y_axis.

    x_mesh_mean :
        The x_axis center of the Gaussian.

    y_mesh_mean :
        The y_axis center of the Gaussian.

    scale :
        The scale parameter of the Gaussian. Must be positive.

    Returns
    -------
        field : array, shape = [n_samples, n_samples]
            A scalar field with a Gaussian with some scale centered at
            x_mesh_mean, y_mesh_mean.
    """
    x_mesh_centered = x_mesh - x_mesh_mean
    y_mesh_centered = y_mesh - y_mesh_mean
    return pi ** 2 * exp(- scale / 2 * (x_mesh_centered ** 2 +
                                        y_mesh_centered ** 2)) / sqrt(scale)


def array2mesh(X, side=None):
    """Array to mesh converter.

    Parameters
    ----------
    X : array, shape = [n_samples, 2]
        The inputs array.

    Returns
    -------
    x_mesh : array, shape = [n_samples, n_samples]
        The x_axis of the mesh corresponding to inputs.

    y_mesh : array, shape = [n_samples, n_samples]
        The y_axis of the mesh corresponding to inputs

    """
    if side is None:
        side = int(floor(sqrt(X.shape[0])))
    x_mesh = X[:, 0].reshape((side, side))
    y_mesh = X[:, 1].reshape((side, side))

    return x_mesh, y_mesh


def mesh2array(x_mesh, y_mesh):
    """Mesh to array converter.

    Parameters
    ----------
    x_mesh : array, shape = [n_samples, n_samples]
        The x_axis mesh.

    y_mesh : array, shape = [n_samples, n_samples]
        The y_axis mesh.

    Returns
    -------
    inputs : array, shape = [n_samples, 2]
        The inputs corresponding to the mesh (x_mesh, y_mesh).

    """
    return hstack((x_mesh.ravel().reshape((-1, 1)),
                   y_mesh.ravel().reshape((-1, 1))))


def toy_data_curl_free_mesh(n_samples, loc=25., space=0.5):
    """Curl-Free toy dataset.

    Generate a scalar field as mixture of five gaussians at location:
        - (0   , 0)
        - (0   , loc)
        - ( loc, 0)
        - (-loc, 0)
        - (0   , -loc)
    whith variance equal to 'space'. Then return the gradient of the field.
    The return result is a pair meshes.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.

    loc: float, optional (default = 25.)
        Centers of the Gaussians.

    space: float, optional (default = .5)
        Variance of the Gaussians.


    Returns
    -------
    X, Y : :rtype: (array, array), shape = [n, n]
            Mesh, X, Y coordinates.

    U, V : :rtype: (array, array), shape = [n, n]
           Mesh, (U, V) velocity at (X, Y) coordinates

    See also
    --------
    operalib.toy_data_curl_free_field
        Generate Curl-Free field.

    operalib.toy_data_div_free_field
        Generate Divergence-Free field.
    """
    x_grid = arange(-1, 1, 2. / sqrt(n_samples))
    y_grid = arange(-1, 1, 2. / sqrt(n_samples))
    x_mesh, y_mesh = meshgrid(x_grid, y_grid)
    field = (_gaussian(x_mesh, y_mesh, -space, 0, loc) +
             _gaussian(x_mesh, y_mesh, space, 0, loc) -
             _gaussian(x_mesh, y_mesh, 0, space, loc) -
             _gaussian(x_mesh, y_mesh, 0, -space, loc))
    v_mesh, u_mesh = gradient(field)

    return (x_mesh, y_mesh), (u_mesh, v_mesh)


def toy_data_div_free_mesh(n_samples, loc=25., space=0.5):
    """Divergence-Free toy dataset.

    Generate a scalar field as mixture of five gaussians at location:
        - (0   , 0)
        - (0   , loc)
        - ( loc, 0)
        - (-loc, 0)
        - (0   , -loc)
    whith variance equal to 'space'. Then return the orthogonal of gradient of
    the field. The return result is a pair of meshes.

    Parameters
    ----------
    n_points : int
        Number of samples to generate.

    loc: float, optional (default = 25.)
        Centers of the Gaussians.

    space: float, optional (default = .5)
        Variance of the Gaussians.


    Returns
    -------
    X, Y : :rtype: (array, array), shape = [n, n]
            Mesh, X, Y coordinates.

    U, V : :rtype: (array, array), shape = [n, n]
           Mesh, (U, V) velocity at (X, Y) coordinates

    See also
    --------
    operalib.toy_data_curl_free_field
        Generate Curl-Free field.

    operalib.toy_data_div_free_field
        Generate Divergence-Free field.
    """
    x_grid = arange(-1, 1, 2. / sqrt(n_samples))
    y_grid = arange(-1, 1, 2. / sqrt(n_samples))
    x_mesh, y_mesh = meshgrid(x_grid, y_grid)
    field = (_gaussian(x_mesh, y_mesh, -space, 0, loc) +
             _gaussian(x_mesh, y_mesh, space, 0, loc) -
             _gaussian(x_mesh, y_mesh, 0, space, loc) -
             _gaussian(x_mesh, y_mesh, 0, -space, loc))
    v_mesh, u_mesh = gradient(field)

    return (x_mesh, y_mesh), (v_mesh, -u_mesh)


def toy_data_curl_free_field(n_samples, loc=25, space=0.5):
    """Curl-Free toy dataset.

    Generate a scalar field as mixture of five gaussians at location:
        - (0   , 0)
        - (0   , loc)
        - ( loc, 0)
        - (-loc, 0)
        - (0   , -loc)
    whith variance equal to 'space'. Then return the gradient of the field.
    The return result is a pair (inputs, targets) of arrays.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.

    loc: float, optional (default = 25.)
        Centers of the Gaussians.

    space: float, optional (default = .5)
        Variance of the Gaussians.


    Returns
    -------
    X : array, shape = [n_samples, 2]
        Array of evenly space points.

    y : array shape = [n_samples, 2]
        Array corresponding to the velocity at the coordinates present in
        inputs.

    See also
    --------
    operalib.toy_data_curl_free_mesh
        Generate Curl-Free mesh.

    operalib.toy_data_div_free_mesh
        Generate Divergence-Free mesh.
    """
    (x_mesh, y_mesh), (u_mesh, v_mesh) = toy_data_curl_free_mesh(n_samples,
                                                                 loc, space)

    inputs = mesh2array(x_mesh, y_mesh)
    targets = mesh2array(u_mesh, v_mesh)
    return inputs, targets


def toy_data_div_free_field(n_samples, loc=25, space=0.5):
    """Divergence-Free toy dataset.

    Generate a scalar field as mixture of five gaussians at location:
        - (0   , 0)
        - (0   , loc)
        - ( loc, 0)
        - (-loc, 0)
        - (0   , -loc)
    whith variance equal to 'space'. Then return the orthogonal of gradient of
    the field. The return result is a pair (inputs, targets) of arrays.

    Parameters
    ----------
    n_points : int
        Number of samples to generate.

    loc: float, optional (default = 25.)
        Centers of the Gaussians.

    space: float, optional (default = .5)
        Variance of the Gaussians.


    Returns
    -------
    X : array, shape = [n_samples, 2]
        Array of evenly space points.

    y : array shape = [n_samples, 2]
        Array corresponding to the velocity at the coordinates present in
        inputs.

    See also
    --------
    operalib.toy_data_div_free_mesh
        Generate Curl-Free mesh.

    operalib.toy_data_div_free_mesh
        Generate Divergence-Free mesh.
    """
    (x_mesh, y_mesh), (u_mesh, v_mesh) = toy_data_div_free_mesh(n_samples,
                                                                loc, space)

    inputs = mesh2array(x_mesh, y_mesh)
    targets = mesh2array(u_mesh, v_mesh)
    return inputs, targets
