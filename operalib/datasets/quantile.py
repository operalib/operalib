"""Synthetic datasets for quantile learning."""

from scipy.stats import norm
from numpy.random import rand, randn
from numpy import sort, sin, array, pi


def toy_data_quantile(n_points=50, probs=[0.5], noise=1.):
    """Sine wave toy dataset.

    Parameters
    ----------
    n : {integer}
        Number of samples to generate.

    probs : {list}, shape = [n_quantiles], default=0.5
        Probabilities (quantiles levels)

    Returns
    -------
    X : {array}, shape = [n]
        Input data.

    y : {array}, shape = [n]
        Targets.

    quantiles : {array}, shape = [n x n_quantiles]
        True conditional quantiles.
    """
    probs = array(probs, ndmin=1)

    t_min, t_max = 0., 1.5  # Bounds for the input data
    t_down, t_up = 0., 1.5  # Bounds for the noise
    t_rand = rand(n_points) * (t_max - t_min) + t_min
    t_rand = sort(t_rand)
    pattern = -sin(2 * pi * t_rand)  # Pattern of the signal
    enveloppe = 1 + sin(2 * pi * t_rand / 3)  # Enveloppe of the signal
    pattern = pattern * enveloppe
    # Noise decreasing std (from noise+0.2 to 0.2)
    noise_std = 0.2 + noise * (t_up - t_rand) / (t_up - t_down)
    # Gaussian noise with decreasing std
    add_noise = noise_std * randn(n_points)
    observations = pattern + add_noise
    quantiles = [pattern + array([norm.ppf(p, loc=0., scale=abs(noise_c))
                                  for noise_c in noise_std]) for p in probs]
    return t_rand[:, None], observations, quantiles
