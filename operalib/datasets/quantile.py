"""Synthetic datasets for quantile learning."""

from scipy.stats import norm
from numpy.random import rand, randn
from numpy import sort, sin, asarray, pi


def toy_data_quantile(n=50, probs=[0.5], noise=1.):
    """Sine wave toy dataset.

    Parameters
    ----------
    n : {integer}
        Number of samples to generate.

    probs : {list}, shape = [n_quantiles]
        Probabilities (quantiles levels).

    Returns
    -------
    X : {array}, shape = [n]
        Input data.

    y : {array}, shape = [n]
        Targets.

    quantiles : {array}, shape = [n x n_quantiles]
        True conditional quantiles.
    """
    t_min, t_max = 0., 1.5  # Bounds for the input data
    t_down, t_up = 0., 1.5  # Bounds for the noise
    t = rand(n) * (t_max - t_min) + t_min
    t = sort(t)
    pattern = -sin(2 * pi * t)  # Pattern of the signal
    enveloppe = 1 + sin(2 * pi * t / 3)  # Enveloppe of the signal
    pattern = pattern * enveloppe
    # Noise decreasing std (from noise+0.2 to 0.2)
    noise_std = 0.2 + noise * (t_up - t) / (t_up - t_down)
    # Gaussian noise with decreasing std
    add_noise = noise_std * randn(n)
    observations = pattern + add_noise
    quantiles = [pattern + asarray([norm.ppf(p, loc=0., scale=abs(noise_c))
                 for noise_c in noise_std]) for p in probs]
    return t[:, None], observations, quantiles
