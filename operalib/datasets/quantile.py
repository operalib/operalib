"""Synthetic datasets for quantile learning."""

from scipy.stats import norm
from numpy import sort, sin, array, pi

from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator


class SinePattern(BaseEstimator):
    """"Sine pattern.

    The inputs X is drawn uniformly in the 1D interval [min, max].  The target
    y is computed as a sine wave with some period modulated by a sine wave
    envelope with another period and shifted by a constant.

    Attributes
    ----------
    inputs_bound : :rtype: (float, float)
        The bounds (min, max) of the inputs.

    sine_period : float
        The period of the sine wave.

    enveloppe : :rtype: (float, float)
        The (shift, period) of the enveloppe of the enveloppe.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    References
    ----------
    * Maxime Sangnier, Olivier Fercoq, Florence D'Alche-Buc
    "Joint quantile regression in vector-valued RKHSs." In Advances in Neural
    Information Processing Systems (2016, pp. 3693-3701).
    """

    def __init__(self, inputs_bound=(0., 1.5), sine_period=1.,
                 enveloppe=(1. / 3., 1.), random_state=None):
        """Initialize the sine pattern.

        Parameters
        ----------
        inputs_bound : :rtype: (float, float), optional (default = (0., 1.5))
            The bounds (min, max) of the inputs

        sine_period : float, optional (default = 1.0)
            The period of the sine wave.

        enveloppe : :rtype: (float, float), optional (default = (1. / 3., 1.))
            The (shift, period) of the enveloppe of the sine wave

        random_state : int, RandomState instance or None,
                       optional (default=None)
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by `np.random`.
        """
        self.inputs_bound = inputs_bound
        self.sine_period = sine_period
        self.enveloppe = enveloppe
        self.random_state = random_state

    def __call__(self, n_samples):
        """Generate the data.

        Parameters
        ----------

        n_points : int
            The number of points in the generated dataset.

        Returns
        -------

        inputs : array, shape = [n_samples, 1]
            Returns the generated inputs.

        targets : array, shape = [n_samples, 1]
            Returns the generated targets.
        """
        random_state = check_random_state(self.random_state)
        envelope_period, envelope_shift = self.enveloppe
        inputs = (random_state.rand(n_samples) *
                  (self.inputs_bound[1] - self.inputs_bound[0]) +
                  self.inputs_bound[0])
        # Pattern of the signal
        inputs = sort(inputs)
        pattern = -sin(2 * pi * inputs * self.sine_period)

        # Enveloppe of the signal
        enveloppe = envelope_shift + sin(2 * pi * inputs * envelope_period)

        return (inputs.reshape((n_samples, 1)),
                (pattern * enveloppe).reshape((n_samples, 1)))


def toy_data_quantile(n_samples=50, probs=0.5, pattern=SinePattern(),
                      noise=(1., .2, 0., 1.5), random_state=None):
    """Sine wave toy dataset.

    The target y is computed as a sine curve at of modulated by a sine envelope
    with some period (default 1/3Hz) and mean (default 1).  Moreover, this
    pattern is distorted with a random Gaussian noise with mean 0 and a
    linearly decreasing standard deviation (default from 1.2 at X = 0 to 0.2 at
    X = 1 .  5).

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.

    probs : list or float, shape = [n_quantiles], default=0.5
        Probabilities (quantiles levels).

    pattern : callable, default = SinePattern()
        Callable which generates n_sample 1D data (inputs and targets).

    noise : :rtype: (float, float, float, float)
        Noise parameters (variance, shift, support_min, support_max).

    Returns
    -------
    X : array, shape = [n_samples, 1]
        Input data.

    y : array, shape = [n_sample, 1]
        Targets.

    quantiles : array, shape = [n_samples x n_quantiles]
        True conditional quantiles.
    """
    probs = array(probs, ndmin=1)
    noise_var, noise_shift, noise_min, noise_max = noise

    random_state = check_random_state(random_state)
    pattern.random_state = random_state
    inputs, targets = pattern(n_samples)

    # Noise decreasing std (from noise+0.2 to 0.2)
    noise_std = noise_shift + (noise_var * (noise_max - inputs) /
                               (noise_max - noise_min))
    # Gaussian noise with decreasing std
    add_noise = noise_std * random_state.randn(n_samples, 1)
    observations = targets + add_noise
    quantiles = [targets + array([norm.ppf(p, loc=0., scale=abs(noise_c))
                                  for noise_c in noise_std]) for p in probs]
    return inputs, observations.ravel(), quantiles
