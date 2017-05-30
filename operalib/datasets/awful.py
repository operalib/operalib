"""Simulate supervised and unsupervised."""

from numpy import NaN
from numpy import bool as npbool
from numpy import float as npfloat

from sklearn.utils import check_random_state

# pylint: disable=C0103


def awful(y, p_unsup=.25, p_weaksup=.25, p_weaksup_inner=.25,
          random_state=None):
    """Take a nice dataset and add some NaNs to simulate partially supervised.

    Parameters
    ----------
    y : array, shape = [n_samples, dim_target]
        Targets.

    p_unsup : float, default = .25
        Probability of a single target to be unsupervised.

    p_weaksup : float, default = .25
        Probability of a single target to be weakly supervised.

    p_weaksup_inner : float, default = .25
        Rate of weak supervision.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the
        random number generator; If None, the random number
        generator is the RandomState instance used by `np.random`.

    Returns
    -------
    y : array, shape = [n_samples, dim_target]
        Awful targets.
    """
    random_state_internal = check_random_state(random_state)
    awful_targets = y.copy()
    p_sup = 1 - (p_unsup + p_weaksup)
    awful_mask = (random_state_internal
                  .multinomial(1, [p_sup, p_unsup, p_weaksup],
                               size=y.shape[0])
                  .argmax(axis=1))
    awful_targets[awful_mask == 1, :] = NaN
    weaksup_mask = (random_state_internal
                    .binomial(1, 1 - p_weaksup_inner,
                              awful_targets[awful_mask == 2, :].shape)
                    .astype(npfloat))
    weaksup_mask[~weaksup_mask.astype(npbool)] = NaN
    awful_targets[awful_mask == 2, :] = (awful_targets[awful_mask == 2, :] *
                                         weaksup_mask)

    return awful_targets
