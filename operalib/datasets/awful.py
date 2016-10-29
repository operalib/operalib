"""Simulate supervised and unsupervised."""

from numpy import NaN, bool, float
from numpy.random import multinomial, binomial


def awful(targets, p_unsup=.25, p_weaksup=.25, p_weaksup_inner=.25):
    """Take a nice dataset and add some NaNs to simulate partially supervised.

    Parameters
    ----------
    y : {array}, shape = [n_outputs, dim_outputs]
        Targets.

    p_unsup : probability of a single target to be unsupervised

    p_weaksup : probability of a single target to be weakly supervised

    p_weaksup_inner : rate of weak supervision

    Returns
    -------
    y : {array}, shape = [n_outputs, dim_outputs]
        Awful targets.
    """
    awful_targets = targets.copy()
    p_sup = 1 - (p_unsup + p_weaksup)
    awful_mask = multinomial(1, [p_sup, p_unsup, p_weaksup],
                             size=targets.shape[0]).argmax(axis=1)
    awful_targets[awful_mask == 1, :] = NaN
    weaksup_mask = binomial(1, 1 - p_weaksup_inner,
                            awful_targets[awful_mask == 2,
                                          :].shape).astype(float)
    weaksup_mask[~weaksup_mask.astype(bool)] = NaN
    awful_targets[awful_mask == 2, :] = \
        awful_targets[awful_mask == 2, :] * weaksup_mask

    return awful_targets
