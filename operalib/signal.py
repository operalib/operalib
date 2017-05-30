"""
:mod:`operalib.signal` implements signal preprocessing routines such as period
detection usefull for periodic kernels.
"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
# License: MIT

# pylint: disable=E0611,E0401
from numpy import (correlate, arange, zeros, mean, diff, hstack,
                   where, ndarray, issubdtype, unsignedinteger, argsort, ones)

from sklearn.externals.six.moves import xrange


def indexes(targets, thres=0.05, min_dist=2):
    #     The MIT License (MIT)

    # Copyright (c) 2014 Lucas Hermann Negri

    # Permission is hereby granted, free of charge, to any person obtaining a
    # copy of this software and associated documentation files
    # (the "Software"), to deal in the Software without restriction, including
    # without limitation the rights to use, copy, modify, merge, publish,
    # distribute, sublicense, and/or sell copies of the Software, and to permit
    # persons to whom the Software is furnished to do so, subject to the
    # following conditions:

    # The above copyright notice and this permission notice shall be included
    # in all copies or substantial portions of the Software.

    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
    # OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
    # MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
    # NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
    # DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    # OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
    # USE OR OTHER DEALINGS IN THE SOFTWARE.
    """Peak detection routine.

    Finds the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    targets : array, (signed)
        1D amplitude data to search for peaks.
    thres : float, (between [0., 1.])
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.

    Returns
    -------
    ndarray
        Array containing the indexes of the peaks that were detected
    """
    if isinstance(targets, ndarray) and issubdtype(targets.dtype,
                                                   unsignedinteger):
        raise ValueError("y must be signed")

    thres *= max(targets) - min(targets)

    # find the peaks by using the first order difference
    d_targets = diff(targets)
    peaks = where((hstack([d_targets, 0.]) < 0.) &
                  (hstack([0., d_targets]) > 0.) &
                  (targets > thres))[0]

    if peaks.size > 1 and min_dist > 1:
        highest = peaks[argsort(targets[peaks])][::-1]
        rem = ones(targets.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                dst_to_pick = slice(max(0, peak - min_dist),
                                    peak + min_dist + 1)
                rem[dst_to_pick] = True
                rem[peak] = False

        peaks = arange(targets.size)[~rem]

    return peaks


def autocorrelation(data):
    """Autocorrelation routine.

    Compute the autocorrelation of a given signal 'data'.

    Parameters
    ----------
    data : darray
        1D signal to compute the autocorrelation.

    Returns
    -------
    ndarray
        the autocorrelation of the signal x.
    """
    n_points = len(data)
    variance = data.var()
    data = data - data.mean()
    corr = correlate(data, data, mode='full')[-n_points:]
    result = corr / (variance * arange(n_points, 0, -1))
    return result


def get_period(inputs, targets, thres=0.05, min_dist=2):
    """Period detection routine.

    Finds the period in *targets* by taking its autocorrelation and its first
    order difference. By using *thres* and *min_dist* parameters, it is
    possible to reduce the number of detected peaks. *targets* must be signed.

    Parameters
    ----------
    inputs : ndarray
        support of *targets*.

    targets : ndarray (signed)
        1D amplitude data to search for peaks.

    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.

    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.

    Returns
    -------
    float
        a period estimation of the signal targets = f(inputs).
    """
    spikes_mean_diff = zeros(targets.shape[1])
    for i in xrange(targets.shape[1]):
        auto_corr = autocorrelation(targets[:, i])
        spikes = indexes(auto_corr, thres=(thres / max(auto_corr)),
                         min_dist=min_dist)
        spikes_mean_diff[i] = mean(diff(spikes.ravel()))
    return mean(diff(spikes.ravel())) * mean(diff(inputs.ravel()))
