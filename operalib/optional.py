"""OVK optional.

Module :mod:`sklearn.ovk.optional` implements optinal features for OVK learning
such as period detection for periodic kernels. It includes code relying on
third-party library except scipy/numpy.
"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
# License: MIT

from numpy import correlate, arange, zeros, mean, diff, hstack, finfo, \
    where, ndarray, issubdtype, unsignedinteger, argsort, ones

eps = finfo(float).eps


def indexes(y, thres=0.3, min_dist=1):
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
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
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
    if isinstance(y, ndarray) and issubdtype(y.dtype, unsignedinteger):
        raise ValueError("y must be signed")

    thres *= max(y) - min(y)

    # find the peaks by using the first order difference
    dy = diff(y)
    peaks = where((hstack([dy, 0.]) < 0.) &
                  (hstack([0., dy]) > 0.) &
                  (y > thres))[0]

    if peaks.size > 1 and min_dist > 1:
        highest = peaks[argsort(y[peaks])][::-1]
        rem = ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = arange(y.size)[~rem]

    return peaks


def _autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = correlate(x, x, mode='full')[-n:]
    result = r / (variance * arange(n, 0, -1))
    return result


def _get_period(X, y, thres=0.01, min_dist=10):
    Ts = zeros(y.shape[1])
    for i in xrange(y.shape[1]):
        cb = _autocorrelation(y[:, i])
        T = indexes(cb, thres=(thres / max(cb)), min_dist=min_dist)
        Ts[i] = mean(diff(T.ravel()))
    return mean(diff(T.ravel())) * mean(diff(X.ravel()))
