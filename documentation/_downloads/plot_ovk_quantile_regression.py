"""
======================================================
Joint quantile regression with operator-valued kernels
======================================================

An example to illustrate joint quantile regression with operator-valued
kernels.

We compare quantile regression estimation with and without non-crossing
constraints.
"""

# Author: Maxime Sangnier <maxime.sangnier@gmail.com>
# License: MIT

# -*- coding: utf-8 -*-
import time

import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt

from operalib import Quantile, toy_data_quantile


def main():
    """example of multiple quantile regression."""
    seed(0)

    print("Creating dataset...")
    probs = np.linspace(0.1, 0.9, 5)  # Quantile levels of interest
    x_train, y_train, _ = toy_data_quantile(50)
    x_test, y_test, z_test = toy_data_quantile(1000, probs=probs)

    print("Fitting...")
    methods = {'Joint':
               Quantile(probs=probs, kernel='DGauss', lbda=1e-2, gamma=8,
                        gamma_quantile=1e-2),
               'Independent': Quantile(probs=probs, kernel='DGauss', lbda=1e-2,
                                       gamma=8, gamma_quantile=np.inf),
               'Non-crossing': Quantile(probs=probs, kernel='DGauss',
                                        lbda=1e-2, gamma=8,
                                        gamma_quantile=np.inf, nc_const=True)}
    # Fit on training data
    for name, reg in methods.items():
        start = time.time()
        reg.fit(x_train, y_train)
        print(name + ' leaning time: ', time.time() - start)
        print(name + ' score ', reg.score(x_test, y_test))

    # Plot the estimated conditional quantiles
    plt.figure(figsize=(12, 7))
    for i, method in enumerate(['Joint', 'Independent', 'Non-crossing']):
        plt.subplot(1, 3, i + 1)
        plt.plot(x_train, y_train, '.')
        plt.gca().set_prop_cycle(None)
        for quantile in methods[method].predict(x_test):
            plt.plot(x_test, quantile, '-')
        plt.gca().set_prop_cycle(None)
        for quantile in z_test:
            plt.plot(x_test, quantile, '--')
        plt.title(method)
    plt.show()

if __name__ == '__main__':
    main()
