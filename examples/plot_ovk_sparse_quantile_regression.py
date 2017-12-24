"""
======================================================
Data sparse quantile regression with operator-valued kernels
======================================================

An example to illustrate data sparse quantile regression with operator-valued
kernels.

We compare quantile regression with several levels of data sparsity.
"""

# Author: Maxime Sangnier <maxime.sangnier@gmail.com>
# License: MIT

# -*- coding: utf-8 -*-
import time

import numpy as np
import matplotlib.pyplot as plt

from operalib import Quantile, toy_data_quantile


def main():
    """Example of multiple quantile regression."""

    print("Creating dataset...")
    probs = np.linspace(0.1, 0.9, 5)  # Quantile levels of interest
    x_train, y_train, _ = toy_data_quantile(100, random_state=0)
    x_test, y_test, z_test = toy_data_quantile(1000, probs=probs,
                                               random_state=1)

    print("Fitting...")
    methods = {'Non-sparse QR':
               Quantile(probs=probs, kernel='DGauss', lbda=1e-2, gamma=8.,
                        gamma_quantile=1e-2),
                'Sparse QR':
               Quantile(probs=probs, kernel='DGauss', lbda=1e-2, gamma=8.,
                        gamma_quantile=1e-2, eps=2.5)}
    # Fit on training data
    for name, reg in sorted(methods.items()):
        print(name)
        start = time.time()
        reg.fit(x_train, y_train)

        coefs = reg.model_["coefs"].reshape(y_train.size, probs.size)
        n_sv = np.sum(np.linalg.norm(coefs, axis=1) *
                      reg.lbda / y_train.size > 1e-5)
        print(name + ' leaning time: ', time.time() - start)
        print(name + ' score ', reg.score(x_test, y_test))
        print(name + ' num of support vectors ', n_sv)

    # Plot the estimated conditional quantiles
    plt.figure(figsize=(12, 7))
    for i, method in enumerate(sorted(methods.keys())):
        plt.subplot(2, 2, i + 1)
        plt.plot(x_train, y_train, '.')
        plt.gca().set_prop_cycle(None)
        for quantile in methods[method].predict(x_test):
            plt.plot(x_test, quantile, '-')
        plt.gca().set_prop_cycle(None)
        for prob, quantile in zip(probs, z_test):
            plt.plot(x_test, quantile, '--', label="theoretical {0:0.2f}".format(prob))
        plt.title(method)
        plt.legend(fontsize=8)

        coefs = methods[method].model_["coefs"].reshape(y_train.size,
                                                        probs.size)
        plt.subplot(2, 2, 2 + i + 1)
        plt.plot(np.linalg.norm(coefs, axis=1))
        plt.title("Norm of dual coefs for each point")
    plt.show()

if __name__ == '__main__':
    main()
