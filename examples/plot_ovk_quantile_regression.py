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

from operalib import Quantile, toy_data_quantile

import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(0)

print("Creating dataset...")
probs = np.linspace(0.1, 0.9, 5)  # Quantile levels of interest
x_train, y_train, z_train = toy_data_quantile(50)
x_test, y_test, z_test = toy_data_quantile(1000, probs=probs)

print("Fitting...")
# Joint quantile regression
lbda = 1e-2
gamma = 1e1
joint = Quantile(probs=probs, kernel='DGauss', lbda=lbda, gamma=gamma,
                 gamma_quantile=1e-2)
# Independent quantile regression
ind = Quantile(probs=probs, kernel='DGauss', lbda=lbda, gamma=gamma,
               gamma_quantile=np.inf)
# Independent quantile regression (with non-crossing constraints)
nc = Quantile(probs=probs, kernel='DGauss', lbda=lbda, gamma=gamma,
              gamma_quantile=np.inf, nc_const=True)

# Fit on training data
methods = {'joint': joint,
           'independant': ind,
           'non-crossing': nc}
for name, reg in methods.items():
    start = time.time()
    reg.fit(x_train, y_train)
#    pred = joint.predict(x_test)
    print(name + ' leaning time: ', time.time() - start)
    print(name + ' score ', reg.score(x_test, y_test))

# Plot the estimated conditional quantiles

plt.figure(figsize=(12, 7))
for (i, (reg, title)) in enumerate(
    [(joint, 'Joint'),
     (ind, 'Independent'),
     (nc, 'Independent (non-crossing)')]):
    plt.subplot(1, 3, i + 1)
    plt.plot(x_train, y_train, '.')
    for q in reg.predict(x_test):
        plt.plot(x_test, q, '-')
    for q in z_test:
        plt.plot(x_test, q, '--')
    plt.title(title)
plt.show()
