"""
===================================================
Vector-field Learning with structured output kernel
===================================================

An example to illustrate structured learning with operator-valued kernels.

We compare Operator-valued kernel (OVK) with scikit-learn multi-output ridge
regression on a semi-supervised dataset.
"""
# Author: Romain Brault <ro.brault@gmail.com>
# License: MIT

# -*- coding: utf-8 -*-

import operalib as ovk

import numpy as np
from numpy.random import RandomState

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge


def main():
    """Example of vector-field learning."""

    # Fix a seed
    random_state = RandomState(0)

    # Generate data
    inputs, targets = ovk.toy_data_curl_free_field(n_samples=2000)
    inputs_mesh = ovk.array2mesh(inputs)

    (inputs_train, inputs_test,
     targets_train, targets_test) = train_test_split(inputs, targets,
                                                     train_size=(inputs
                                                                 .shape[0] -
                                                                 40 ** 2),
                                                     random_state=random_state)

    # Add unlabelled data
    targets_train = ovk.datasets.awful(targets_train, .85, .1, .5,
                                       random_state=random_state)
    sup_mask = ~np.any(np.isnan(targets_train), axis=1)
    weaksup_mask = ~np.all(np.isnan(targets_train), axis=1)

    scores = {'CF weak semisup': 0, 'CF weak': 0, 'CF': 0, 'Indep': 0}
    regressor = {'CF weak semisup':
                 ovk.OVKRidge(ovkernel=ovk.RBFCurlFreeKernel(gamma=2.),
                              lbda=0.0001, gamma_m=1., lbda_m=0.0001),
                 'CF weak':
                 ovk.OVKRidge(ovkernel=ovk.RBFCurlFreeKernel(gamma=2.),
                              lbda=0.0001),
                 'CF':
                 ovk.OVKRidge(ovkernel=ovk.RBFCurlFreeKernel(gamma=2.),
                              lbda=0.0001),
                 'Indep':
                 KernelRidge(kernel='rbf', alpha=0.0001, gamma=.5)}
    targets_mesh = {'CF weak semisup': (None, None),
                    'CF weak': (None, None),
                    'CF': (None, None),
                    'Indep': (None, None)}

    # Learning with curl-free semisup + weak
    regressor['CF weak semisup'].fit(inputs_train, targets_train)
    scores['CF weak semisup'] = (regressor['CF weak semisup']
                                 .score(inputs_test, targets_test))
    print('R2 curl-free semisup + weak ridge: ', scores['CF weak semisup'])
    targets_mesh['CF weak semisup'] = ovk.array2mesh(
        regressor['CF weak semisup'].predict(inputs))

    # Learning with curl-free weak
    regressor['CF weak'].fit(inputs_train[weaksup_mask, :],
                             targets_train[weaksup_mask, :])
    scores['CF weak'] = regressor['CF weak'].score(inputs_test, targets_test)
    print('R2 curl-free weak ridge: ', scores['CF weak'])
    targets_mesh['CF weak'] = ovk.array2mesh(
        regressor['CF weak'].predict(inputs))

    # Learning with curl-free
    regressor['CF'].fit(inputs_train[sup_mask, :], targets_train[sup_mask, :])
    scores['CF'] = regressor['CF'].score(inputs_test, targets_test)
    print('R2 curl-free ridge: ', scores['CF'])
    targets_mesh['CF'] = ovk.array2mesh(regressor['CF'].predict(inputs))

    # Learning with sklearn ridge
    regressor['Indep'].fit(inputs_train[sup_mask, :],
                           targets_train[sup_mask, :])
    scores['Indep'] = regressor['Indep'].score(inputs_test, targets_test)
    print('R2 independant ridge: ', scores['Indep'])
    targets_mesh['Indep'] = ovk.array2mesh(regressor['Indep'].predict(inputs))

    # Plotting
    # pylint: disable=E1101
    fig, axarr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14, 14))
    axarr[0, 0].streamplot(inputs_mesh[0], inputs_mesh[1],
                           targets_mesh['CF weak semisup'][0],
                           targets_mesh['CF weak semisup'][1],
                           color=np.sqrt(
                               targets_mesh['CF weak semisup'][0]**2 +
                               targets_mesh['CF weak semisup'][1]**2),
                           linewidth=.5, cmap=plt.cm.jet, density=2,
                           arrowstyle=u'->')
    axarr[0, 1].streamplot(inputs_mesh[0], inputs_mesh[1],
                           targets_mesh['CF weak'][0],
                           targets_mesh['CF weak'][1],
                           color=np.sqrt(targets_mesh['CF weak'][0]**2 +
                                         targets_mesh['CF weak'][1]**2),
                           linewidth=.5, cmap=plt.cm.jet, density=2,
                           arrowstyle=u'->')
    axarr[1, 0].streamplot(inputs_mesh[0], inputs_mesh[1],
                           targets_mesh['CF'][0], targets_mesh['CF'][1],
                           color=np.sqrt(targets_mesh['CF'][0]**2 +
                                         targets_mesh['CF'][1]**2),
                           linewidth=.5, cmap=plt.cm.jet, density=2,
                           arrowstyle=u'->')
    axarr[1, 1].streamplot(inputs_mesh[0], inputs_mesh[1],
                           targets_mesh['Indep'][0], targets_mesh['Indep'][1],
                           color=np.sqrt(targets_mesh['Indep'][0]**2 +
                                         targets_mesh['Indep'][1]**2),
                           linewidth=.5, cmap=plt.cm.jet, density=2,
                           arrowstyle=u'->')
    axarr[0, 0].set_ylim([-1, 1])
    axarr[0, 0].set_xlim([-1, 1])
    axarr[0, 0].set_title('Curl-free semisup + weak ridge: ' +
                          str(scores['CF weak semisup']))
    axarr[0, 1].set_ylim([-1, 1])
    axarr[0, 1].set_xlim([-1, 1])
    axarr[0, 1].set_title('Curl-free weak ridge, R2: ' +
                          str(scores['CF weak']))
    axarr[1, 0].set_ylim([-1, 1])
    axarr[1, 0].set_xlim([-1, 1])
    axarr[1, 0].set_title('Curl-free ridge, R2: ' +
                          str(scores['CF']))
    axarr[1, 1].set_ylim([-1, 1])
    axarr[1, 1].set_xlim([-1, 1])
    axarr[1, 1].set_title('Independant ridge, R2: ' +
                          str(scores['Indep']))
    fig.suptitle('Vectorfield learning')
    plt.show()

if __name__ == '__main__':
    main()
