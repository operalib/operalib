"""
===================================================
Vector-field Learning with structured output kernel
===================================================

An example to illustrate structured learning with operator-valued kernels.

We compare Operator-valued kernel (OVK) with scikit-learn multi-output ridge
regression on a semi-supervised dataset.
"""

import operalib as ovk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.kernel_ridge import KernelRidge

# Generate data
np.random.seed(0)
X, y = ovk.toy_data_curl_free_field(n=2000)
Xc = X.copy()
# Add some noise
yc = y.copy() + .075 * np.random.randn(y.shape[0], y.shape[1])
K = ovk.RBFCurlFreeKernel(1.)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, train_size=500)

# Add unlabelled data
ytr = ovk.datasets.awful(ytr, .85, .1, .5)
sup_mask = ~np.any(np.isnan(ytr), axis=1)
weaksup_mask = ~np.all(np.isnan(ytr), axis=1)

# Learning with curl-free semisup + weak
regr_1 = ovk.OVKRidge(ovkernel=ovk.RBFCurlFreeKernel(gamma=2.), lbda=0.0001,
                      gamma_m=1., lbda_m=0.0001)
regr_1.fit(Xtr, ytr)
s1 = regr_1.score(Xte, yte)
print('R2 curl-free semisup + weak ridge: ', s1)
yp1 = regr_1.predict(Xc)
X1, Y1 = ovk.array2mesh(Xc)
U1, V1 = ovk.array2mesh(yp1)

# Learning with curl-free weak
regr_2 = ovk.OVKRidge(ovkernel=ovk.RBFCurlFreeKernel(gamma=2.), lbda=0.0001)
regr_2.fit(Xtr[weaksup_mask, :], ytr[weaksup_mask, :])
s2 = regr_2.score(Xte, yte)
print('R2 curl-free weak ridge: ', s2)
yp2 = regr_2.predict(Xc)
X2, Y2 = ovk.array2mesh(Xc)
U2, V2 = ovk.array2mesh(yp2)

# Learning with curl-free
regr_3 = ovk.OVKRidge(ovkernel=ovk.RBFCurlFreeKernel(gamma=2.), lbda=0.0001)
regr_3.fit(Xtr[sup_mask, :], ytr[sup_mask, :])
s3 = regr_3.score(Xte, yte)
print('R2 curl-free ridge: ', s3)
yp3 = regr_3.predict(Xc)
X3, Y3 = ovk.array2mesh(Xc)
U3, V3 = ovk.array2mesh(yp3)

# Learning with sklearn ridge
regr_4 = KernelRidge(kernel='rbf', alpha=0.0001, gamma=.5)
regr_4.fit(Xtr[sup_mask, :], ytr[sup_mask, :])
s4 = regr_4.score(Xte, yte)
print('R2 independant ridge: ', s4)
yp4 = regr_4.predict(X)
X4, Y4 = ovk.array2mesh(X)
U4, V4 = ovk.array2mesh(yp4)

# Plotting
f, axarr = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(14, 7))
axarr[0, 0].streamplot(X1, Y1, U1, V1, color=np.sqrt(U1**2 + V1**2),
                       linewidth=.5, cmap=plt.cm.jet, density=2,
                       arrowstyle=u'->')
axarr[0, 1].streamplot(X2, Y2, U2, V2, color=np.sqrt(U2**2 + V2**2),
                       linewidth=.5, cmap=plt.cm.jet, density=2,
                       arrowstyle=u'->')
axarr[1, 0].streamplot(X3, Y3, U3, V3, color=np.sqrt(U2**2 + V2**2),
                       linewidth=.5, cmap=plt.cm.jet, density=2,
                       arrowstyle=u'->')
axarr[1, 1].streamplot(X4, Y4, U4, V4, color=np.sqrt(U2**2 + V2**2),
                       linewidth=.5, cmap=plt.cm.jet, density=2,
                       arrowstyle=u'->')
axarr[0, 0].set_ylim([-1, 1])
axarr[0, 0].set_xlim([-1, 1])
axarr[0, 0].set_title('Curl-free semisup + weak ridge: ' + str(s1))
axarr[0, 1].set_ylim([-1, 1])
axarr[0, 1].set_xlim([-1, 1])
axarr[0, 1].set_title('Curl-free weak ridge, R2: ' + str(s2))
axarr[1, 0].set_ylim([-1, 1])
axarr[1, 0].set_xlim([-1, 1])
axarr[1, 0].set_title('Curl-free ridge, R2: ' + str(s3))
axarr[1, 1].set_ylim([-1, 1])
axarr[1, 1].set_xlim([-1, 1])
axarr[1, 1].set_title('Independant ridge, R2: ' + str(s4))
f.suptitle('Vectorfield learning')
plt.show()
