"""
===================================================
Vector-field Learning with structured output kernel
===================================================

An example to illustrate structured learning with operator-valued kernels.

We compare Operator-valued kernel (OVK) with scikit-learn multi-output ridge
regression.
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
yc = y.copy() + .05 * np.random.randn(y.shape[0], y.shape[1])  # Add some noise
K = ovk.RBFCurlFreeKernel(1.)
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, train_size=100)

regr_1 = ovk.Ridge(kernel=ovk.RBFCurlFreeKernel(gamma=2.), lbda=0.0001)

# Learning with curl-free
regr_1.fit(Xtr, ytr)
s1 = regr_1.score(Xte, yte)
print('R2 curl-free ridge: ', s1)
yp1 = regr_1.predict(Xc)
X1, Y1 = ovk.array2mesh(Xc)
U1, V1 = ovk.array2mesh(yp1)

# Learning with sklearn ridge
regr_2 = KernelRidge(kernel='rbf', alpha=0.0001, gamma=1.5)
regr_2.fit(Xtr, ytr)
s2 = regr_2.score(Xte, yte)
print('R2 independant ridge: ', s2)
yp2 = regr_2.predict(X)
X2, Y2 = ovk.array2mesh(X)
U2, V2 = ovk.array2mesh(yp2)

# Plotting
f, axarr = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 7))
axarr[0].streamplot(X1, Y1, U1, V1, color=np.sqrt(U1**2 + V1**2),
                    linewidth=.5, cmap=plt.cm.jet, density=2, arrowstyle=u'->')
axarr[1].streamplot(X2, Y2, U2, V2, color=np.sqrt(U2**2 + V2**2),
                    linewidth=.5, cmap=plt.cm.jet, density=2, arrowstyle=u'->')
axarr[0].set_ylim([-1, 1])
axarr[0].set_title('Curl-Free Ridge, R2: ' + str(s1))
axarr[1].set_ylim([-1, 1])
axarr[1].set_title('Independant Ridge, R2: ' + str(s2))
f.suptitle('Vectorfield learning')
plt.show()
