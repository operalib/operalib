"""OVK risk.

Module :mod:`sklearn.ovk.risk` implements risk model and their gradients.
"""
# Author: Romain Brault <romain.brault@telecom-paristech.fr> with help from
#         the scikit-learn community.
# License: MIT

from numpy.linalg import norm
from numpy import inner

class KernelRidgeRisk(object):

    def __init__(self, lbda):
        self.lbda = lbda

    def __call__(self, dual_coefs, ground_truth, Gram):
        pred = Gram * dual_coefs
        res = pred - ground_truth
        np = ground_truth.size
        reg = inner(dual_coefs, pred)
        return norm(res) ** 2 / (2 * np) + self.lbda * reg / (2 * np)

    def functional_grad(self, dual_coefs, ground_truth, Gram):
        pred = Gram * dual_coefs
        res = pred - ground_truth
        np = ground_truth.size
        return Gram * res / np + self.lbda * pred / np

    def functional_grad_val(self, dual_coefs, ground_truth, Gram):
        pred = Gram * dual_coefs
        res = pred - ground_truth
        np = ground_truth.size
        reg = inner(dual_coefs, pred)
        return (norm(res) ** 2 / (2 * np) + self.lbda * reg / (2 * np), Gram *
                res / np + self.lbda * pred / np)
