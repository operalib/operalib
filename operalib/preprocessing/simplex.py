"""Simplex coding module."""

from numpy import dot, array, vstack, hstack, ones, zeros, sqrt, asarray

from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

# pylint: disable=W0201,C0103


class SimplexCoding(BaseEstimator, TransformerMixin):
    """Simplex coding."""

    def __init__(self, binarizer=None):
        self.binarizer = binarizer

        # self.simplex_operator_ = None
        # self.binarizer_ = None

    @staticmethod
    def _code_i(dimension):
        """Simplex coding operator (internal).

        https://papers.nips.cc/paper/4764-multiclass-learning-with-simplex-
        coding.pdf
        """
        if dimension > 1:
            block1 = vstack((ones((1, 1)), zeros((dimension - 1, 1))))
            block2 = vstack((-ones((1, dimension)) / dimension,
                             SimplexCoding._code_i(dimension - 1) *
                             sqrt(1. - 1. / (dimension * dimension))))
            return hstack((block1, block2))
        elif dimension == 1:
            return array([1., -1.])
        else:
            raise ValueError('dimension should be at least one.')

    @staticmethod
    def code(dimension):
        """Simplex coding operator."""
        return SimplexCoding._code_i(dimension - 1)

    def fit(self, y):
        """Fit simplex coding

        Parameters
        ----------
        targets : array, shape = [n_samples,] or [n_samples, n_classes]
            Target values. The 2-d array represents the simplex coding for
            multilabel classification.

        Returns
        -------
        self : returns an instance of self.
        """
        if self.binarizer is None:
            self.binarizer_ = LabelBinarizer(neg_label=0, pos_label=1,
                                             sparse_output=True)
        self.binarizer_.fit(y)
        dimension = self.binarizer_.classes_.size
        if dimension > 2:
            self.simplex_operator_ = SimplexCoding.code(dimension)
        else:
            self.simplex_operator_ = ones((1, 1))
        return self

    def transform(self, y):
        """Transform multi-class labels to the simplex code.

        Parameters
        ----------
        targets : array or sparse matrix, shape = [n_samples,] or
                  [n_samples, n_classes]
            Target values. The 2-d matrix represents the simplex code for
            multilabel classification.

        Returns
        -------
        Y : numpy array of shape [n_samples, n_classes - 1]
        """
        check_is_fitted(self, 'simplex_operator_', 'binarizer_')
        dimension = self.binarizer_.classes_.size
        if dimension == 2:
            return self.binarizer_.transform(y).toarray()
        else:
            return self.binarizer_.transform(y).dot(
                asarray(self.simplex_operator_).T)

    def inverse_transform(self, y):
        """Inverse transform."""
        check_is_fitted(self, 'simplex_operator_', 'binarizer_')
        dimension = self.binarizer_.classes_.size
        if dimension == 2:
            return self.binarizer_.inverse_transform(y)
        else:
            return self.binarizer_.inverse_transform(
                dot(y, self.simplex_operator_))
