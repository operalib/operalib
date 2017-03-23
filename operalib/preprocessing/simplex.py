"""Simplex coding module."""

from numpy import dot, array, vstack, hstack, ones, zeros, sqrt
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class SimplexCoding(BaseEstimator, TransformerMixin):
    """Simplex coding."""

    def __init__(self, binarizer=LabelBinarizer(neg_label=0, pos_label=1,
                                                sparse_output=True)):
        self.binarizer = binarizer

    @staticmethod
    def _code_i(dimension):
        """Simplex coding operator (internal).

        https://papers.nips.cc/paper/4764-multiclass-learning-with-simplex-coding.pdf
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

    def fit(self, targets):
        """Fit simplex coding

        Parameters
        ----------
        targets : numpy array of shape (n_samples,) or
                  (n_samples, n_classes) Target values. The 2-d array
                  represents the simplex coding for multilabel classification.

        Returns
        -------
        self : returns an instance of self.
        """

        self.binarizer.fit(targets)
        dimension = self.binarizer.classes_.size
        self.simplex_operator_ = SimplexCoding.code(dimension)
        return self

    def transform(self, targets):
        """Transform multi-class labels to the simplex code.

        Parameters
        ----------
        targets : numpy array or sparse matrix of shape (n_samples,) or
                  (n_samples, n_classes) Target values. The 2-d matrix
                  represents the simplex code for multilabel classification.

        Returns
        -------
        Y : numpy array of shape [n_samples, n_classes - 1]
        """
        check_is_fitted(self, 'simplex_operator_')
        return self.binarizer.transform(targets).dot(self.simplex_operator_.T)

    def inverse_transform(self, targets):
        check_is_fitted(self, 'simplex_operator_')
        return dot(targets, self.simplex_operator_).argmax(axis=1)
