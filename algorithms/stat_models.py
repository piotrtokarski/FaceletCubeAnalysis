# -*- coding: utf-8 -*-
""" A collection of statistical models
"""
# Author: Yue Zhao <zhaoy@cmu.edu>
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

import numpy as np
from numba import njit
from scipy.stats import pearsonr
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_consistent_length


def pairwise_distances_no_broadcast(X, Y):
    X = check_array(X)
    Y = check_array(Y)

    if X.shape[0] != Y.shape[0] or X.shape[1] != Y.shape[1]:
        raise ValueError("pairwise_distances_no_broadcast function receive"
                         "matrix with different shapes {0} and {1}".format(
            X.shape, Y.shape))
    return _pairwise_distances_no_broadcast_helper(X, Y)


@njit
def _pairwise_distances_no_broadcast_helper(X, Y):  # pragma: no cover
    euclidean_sq = np.square(Y - X)
    return np.sqrt(np.sum(euclidean_sq, axis=1)).ravel()


def wpearsonr(x, y, w=None):
    if w is None:
        return pearsonr(x, y)

    x = np.asarray(x)
    y = np.asarray(y)
    w = np.asarray(w)

    check_consistent_length([x, y, w])

    w_sum = w.sum()
    mx = np.sum(x * w) / w_sum
    my = np.sum(y * w) / w_sum

    xm, ym = (x - mx), (y - my)

    r_num = np.sum(xm * ym * w) / w_sum

    xm2 = np.sum(xm * xm * w) / w_sum
    ym2 = np.sum(ym * ym * w) / w_sum

    r_den = np.sqrt(xm2 * ym2)
    r = r_num / r_den

    r = max(min(r, 1.0), -1.0)
    return r

def pearsonr_mat(mat, w=None):
    mat = check_array(mat)
    n_row = mat.shape[0]
    n_col = mat.shape[1]
    pear_mat = np.full([n_row, n_row], 1).astype(float)

    if w is not None:
        for cx in range(n_row):
            for cy in range(cx + 1, n_row):
                curr_pear = wpearsonr(mat[cx, :], mat[cy, :], w)
                pear_mat[cx, cy] = curr_pear
                pear_mat[cy, cx] = curr_pear
    else:
        for cx in range(n_col):
            for cy in range(cx + 1, n_row):
                curr_pear = pearsonr(mat[cx, :], mat[cy, :])[0]
                pear_mat[cx, cy] = curr_pear
                pear_mat[cy, cx] = curr_pear

    return pear_mat


def column_ecdf(matrix: np.ndarray) -> np.ndarray:
    assert len(matrix.shape) == 2, 'Matrix needs to be two dimensional for the ECDF computation.'
    probabilities = np.linspace(np.ones(matrix.shape[1]) / matrix.shape[0], np.ones(matrix.shape[1]), matrix.shape[0])
    sort_idx = np.argsort(matrix, axis=0)
    matrix = np.take_along_axis(matrix, sort_idx, axis=0)
    ecdf_terminate_equals_inplace(matrix, probabilities)
    reordered_probabilities = np.ones_like(probabilities)
    np.put_along_axis(reordered_probabilities, sort_idx, probabilities, axis=0)
    return reordered_probabilities


@njit
def ecdf_terminate_equals_inplace(matrix: np.ndarray, probabilities: np.ndarray):
    for cx in range(probabilities.shape[1]):
        for rx in range(probabilities.shape[0] - 2, -1, -1):
            if matrix[rx, cx] == matrix[rx + 1, cx]:
                probabilities[rx, cx] = probabilities[rx + 1, cx]
