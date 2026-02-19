import time

import numpy as np
from numpy import percentile
from scipy.stats import skew as skew_sp

from algorithms.stat_models import column_ecdf


def skew(X, axis=0):
    return np.nan_to_num(skew_sp(X, axis=axis))


def _parallel_ecdf(n_dims, X):
    U_l_mat = np.zeros([X.shape[0], n_dims])
    U_r_mat = np.zeros([X.shape[0], n_dims])

    for i in range(n_dims):
        U_l_mat[:, i: i + 1] = column_ecdf(X[:, i: i + 1])
        U_r_mat[:, i: i + 1] = column_ecdf(X[:, i: i + 1] * -1)
    return U_l_mat, U_r_mat

class EmpiricalCumulativeDistributionFunctions:
    def getName(self):
        return f"EmpiricalCumulativeDistributionFunctions"

    def __init__(self, contamination=0.1, n_jobs=1, **kwargs):
        self.uses_gpu = False
        if (isinstance(contamination, (float, int))):

            if not (0. < contamination <= 0.5):
                raise ValueError("contamination must be in (0, 0.5], "
                                 "got: %f" % contamination)

        # allow arbitrary input such as PyThreshld object
        self.contamination = contamination

        self.n_jobs = n_jobs
        self._classes = 2  # default as binary classification

    def fit(self, X, Y=None):
        cpu_phase_1 = time.perf_counter_ns()
        self.train_gpu_time = None

        self.decision_scores_ = self.decision_function(X)
        self.X_train = X
        self._process_decision_scores()

        cpu_phase_2 = time.perf_counter_ns()
        self.train_cpu_time = cpu_phase_2 - cpu_phase_1
        self.train_all_time = self.train_cpu_time

    def decision_function(self, X, batch_size=2048):
        cpu_phase_1 = time.perf_counter_ns()
        self.eval_gpu_time = None

        if hasattr(self, 'X_train'):
            original_size = X.shape[0]
            X = np.concatenate((self.X_train, X), axis=0)
        self.U_l = -1 * np.log(column_ecdf(X))
        self.U_r = -1 * np.log(column_ecdf(-X))

        skewness = np.sign(skew(X, axis=0))
        self.U_skew = self.U_l * -1 * np.sign(
            skewness - 1) + self.U_r * np.sign(skewness + 1)

        self.O = np.maximum(self.U_l, self.U_r)
        self.O = np.maximum(self.U_skew, self.O)

        if hasattr(self, 'X_train'):
            decision_scores_ = self.O.sum(axis=1)[-original_size:]
        else:
            decision_scores_ = self.O.sum(axis=1)
        scores = decision_scores_.ravel()

        cpu_phase_2 = time.perf_counter_ns()
        self.eval_cpu_time = cpu_phase_2 - cpu_phase_1
        self.eval_all_time = self.eval_cpu_time

        return scores

    def _process_decision_scores(self):
        if isinstance(self.contamination, (float, int)):
            self.threshold_ = percentile(self.decision_scores_,
                                         100 * (1 - self.contamination))
            self.labels_ = (self.decision_scores_ > self.threshold_).astype(
                'int').ravel()

        # if this is a PyThresh object
        else:
            self.labels_ = self.contamination.eval(self.decision_scores_)
            self.threshold_ = self.contamination.thresh_
            if not self.threshold_:
                self.threshold_ = np.sum(self.labels_) / len(self.labels_)