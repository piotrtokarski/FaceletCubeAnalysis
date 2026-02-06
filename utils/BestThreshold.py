import numpy as np
from sklearn.metrics import f1_score


class BestThreshold:
    @staticmethod
    def _best_threshold_f1(y_true, y_score, average="binary",max_candidates = 1024):
        max_score = np.max(y_score)
        min_score = np.min(y_score)

        thresholds = np.linspace(min_score, max_score, max_candidates)

        best_threshold = thresholds[0]
        best_f1 = -1.0

        for t in thresholds:
            y_pred = (y_score >= t).astype(np.int32)
            f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
            if f1 > best_f1:
                best_f1 = float(f1)
                best_threshold = float(t)

        return best_threshold, best_f1