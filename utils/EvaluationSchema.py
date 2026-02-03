import numpy as np


class EvaluationSchema:
    def getName(self):
        raise NotImplementedError

    def split(self, X, y=None, rng=None):
        """
        Zwraca listę foldów: [(train_idx, test_idx), ...]
        """
        raise NotImplementedError

class RandomTrainTestSplit(EvaluationSchema):
    def __init__(self, test_size=0.2, shuffle=True):
        self.test_size = test_size
        self.shuffle = shuffle

    def getName(self):
        return f"random_ts_{self.test_size}"

    def split(self, X, y=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        n = len(X)
        idx = np.arange(n)
        folds = []

        ts = float(self.test_size)
        n_test = max(1, int(round(n * ts)))

        perm = idx.copy()
        if self.shuffle:
            rng.shuffle(perm)

        test_idx = perm[:n_test]
        train_idx = perm[n_test:]
        folds.append((train_idx, test_idx))

        return folds

class KFoldSplit(EvaluationSchema):
    def __init__(self, k=5, shuffle=True):
        self.k = k
        self.shuffle = shuffle

    def getName(self):
        return f"kfold_{self.k}"

    def split(self, X, y=None, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        n = len(X)
        base = np.arange(n)
        out = []

        k = int(self.k)
        k = max(2, min(k, n))

        idx = base.copy()
        if self.shuffle:
            rng.shuffle(idx)

        fold_sizes = np.full(k, n // k, dtype=int)
        fold_sizes[: n % k] += 1

        start = 0
        for fs in fold_sizes:
            test_idx = idx[start:start + fs]
            train_idx = np.concatenate([idx[:start], idx[start + fs:]])
            out.append((train_idx, test_idx))
            start += fs

        return out