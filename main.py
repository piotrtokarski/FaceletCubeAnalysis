import os

import numpy as np
import pycuber
from sklearn.metrics import accuracy_score, f1_score

from algorithms.CubeMovementRanker import CubeMovementRanker
from utils.DataLoader import DataLoader
from utils.DataTransformer import DataTransformer
from utils.EvaluationSchema import KFoldSplit


def evaluate_with_schema(df, target_column, schema, seed=None, average="macro"):
    rng = np.random.default_rng(seed)
    X = df
    y = df[target_column]

    folds = schema.split(X, y=None, rng=rng)

    fold_metrics = []
    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        train_df = df.iloc[train_idx]
        y_train = train_df[target_column]
        test_df = df.iloc[test_idx]
        y_test = test_df[target_column]

        X_train = train_df
        X_test = test_df

        ranker = CubeMovementRanker()
        ranker.train(X_train, y_train)
        y_pred = ranker.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=average)

        fold_metrics.append({"fold": i, "accuracy": acc, "f1": f1})

        print(f"Fold {i}: accuracy={acc:.4f}, f1={f1:.4f}")

    accs = [m["accuracy"] for m in fold_metrics]
    f1s = [m["f1"] for m in fold_metrics]

    print("\n=== Summary ===")
    print(f"accuracy: mean={np.mean(accs):.4f}, std={np.std(accs, ddof=0):.4f}")
    print(f"f1      : mean={np.mean(f1s):.4f}, std={np.std(f1s, ddof=0):.4f}")

    return fold_metrics

if __name__ == "__main__":
    path = os.path.join("datasets", "solves.csv")

    df = DataLoader.load_data(path)
    schema = KFoldSplit(k=5, shuffle=True)

    df_transformer = DataTransformer.prepare(df)



    # evaluate_with_schema(df, target_column="target", schema=schema)