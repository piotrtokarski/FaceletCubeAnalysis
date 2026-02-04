import os

import numpy as np
import pycuber
from sklearn.metrics import accuracy_score, f1_score

from algorithms.CubeMovementRanker import CubeMovementRanker
from algorithms.SimpleMLPNet import SimpleMLPNet
from utils.DataLoader import DataLoader
from utils.DataTransformer import DataTransformer
from utils.EvaluationSchema import KFoldSplit, RandomTrainTestSplit

import pandas as pd

# stride = 1 -> sliding window - okna sa tworzone z przesunieciem o jeden
def solves_to_windows(df_solves, window_size = 10, stride = 1):
    rows = []

    for r in df_solves.itertuples(index=False):
        solve_id = r.id
        states = list(r.state_list)
        moves  = list(r.moves_list)
        red_state = list(r.redundant_states)
        red_state_bin = list(r.redundant_states_binary)

        n = len(moves)
        if len(states) != n + 1 or len(red_state_bin) != n + 1 or len(red_state) != n + 1:
            print(f"id={solve_id}: something is wrong with states number or with moves number!")
            continue

        start_index = 0
        while start_index < n - window_size + 1:
            end_index = start_index + window_size - 1

            states_ctx = states[start_index: start_index + window_size + 1]  # W+1 stanów
            moves_ctx = moves[start_index: start_index + window_size]  # W ruchów

            rows.append({
                "solve_id": solve_id,
                "start": start_index,
                "end": end_index,
                "states_ctx": states_ctx,
                "moves_ctx": moves_ctx,
                "redundant_state_marker": int(red_state[end_index + 1]),
                "redundant_state_marker_binary": int(red_state_bin[end_index + 1])
            })

            start_index += stride

    return pd.DataFrame(rows)

def evaluate_with_schema(df_solves, target_column, schema, window_size=10, stride=1, seed=None, average="macro"):
    label_cols = ["redundant_state_marker", "redundant_state_marker_binary"]
    meta_cols = ["solve_id", "start", "end"]
    rng = np.random.default_rng(seed)

    folds = schema.split(df_solves, y=None, rng=rng)
    fold_metrics = []
    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        train_solves = df_solves.iloc[train_idx].reset_index(drop=True)
        test_solves  = df_solves.iloc[test_idx].reset_index(drop=True)

        train_windows = solves_to_windows(train_solves, window_size=window_size, stride=stride)
        test_windows  = solves_to_windows(test_solves,  window_size=window_size, stride=stride)

        if len(train_windows) == 0 or len(test_windows) == 0:
            print(f"Fold {i}: skipped (no windows) train={len(train_windows)} test={len(test_windows)}")
            continue

        y_train = train_windows[target_column].to_numpy(dtype=np.int32)
        X_train = train_windows.drop(columns=[c for c in label_cols if c in train_windows.columns])
        X_train = X_train.drop(columns=[c for c in meta_cols if c in X_train.columns])

        y_test = test_windows[target_column].to_numpy(dtype=np.int32)
        X_test = test_windows.drop(columns=[c for c in label_cols if c in test_windows.columns])
        X_test = X_test.drop(columns=[c for c in meta_cols if c in X_test.columns])

        model = SimpleMLPNet(move_vocab_size=18)

        ranker = CubeMovementRanker(
            model=model,
            task="binary",
            move_mode="18",
            epochs=300,
            batch_size=256,
            lr=1e-3,
            threshold=0.5,
            verbose=True,
            f1_average=average
        )
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
    calculate_redundant_states = False
    redundant_states_path = "solves_marked.pkl"

    if calculate_redundant_states:
        path = os.path.join("datasets", "solves.csv")
        df = DataLoader.load_data(path)
        df_transformer = DataTransformer.prepare(df,verbose=True)
        DataLoader.save_prepared_df(df_prepared=df_transformer, path=redundant_states_path)
    else:
        df_transformer = DataLoader.load_prepared_df(path=redundant_states_path)

    schema = KFoldSplit(k=10, shuffle=True)
    schema = RandomTrainTestSplit(test_size=0.1, shuffle=True)
    evaluate_with_schema(
        df_transformer,
        target_column="redundant_state_marker_binary",
        schema=schema,
        window_size=10,
        average="binary",
        seed=42
    )
