import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Callable

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, average_precision_score, roc_auc_score,
    matthews_corrcoef, mean_absolute_error, mean_squared_error, r2_score
)

from algorithms.CubeMovementRanker import CubeMovementRanker
from utils.BestThreshold import BestThreshold
from utils.DataLoader import DataLoader
from utils.DataTransformer import DataTransformer


@dataclass
class ExperimentConfig:
    # data
    calculate_redundant_states: bool = False
    dataset_path: str = os.path.join("datasets", "solves.csv")
    redundant_states_path: str = "solves_marked.pkl"

    # experiment
    target_column: str = "redundant_state_marker_binary"
    window_size: int = 10
    stride: int = 1
    seed: Optional[int] = 42

    # required objects
    schema: Any = None
    model: Any = None                      # fallback (gdy nie podasz model_factory)
    model_factory: Optional[Callable[[], Any]] = None  # preferowane

    # ranker params
    ranker_params: Dict[str, Any] = field(default_factory=lambda: {
        "task": "binary",
        "stop_metric": "loss",
        "threshold_technique": "max_f1",
        "move_mode": "18",
        "epochs": 300,
        "batch_size": 256,
        "lr": 1e-3,
        "verbose": True,
        "val_size": 0.15
    })

    # które metryki policzyć i podsumować na TEST (binary)
    binary_eval_metrics: tuple[str, ...] = (
        "accuracy", "f1", "precision", "recall",
        "balanced_accuracy", "mcc", "pr_auc", "roc_auc"
    )

class Experiment:

    @staticmethod
    def solves_to_windows(df_solves: pd.DataFrame, window_size: int = 10, stride: int = 1) -> pd.DataFrame:
        rows = []

        for r in df_solves.itertuples(index=False):
            solve_id = r.id
            states = list(r.state_list)
            moves = list(r.moves_list)
            red_state = list(r.redundant_states)
            red_state_bin = list(r.redundant_states_binary)
            red_state_first = list(r.redundant_states_first)
            red_state_bin_first = list(r.redundant_states_binary_first)

            n = len(moves)
            if len(states) != n + 1 or len(red_state_bin) != n + 1 or len(red_state) != n + 1 or len(red_state_bin_first) != n + 1 or len(red_state_first) != n + 1:
                print(f"id={solve_id}: something is wrong with states number or with moves number!")
                continue

            start_index = 0
            while start_index <= n - window_size:
                end_index = start_index + window_size - 1

                states_ctx = states[start_index:start_index + window_size + 1]  # W+1 stanów
                moves_ctx = moves[start_index:start_index + window_size]         # W ruchów

                rows.append({
                    "solve_id": solve_id,
                    "start": start_index,
                    "end": end_index,
                    "states_ctx": states_ctx,
                    "moves_ctx": moves_ctx,
                    "redundant_state_marker": int(red_state[end_index + 1]),
                    "redundant_state_marker_binary": int(red_state_bin[end_index + 1]),
                    "redundant_state_marker_first": int(red_state_first[end_index + 1]),
                    "redundant_state_marker_binary_first": int(red_state_bin_first[end_index + 1]),
                })

                start_index += stride

        return pd.DataFrame(rows)

    @staticmethod
    def evaluate_with_schema(
        model: Any,
        df_solves: pd.DataFrame,
        target_column: str,
        schema: Any,
        window_size: int = 10,
        stride: int = 1,
        seed: Optional[int] = None,
        ranker_params: Optional[Dict[str, Any]] = None,
        model_factory: Optional[Callable[[], Any]] = None,
        binary_eval_metrics: tuple[str, ...] = ("accuracy", "f1"),
    ) -> List[Dict[str, float]]:
        label_cols = ["redundant_state_marker", "redundant_state_marker_binary", "redundant_state_marker_first", "redundant_state_marker_binary_first"]
        meta_cols = ["solve_id", "start", "end"]

        ranker_params = ranker_params or {}
        task = str(ranker_params.get("task", "binary")).lower()

        rng = np.random.default_rng(seed)
        folds = schema.split(df_solves, y=None, rng=rng)

        fold_metrics = []

        for i, (train_idx, test_idx) in enumerate(folds, start=1):
            train_solves = df_solves.iloc[train_idx].reset_index(drop=True)
            test_solves = df_solves.iloc[test_idx].reset_index(drop=True)

            train_windows = Experiment.solves_to_windows(train_solves, window_size=window_size, stride=stride)
            test_windows = Experiment.solves_to_windows(test_solves, window_size=window_size, stride=stride)

            if len(train_windows) == 0 or len(test_windows) == 0:
                print(f"Fold {i}: skipped (no windows) train={len(train_windows)} test={len(test_windows)}")
                continue

            y_train = train_windows[target_column].to_numpy(dtype=np.int32)
            X_train = train_windows.drop(columns=[c for c in label_cols if c in train_windows.columns])
            X_train = X_train.drop(columns=[c for c in meta_cols if c in X_train.columns])

            y_test = test_windows[target_column].to_numpy(dtype=np.int32)
            X_test = test_windows.drop(columns=[c for c in label_cols if c in test_windows.columns])
            X_test = X_test.drop(columns=[c for c in meta_cols if c in X_test.columns])

            # świeży model na fold
            if model_factory is not None:
                fold_model = model_factory()
            else:
                # fallback (głębsza kopia)
                fold_model = copy.deepcopy(model)

            ranker = CubeMovementRanker(
                model=fold_model,
                **ranker_params
            )

            ranker.train(X_train, y_train, X_test_wyciek=X_test, y_test_wyciek=y_test)

            y_score_test = ranker.score_samples(X_test)

            if task == "binary":
                best_threshold = float(ranker.threshold)  # próg po walidacji z rankera
                m = Experiment._compute_binary_metrics(
                    y_true=y_test,
                    y_score=y_score_test,
                    threshold=best_threshold,
                    metrics=binary_eval_metrics
                )

                row = {"fold": i, "task": "binary", "threshold": best_threshold}
                row.update(m)
                fold_metrics.append(row)

                pretty = ", ".join([f"{k}={v:.4f}" if np.isfinite(v) else f"{k}=nan" for k, v in m.items()])
                print(f"Fold {i}: thr={best_threshold:.4f}, {pretty}")

            elif task == "regression":
                mae = mean_absolute_error(y_test, y_score_test)
                rmse = float(np.sqrt(mean_squared_error(y_test, y_score_test)))
                r2 = r2_score(y_test, y_score_test)

                fold_metrics.append({
                    "fold": i, "task": "regression",
                    "mae": float(mae), "rmse": float(rmse), "r2": float(r2),
                })
                print(f"Fold {i}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
            else:
                raise ValueError(f"Unsupported task='{task}'. Use 'binary' or 'regression'.")

        # ===== summary =====
        print("\n=== Summary ===")
        if not fold_metrics:
            print("Brak metryk (wszystkie foldy pominięte).")
            return fold_metrics

        if task == "binary":
            metric_names = [m for m in binary_eval_metrics if m in fold_metrics[0]]
            for name in metric_names:
                vals = np.array([fm[name] for fm in fold_metrics], dtype=np.float64)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    print(f"{name:<16}: mean=nan, std=nan")
                else:
                    print(f"{name:<16}: mean={vals.mean():.4f}, std={vals.std(ddof=0):.4f}")
        else:
            maes = [m["mae"] for m in fold_metrics]
            rmses = [m["rmse"] for m in fold_metrics]
            r2s = [m["r2"] for m in fold_metrics]
            print(f"MAE : mean={np.mean(maes):.4f}, std={np.std(maes, ddof=0):.4f}")
            print(f"RMSE: mean={np.mean(rmses):.4f}, std={np.std(rmses, ddof=0):.4f}")
            print(f"R2  : mean={np.mean(r2s):.4f}, std={np.std(r2s, ddof=0):.4f}")

        return fold_metrics

    @staticmethod
    def _compute_binary_metrics(y_true, y_score, threshold, metrics):
        y_true = np.asarray(y_true).astype(np.int32).reshape(-1)
        y_score = np.asarray(y_score).astype(np.float32).reshape(-1)
        y_pred = (y_score >= float(threshold)).astype(np.int32)

        uniq = np.unique(y_true)
        has_both_classes = (uniq.size == 2)

        out = {}
        for m in metrics:
            m = m.lower().strip()
            if m == "accuracy":
                out[m] = float(accuracy_score(y_true, y_pred))
            elif m == "f1":
                out[m] = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
            elif m == "precision":
                out[m] = float(precision_score(y_true, y_pred, zero_division=0))
            elif m == "recall":
                out[m] = float(recall_score(y_true, y_pred, zero_division=0))
            elif m == "balanced_accuracy":
                out[m] = float(balanced_accuracy_score(y_true, y_pred))
            elif m == "mcc":
                out[m] = float(matthews_corrcoef(y_true, y_pred))
            elif m == "pr_auc":
                out[m] = float(average_precision_score(y_true, y_score)) if has_both_classes else float("nan")
            elif m == "roc_auc":
                out[m] = float(roc_auc_score(y_true, y_score)) if has_both_classes else float("nan")
            else:
                raise ValueError(f"Unknown binary metric: {m}")
        return out

    @staticmethod
    def run(cfg: ExperimentConfig) -> List[Dict[str, float]]:
        if cfg.calculate_redundant_states:
            df = DataLoader.load_data(cfg.dataset_path)
            df_transformer = DataTransformer.prepare(df, verbose=True)
            DataLoader.save_prepared_df(df_prepared=df_transformer, path=cfg.redundant_states_path)
        else:
            df_transformer = DataLoader.load_prepared_df(path=cfg.redundant_states_path)

        return Experiment.evaluate_with_schema(
            model=cfg.model,
            model_factory=cfg.model_factory,
            df_solves=df_transformer,
            target_column=cfg.target_column,
            schema=cfg.schema,
            window_size=cfg.window_size,
            stride=cfg.stride,
            seed=cfg.seed,
            ranker_params=cfg.ranker_params,
            binary_eval_metrics=cfg.binary_eval_metrics,
        )
