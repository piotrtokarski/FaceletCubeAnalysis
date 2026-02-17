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
class UncertaintyExperimentConfig:
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

class UncertaintyExperiment:

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
    def evaluate_and_plot_with_schema(
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
    ):
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

            train_windows = UncertaintyExperiment.solves_to_windows(train_solves, window_size=window_size, stride=stride)
            test_windows = UncertaintyExperiment.solves_to_windows(test_solves, window_size=window_size, stride=stride)

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
            # ranker.predict_with_uncertainty(X_test)
            ranker.plot_three_sets(output_file_name=f"res/ws_{window_size}_fold_{i}.png",scores=y_score_test,show_samples=False)
            ranker.plot_three_density_stacked(f"res/ws_{window_size}_fold_{i}_density.png", scores=y_score_test, bins=50, kde=True)
            ranker.plot_component_densities_over_score(
                output_file_name=f"res/ws_{window_size}_fold_{i}_density_score.png",
                scores=y_score_test,
                bins=50,
                kde=True,
                stacked=True
            )


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
    def run(cfg: UncertaintyExperimentConfig):
        if cfg.calculate_redundant_states:
            df = DataLoader.load_data(cfg.dataset_path)
            df_transformer = DataTransformer.prepare(df, verbose=True)
            DataLoader.save_prepared_df(df_prepared=df_transformer, path=cfg.redundant_states_path)
        else:
            df_transformer = DataLoader.load_prepared_df(path=cfg.redundant_states_path)

        return UncertaintyExperiment.evaluate_and_plot_with_schema(
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

    @staticmethod
    def _stats_from_values(values: np.ndarray) -> Dict[str, float]:
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]

        if finite.size == 0:
            return {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "median": float("nan"),
            }

        return {
            "count": int(finite.size),
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite, ddof=0)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
            "median": float(np.median(finite)),
        }

    @staticmethod
    def _build_summary(
        fold_metrics: List[Dict[str, float]],
        task: str,
        binary_eval_metrics: tuple[str, ...]
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "task": task,
            "n_folds": int(len(fold_metrics)),
            "metrics": {}
        }

        if len(fold_metrics) == 0:
            return summary

        if task == "binary":
            # agregacja progu
            thr_vals = np.array([fm.get("threshold", np.nan) for fm in fold_metrics], dtype=np.float64)
            summary["threshold"] = UncertaintyExperiment._stats_from_values(thr_vals)

            # agregacja metryk binarnych
            metric_names = [m for m in binary_eval_metrics if any(m in fm for fm in fold_metrics)]
            for name in metric_names:
                vals = np.array([fm.get(name, np.nan) for fm in fold_metrics], dtype=np.float64)
                summary["metrics"][name] = UncertaintyExperiment._stats_from_values(vals)

        elif task == "regression":
            for name in ("mae", "rmse", "r2"):
                vals = np.array([fm.get(name, np.nan) for fm in fold_metrics], dtype=np.float64)
                summary["metrics"][name] = UncertaintyExperiment._stats_from_values(vals)

        else:
            raise ValueError(f"Unsupported task='{task}'. Use 'binary' or 'regression'.")

        return summary