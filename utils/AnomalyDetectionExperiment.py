import copy
from typing import Any, Optional, Dict, Callable

import numpy as np
import pandas as pd

from utils.BestThreshold import BestThreshold
from utils.CubeEncoder import CubeEncoder
from utils.DataLoader import DataLoader
from utils.DataTransformer import DataTransformer
from utils.Experiment import Experiment, ExperimentConfig


class AnomalyDetectionExperiment:
    @staticmethod
    def _apply_common_params_to_model(model: Any, ranker_params: Dict[str, Any]) -> None:
        """
        Opcjonalnie przenosi wspólne parametry z ranker_params do modelu,
        jeśli model posiada takie atrybuty.
        """
        for key in ("batch_size", "device", "verbose"):
            if key in ranker_params and hasattr(model, key):
                setattr(model, key, ranker_params[key])

    @staticmethod
    def _extract_features_for_anomaly(
        X_df: pd.DataFrame,
        move_mode: str = "18",
        move2id: Optional[Dict[str, int]] = None,
        one_hot_moves: bool = True,
        use_states: bool = False,
    ) -> np.ndarray:
        """
        Kodowanie zgodne z CubeEncoder (jak w CubeMovementRanker):
        - moves_ctx -> ids (lub one-hot),
        - opcjonalnie dokładamy states_ctx jako flatten.
        """
        X_states, X_moves = CubeEncoder.encode_windows_df(
            X_df,
            move_mode=move_mode,
            move2id=move2id
        )
        # X_states: (N, T, 54), X_moves: (N, W)

        X_moves = np.asarray(X_moves, dtype=np.int32)
        n = X_moves.shape[0]
        if n == 0:
            return np.empty((0, 0), dtype=np.float32)

        if one_hot_moves:
            k = int(np.max(X_moves)) + 1 if X_moves.size > 0 else 1
            w = X_moves.shape[1]
            X_m = np.zeros((n, w * k), dtype=np.float32)
            rows = np.arange(n)[:, None]
            cols = np.arange(w)[None, :]
            X_m[rows, cols * k + X_moves] = 1.0
        else:
            X_m = X_moves.astype(np.float32)

        if not use_states:
            return X_m.astype(np.float32)

        X_s = np.asarray(X_states, dtype=np.float32).reshape(n, -1)
        # lekka normalizacja stanów, jeśli kodowane np. 0..5
        if X_s.size > 0:
            mx = float(np.max(X_s))
            if mx > 0:
                X_s = X_s / mx

        return np.concatenate([X_s, X_m], axis=1).astype(np.float32)

    @staticmethod
    def _pick_threshold(
        y_true: np.ndarray,
        y_score: np.ndarray,
        threshold_technique: str = "max_f1",
        default_threshold: float = 0.5,
    ) -> float:
        threshold_technique = str(threshold_technique).lower().strip()

        if threshold_technique == "fixed":
            return float(default_threshold)

        if y_true is None or np.unique(y_true).size < 2:
            return float(default_threshold)

        thr, _ = BestThreshold._best_threshold(
            y_true=y_true,
            y_score=y_score,
            technique=threshold_technique
        )
        return float(thr)

    @staticmethod
    def evaluate_with_schema_deepif(
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
        label_cols = [
            "redundant_state_marker",
            "redundant_state_marker_binary",
            "redundant_state_marker_first",
            "redundant_state_marker_binary_first",
        ]
        meta_cols = ["solve_id", "start", "end"]

        # Ustawienia w tym samym stylu co Experiment / CubeMovementRanker
        ranker_params = ranker_params or {}
        task = str(ranker_params.get("task", "binary")).lower()
        threshold_technique = str(ranker_params.get("threshold_technique", "max_f1")).lower()
        move_mode = str(ranker_params.get("move_mode", "18"))
        verbose = bool(ranker_params.get("verbose", True))

        # dodatkowe (opcjonalne) ustawienia cech
        one_hot_moves = bool(ranker_params.get("one_hot_moves", True))
        use_states = bool(ranker_params.get("use_states", False))
        default_threshold = float(ranker_params.get("threshold", 0.5))

        if task != "binary":
            raise ValueError("evaluate_with_schema_deepif aktualnie obsługuje task='binary'.")

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
            X_train_df = train_windows.drop(columns=[c for c in label_cols if c in train_windows.columns])
            X_train_df = X_train_df.drop(columns=[c for c in meta_cols if c in X_train_df.columns])

            y_test = test_windows[target_column].to_numpy(dtype=np.int32)
            X_test_df = test_windows.drop(columns=[c for c in label_cols if c in test_windows.columns])
            X_test_df = X_test_df.drop(columns=[c for c in meta_cols if c in X_test_df.columns])

            # Spójnie z CubeMovementRanker: vocab budowany na TRAIN
            if move_mode == "vocab":
                move2id = CubeEncoder.build_move_vocab_from_X(X_train_df)
            else:
                move2id = None

            X_train = AnomalyDetectionExperiment._extract_features_for_anomaly(
                X_train_df,
                move_mode=move_mode,
                move2id=move2id,
                one_hot_moves=one_hot_moves,
                use_states=use_states,
            )
            X_test = AnomalyDetectionExperiment._extract_features_for_anomaly(
                X_test_df,
                move_mode=move_mode,
                move2id=move2id,
                one_hot_moves=one_hot_moves,
                use_states=use_states,
            )

            # świeży model na fold
            fold_model = model_factory() if model_factory is not None else copy.deepcopy(model)
            AnomalyDetectionExperiment._apply_common_params_to_model(fold_model, ranker_params)

            # DeepIF-like flow: fit + decision_function
            fold_model.fit(X_train)
            y_score_train = np.asarray(fold_model.decision_function(X_train), dtype=np.float32)
            y_score_test = np.asarray(fold_model.decision_function(X_test), dtype=np.float32)
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # odwrocenie funkcji decyzyjnej bo to jest nienadzorowana detekcja wiec te nietypowe ruchy beda bardziej geste bo sie powtarzaja
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            y_score_train = -y_score_train
            y_score_test = -y_score_test

            best_threshold = AnomalyDetectionExperiment._pick_threshold(
                y_true=y_train,
                y_score=y_score_train,
                threshold_technique=threshold_technique,
                default_threshold=default_threshold,
            )

            m = Experiment._compute_binary_metrics(
                y_true=y_test,
                y_score=y_score_test,
                threshold=best_threshold,
                metrics=binary_eval_metrics
            )

            row = {"fold": i, "task": "binary", "threshold": float(best_threshold)}
            row.update(m)
            fold_metrics.append(row)

            if verbose:
                pretty = ", ".join([f"{k}={v:.4f}" if np.isfinite(v) else f"{k}=nan" for k, v in m.items()])
                print(f"Fold {i}: thr={best_threshold:.4f}, {pretty}")

        summary = Experiment._build_summary(
            fold_metrics=fold_metrics,
            task="binary",
            binary_eval_metrics=binary_eval_metrics
        )

        print("\n=== Summary ===")
        if summary["n_folds"] == 0:
            print("Brak metryk (wszystkie foldy pominięte).")
            return {"fold_metrics": fold_metrics, "summary": summary}

        sthr = summary.get("threshold", {})
        if sthr:
            print(
                f"{'threshold':<16}: mean={sthr['mean']:.4f}, std={sthr['std']:.4f}, "
                f"min={sthr['min']:.4f}, max={sthr['max']:.4f}"
            )
        for name, s in summary["metrics"].items():
            print(
                f"{name:<16}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
                f"min={s['min']:.4f}, max={s['max']:.4f}"
            )

        return {
            "fold_metrics": fold_metrics,
            "summary": summary
        }

    @staticmethod
    def run_deepif(cfg: ExperimentConfig):
        if cfg.calculate_redundant_states:
            df = DataLoader.load_data(cfg.dataset_path)
            df_transformer = DataTransformer.prepare(df, verbose=True)
            DataLoader.save_prepared_df(df_prepared=df_transformer, path=cfg.redundant_states_path)
        else:
            df_transformer = DataLoader.load_prepared_df(path=cfg.redundant_states_path)

        return AnomalyDetectionExperiment.evaluate_with_schema_deepif(
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