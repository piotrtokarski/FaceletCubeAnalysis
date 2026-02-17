import json
from datetime import datetime
from pathlib import Path

import numpy as np

from algorithms.CubeCNN1D import CubeCNN1D
from algorithms.CubeCNN2D2 import CubeCNN2D
from algorithms.IF import IsolationForest
from algorithms.ODIF import DeepIF
from algorithms.RecurrentNet import RecurrentNet
from algorithms.SequenceRNNNet import SequenceRNNNet
from algorithms.SimpleMLPNet import SimpleMLPNet
from algorithms.SimpleTransformerNet2 import SimpleTransformerNet2
from utils.AnomalyDetectionExperiment import AnomalyDetectionExperiment
from utils.EvaluationSchema import RandomTrainTestSplit, KFoldSplit
from utils.Experiment import ExperimentConfig


def to_serializable(obj):
    """Rekurencyjnie zamienia obiekty na typy serializowalne do JSON."""
    if np is not None:
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()

    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_serializable(x) for x in obj]

    # fallback dla nietypowych obiektów
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


COMMON_PARAMS = {
    "task": "binary",
    "threshold_technique": "max_f1",
    "move_mode": "18",  # albo "vocab"
    "one_hot_moves": False,  # polecam dla IF
    "use_states": True,  # True jeśli chcesz dorzucić stany
}

COMMON_METRICS = (
    "accuracy", "f1", "precision", "recall",
    "balanced_accuracy", "mcc", "pr_auc", "roc_auc"
)

MODEL_FACTORIES = [
    ("IsolationForest", lambda: IsolationForest()),
    # ("OptimizedDeepIF", lambda: DeepIF())
]

WINDOW_SIZES = (5, 10, 15)

if __name__ == "__main__":
    all_results = {}
    exp_idx = 0

    for ws in WINDOW_SIZES:
        ws_key = f"window_size_{ws}"
        all_results[ws_key] = {}
        print(f"\n\n################# WINDOW_SIZE = {ws} #################")

        for model_name, factory in MODEL_FACTORIES:
            exp_idx += 1
            print(f"\n================= EXPERIMENT {exp_idx}: {model_name}, ws={ws} =================")

            cfg = ExperimentConfig(
                calculate_redundant_states=False,
                redundant_states_path="solves_marked.pkl",
                target_column="redundant_state_marker_binary",
                window_size=ws,
                stride=1,
                seed=42,
                schema=RandomTrainTestSplit(repetition=10, test_size=0.1, shuffle=True),
                model_factory=factory,
                ranker_params=dict(COMMON_PARAMS),
                binary_eval_metrics=COMMON_METRICS,
            )

            try:
                result = AnomalyDetectionExperiment.run_deepif(cfg)  # <-- NOWY TRYB
                all_results[ws_key][model_name] = result
                print(f"Wyniki ({model_name}, ws={ws}): {result}")
            except Exception as e:
                all_results[ws_key][model_name] = {"error": str(e)}
                print(f"[BŁĄD] {model_name}, ws={ws}: {e}")

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "experiment_type": "DeepIF_direct_fit",
        "target_column": "redundant_state_marker_binary",
        "window_sizes": list(WINDOW_SIZES),
        "stride": 1,
        "seed": 42,
        "schema": "RandomTrainTestSplit(repetition=10, test_size=0.1, shuffle=True)",
        "params": COMMON_PARAMS,
        "metrics": list(COMMON_METRICS),
        "results": all_results,
    }

    out_path = Path("experiment_deepif_ws_5_10_15.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, ensure_ascii=False, indent=2)

    print(f"\n✅ Zapisano wyniki do pliku: {out_path.resolve()}")
