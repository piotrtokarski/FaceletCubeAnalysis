import json
from datetime import datetime
from pathlib import Path

import numpy as np

from algorithms.CubeCNN1D import CubeCNN1D
from algorithms.CubeCNN2D2 import CubeCNN2D
from algorithms.RecurrentNet import RecurrentNet
from algorithms.SequenceRNNNet import SequenceRNNNet
from algorithms.SimpleMLPNet import SimpleMLPNet
from algorithms.SimpleTransformerNet2 import SimpleTransformerNet2
from utils.EvaluationSchema import RandomTrainTestSplit, KFoldSplit
from utils.Experiment import ExperimentConfig, Experiment

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

COMMON_RANKER = {
    "task": "binary",
    "stop_metric": "loss",
    "threshold_technique": "max_f1",  # "max_f1", "youden", "dist01", "fixed"
    "metrics_to_log": (
        "f1",
        "precision",
        "recall",
        "accuracy",
        "balanced_accuracy",
        "mcc",
        "pr_auc",  # metryka niezalezna od progu
        "roc_auc",  # metryka niezalezna od progu
    ),
    "move_mode": "18",
    "epochs": 100,
    "batch_size": 256,
    "lr": 1e-3,
    "patience": 10,
    "weight_decay": 1e-2, # kara sa duze wagi modelu
    "verbose": True,
    "val_size": 0.10,
}

COMMON_METRICS = (
    "accuracy", "f1", "precision", "recall",
    "balanced_accuracy", "mcc", "pr_auc", "roc_auc"
)

MODEL_FACTORIES = [
    ("SimpleMLPNet", lambda: SimpleMLPNet()),
    ("CubeCNN1D", lambda: CubeCNN1D()),
    ("CubeCNN2D", lambda: CubeCNN2D()),
    ("SimpleTransformerNet2", lambda: SimpleTransformerNet2()),
    ("RecurrentNet", lambda: RecurrentNet()),
    ("SequenceRNNNet", lambda: SequenceRNNNet())
]

WINDOW_SIZES = (5, 10, 15)


if __name__ == "__main__":
    all_results = {}
    experiment_counter = 0

    for window_size in WINDOW_SIZES:
        print(f"\n\n################# WINDOW_SIZE = {window_size} #################")
        all_results[f"window_size_{window_size}"] = {}

        for model_name, factory in MODEL_FACTORIES:
            experiment_counter += 1
            print(f"\n================= EXPERIMENT {experiment_counter}: {model_name}, ws={window_size} =================")

            cfg = ExperimentConfig(
                calculate_redundant_states=False,
                redundant_states_path="solves_marked.pkl",
                target_column="redundant_state_marker_binary",
                # target_column="redundant_state_marker_binary_first",
                window_size=window_size,
                stride=1,
                seed=42,
                schema=RandomTrainTestSplit(repetition=10, test_size=0.1, shuffle=True),
                # schema=KFoldSplit(k=10, shuffle=True),

                # świeży model na każdy fold
                model_factory=factory,

                ranker_params=dict(COMMON_RANKER),  # kopia na wszelki wypadek
                binary_eval_metrics=COMMON_METRICS,
            )

            try:
                result = Experiment.run(cfg)  # może być lista foldów albo dict z fold_metrics+summary
                all_results[f"window_size_{window_size}"][model_name] = result
                print(f"Wyniki ({model_name}, ws={window_size}): {result}")
            except Exception as e:
                all_results[f"window_size_{window_size}"][model_name] = {"error": str(e)}
                print(f"[BŁĄD] {model_name}, ws={window_size}: {e}")

    print("\n================= PODSUMOWANIE =================")
    for ws_key, ws_results in all_results.items():
        print(f"\n--- {ws_key} ---")
        for model_name, result in ws_results.items():
            print(f"{model_name}: {result}")

    # --- ZAPIS DO JSON ---
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target_column": "redundant_state_marker_binary",
        "window_sizes": list(WINDOW_SIZES),
        "stride": 1,
        "seed": 42,
        "schema": "RandomTrainTestSplit(repetition=10, test_size=0.1, shuffle=True)",
        "common_ranker": COMMON_RANKER,
        "common_metrics": list(COMMON_METRICS),
        "results": all_results,
    }

    out_path = Path("experiment_results_ws_5_10_15.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, ensure_ascii=False, indent=2)

    print(f"\n✅ Zapisano wyniki do pliku: {out_path.resolve()}")