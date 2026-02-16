from algorithms.SimpleMLPNet import SimpleMLPNet
from utils.EvaluationSchema import RandomTrainTestSplit, KFoldSplit
from utils.Experiment import ExperimentConfig, Experiment

COMMON_RANKER = {
    "task": "binary",
    "stop_metric": "loss",
    "threshold_technique": "max_f1",  # "max_f1", "youden", "dist01", "fixed"
    "metrics_to_log": (
        "f1",
        # "precision",
        # "recall",
        "accuracy",
        "balanced_accuracy",
        # "mcc",
        "pr_auc",  # metryka niezalezna od progu
        "roc_auc",  # metryka niezalezna od progu
    ),
    "move_mode": "18",
    "epochs": 300,
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

if __name__ == "__main__":
    experiments = [
        ExperimentConfig(
            calculate_redundant_states=False,
            redundant_states_path="solves_marked.pkl",
            target_column="redundant_state_marker_binary",
            # target_column="redundant_state_marker_binary_first",
            window_size=10,
            stride=1,
            seed=42,
            schema=RandomTrainTestSplit(repetition=3,test_size=0.1, shuffle=True),
            # schema=KFoldSplit(k=10, shuffle=True),

            # KLUCZ: świeży model na fold
            model_factory=lambda: SimpleMLPNet(move_vocab_size=18),

            ranker_params=COMMON_RANKER,
            binary_eval_metrics=COMMON_METRICS,
        ),
    ]

    for idx, cfg in enumerate(experiments, start=1):
        print(f"\n================= EXPERIMENT {idx} =================")
        metrics = Experiment.run(cfg)
        print(f"Fold metrics ({idx}): {metrics}")
