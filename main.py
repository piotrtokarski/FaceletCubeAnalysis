from algorithms.SequenceRNNNet_v7 import SequenceRNNNet_v7
from utils.EvaluationSchema import RandomTrainTestSplit, KFoldSplit
from utils.Experiment import ExperimentConfig, Experiment

COMMON_RANKER = {
    "task": "binary",
    "stop_metric": "f1",
    "threshold_technique": "max_f1",  # "max_f1", "youden", "dist01", "fixed"
    "metrics_to_log": (
        "f1",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "pr_auc",  # metryka niezalezna od progu
        "roc_auc",  # metryka niezalezna od progu
    ),
    "move_mode": "18",
    "epochs": 140,
    "batch_size": 128,
    "lr": 5e-4,
    "patience": 12,
    "min_delta": 5e-4,
    "weight_decay": 3e-3,
    "optimizer_name": "adamw",
    "scheduler_name": "cosine",
    "scheduler_factor": 0.6,
    "scheduler_patience": 2,
    "min_lr": 1e-5,
    "grad_clip_norm": 1.0,
    "loss_name": "bce",
    "focal_alpha": 0.80,
    "focal_gamma": 1.5,
    "train_sampling": "uniform",
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
            window_size=5,
            stride=1,
            seed=42,
            schema=RandomTrainTestSplit(repetition=10,test_size=0.1, shuffle=True),
            # schema=KFoldSplit(k=10, shuffle=True),

            # KLUCZ: świeży model na fold
            model_factory=lambda: SequenceRNNNet_v7(
                move_vocab_size=18,
                d_state=32,
                d_move=32,
                d_state_token=128,
                rnn_hidden=192,
                dropout=0.2,
                rnn_type="gru",
            ),

            ranker_params=COMMON_RANKER,
            binary_eval_metrics=COMMON_METRICS,
        ),
    ]

    for idx, cfg in enumerate(experiments, start=1):
        print(f"\n================= EXPERIMENT {idx} =================")
        metrics = Experiment.run(cfg)
        print(f"Fold metrics ({idx}): {metrics}")
