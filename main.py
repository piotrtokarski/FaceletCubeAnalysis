from algorithms.SequenceRNNNet_v6 import SequenceRNNNet_v6
from algorithms.SequenceRNNNet_v5 import SequenceRNNNet_v5
from algorithms.SequenceRNNNet_v4 import SequenceRNNNet_v4
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
    "epochs": 90,
    "batch_size": 192,
    "lr": 8e-4,
    "patience": 7,
    "min_delta": 8e-4,
    "weight_decay": 7e-3,
    "optimizer_name": "adamw",
    "scheduler_name": "reduce_on_plateau",
    "scheduler_factor": 0.6,
    "scheduler_patience": 2,
    "min_lr": 3e-5,
    "grad_clip_norm": 1.0,
    "loss_name": "focal",
    "focal_alpha": 0.80,
    "focal_gamma": 1.5,
    "train_sampling": "balanced",
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
            model_factory=lambda: SequenceRNNNet_v6(
                move_vocab_size=18,
                d_state=28,
                d_move=28,
                d_state_token=112,
                rnn_hidden=192,
                dropout=0.25,
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
