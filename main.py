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

from utils.Experiment import ExperimentConfig, Experiment

if __name__ == "__main__":
    experiments = [
        ExperimentConfig(
            calculate_redundant_states=False,
            redundant_states_path="solves_marked.pkl",
            target_column="redundant_state_marker_binary",
            window_size=10,
            stride=1,
            seed=None,
            # schema=RandomTrainTestSplit(test_size=0.1, shuffle=True),
            schema=KFoldSplit(k=10, shuffle=True),
            model=SimpleMLPNet(move_vocab_size=18),
            ranker_params={
                "task": "binary",
                "stop_metric": "loss",
                "move_mode": "18",
                "epochs": 300,
                "batch_size": 256,
                "lr": 1e-3,
                "verbose": True,
                "val_size": 0.10
            },
        ),
        # ExperimentConfig(
        #     calculate_redundant_states=False,
        #     redundant_states_path="solves_marked.pkl",
        #     target_column="redundant_state_marker",
        #     window_size=10,
        #     stride=1,
        #     seed=None,
        #     # schema=RandomTrainTestSplit(test_size=0.1, shuffle=True),
        #     schema=KFoldSplit(k=10, shuffle=True),
        #     model=SimpleMLPNet(move_vocab_size=18),
        #     ranker_params={
        #         "task": "regression",
        #         "move_mode": "18",
        #         "epochs": 300,
        #         "batch_size": 256,
        #         "lr": 1e-3,
        #         "verbose": True,
        #         "val_size": 0.10
        #     },
        # )
    ]

    for idx, cfg in enumerate(experiments, start=1):
        print(f"\n================= EXPERIMENT {idx} =================")
        metrics = Experiment.run(cfg)
        print(f"Fold metrics ({idx}): {metrics}")