
import random
import time

import numpy as np
import torch
import torch.nn.init
from torch.utils.data import DataLoader

from algorithms.IF import IsolationForest
from networks.MLPNetwork import MLPNetwork


class DeepIF:
    def __init__(self, optimization=True, representations_number=50, trees_per_representation=6,
                 samples_number_per_tree=256,
                 network_hidden_dimensions=[500, 100], representation_dimensionality=20, batch_size=64,
                 device='cuda', seed=None):

        self.representations_number = representations_number
        self.trees_per_representation = trees_per_representation
        self.samples_number_per_tree = samples_number_per_tree
        self.network_hidden_dimensions = network_hidden_dimensions
        self.representation_dimensionality = representation_dimensionality
        self.batch_size = batch_size
        self.device = device
        self.optimization = optimization
        self.cpu_stages_fit_time = None
        self.gpu_stages_fit_time = None

        if seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        else:
            np.random.seed(random.randint(0, 2 ** 32 - 1))
            torch.manual_seed(random.randint(0, 2 ** 32 - 1))
            torch.cuda.manual_seed_all(random.randint(0, 2 ** 32 - 1))
        return

    def fit(self, X, Y=None):
        cpu_stage_1_start_time = time.perf_counter_ns()

        self.n_features = X.shape[-1]
        self.network = MLPNetwork(n_features=self.n_features, network_hidden_dimensions=self.network_hidden_dimensions,
                                  representation_dimensionality=self.representation_dimensionality,
                                  representations_number=self.representations_number, activation_fun='tanh',
                                  device=self.device)
        for name, parameter in self.network.named_parameters():
            if name.endswith('weight'):
                torch.nn.init.normal_(parameter, mean=0.0, std=1.0)

        if self.optimization == False:
            cpu_stage_1_stop_time = time.perf_counter_ns()

            x_representation = []
            with torch.no_grad():
                data_loader = DataLoader(X, batch_size=self.batch_size, drop_last=False, pin_memory=True, shuffle=False)
                for batch in data_loader:
                    batch = batch.float().to(self.device)
                    batch_representation = self.network(batch)
                    batch_representation = batch_representation.reshape(
                        [self.representations_number, batch.shape[0], -1])
                    x_representation.append(batch_representation)

            cpu_stage_2_start_time = time.perf_counter_ns()

            x_representation_list = [
                torch.cat([x_representation[i][j] for i in range(len(x_representation))]).data.cpu().numpy()
                for j in range(x_representation[0].shape[0])]

            self.isolation_forest_list = []
            for i in range(self.representations_number):
                i_forest = IsolationForest(trees_number=self.trees_per_representation,
                                           samples_number=self.samples_number_per_tree)
                self.isolation_forest_list.append(i_forest)
                i_forest.fit(x_representation_list[i])
            cpu_stage_2_stop_time = time.perf_counter_ns()
        else:
            sampled_indices = np.random.choice(len(X), self.samples_number_per_tree * self.trees_per_representation,
                                               replace=True)
            unique_indices, inverse_indices = np.unique(sampled_indices, return_inverse=True)

            cpu_stage_1_stop_time = time.perf_counter_ns()

            x_representation = []
            with torch.no_grad():
                data_loader = DataLoader(X[unique_indices], batch_size=self.batch_size, drop_last=False,
                                         pin_memory=True, shuffle=False)
                for batch in data_loader:
                    batch = batch.float().to(self.device)
                    batch_representation = self.network(batch)
                    batch_representation = batch_representation.reshape(
                        [self.representations_number, batch.shape[0], -1])
                    x_representation.append(batch_representation)

            cpu_stage_2_start_time = time.perf_counter_ns()

            x_representation_list = [
                torch.cat([x_representation[i][j] for i in range(len(x_representation))]).data.cpu().numpy()
                for j in range(x_representation[0].shape[0])]

            self.isolation_forest_list = []
            for i in range(self.representations_number):
                i_forest = IsolationForest(trees_number=self.trees_per_representation,
                                           samples_number=self.samples_number_per_tree)
                self.isolation_forest_list.append(i_forest)
                i_forest.clear_no_subsample()
                for j in range(self.trees_per_representation):
                    i_forest.fit_no_subsample(x_representation_list[i][inverse_indices[
                                                                       j * self.samples_number_per_tree:(
                                                                                                                    j + 1) * self.samples_number_per_tree]])
            cpu_stage_2_stop_time = time.perf_counter_ns()

        cpu_stages_elapsed_time = cpu_stage_1_stop_time - cpu_stage_1_start_time + cpu_stage_2_stop_time - cpu_stage_2_start_time
        gpu_stages_elapsed_time = cpu_stage_2_start_time - cpu_stage_1_stop_time
        self.cpu_stages_fit_time = cpu_stages_elapsed_time
        self.gpu_stages_fit_time = gpu_stages_elapsed_time

    def decision_function(self, X):
        score_partial = []
        with torch.no_grad():
            data_loader = DataLoader(X, batch_size=self.batch_size, drop_last=False, pin_memory=True, shuffle=False)
            for batch in data_loader:
                batch = batch.float().to(self.device)
                batch_representation = self.network(batch)

                R = self.representations_number
                B = batch.shape[0]
                batch_representation = batch_representation.reshape(R, B, -1).cpu()

                batch_scores = np.zeros((R, B))
                for r in range(R):
                    data = batch_representation[r].numpy()  # [B, F]
                    batch_scores[r] = self.isolation_forest_list[r].new_decision_function(data)

                score_partial.append(np.mean(batch_scores, axis=0))  # [B]
        final_scores = np.concatenate(score_partial, axis=0)  # shape: [N]
        return final_scores

    def algorithm_name(self):
        if self.optimization == True:
            return "OptimizedDeepIF"
        else:
            return "DeepIF"
