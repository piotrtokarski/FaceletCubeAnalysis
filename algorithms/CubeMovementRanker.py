import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.CubeEncoder import CubeEncoder


class _CubeDataset(Dataset):
    def __init__(self, X_states, X_moves, y):
        self.states = torch.as_tensor(X_states, dtype=torch.long)  # (N,T,54)
        self.moves = torch.as_tensor(X_moves, dtype=torch.long)  # (N,W)
        self.y = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)  # (N,1)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.moves[idx], self.y[idx]


class CubeMovementRanker:
    def __init__(
            self,
            model: nn.Module,
            task: str = "binary",
            threshold: float = 0.5,
            move_mode: str = "18",
            epochs: int = 100,
            batch_size: int = 256,
            lr: float = 1e-3,
            device: str | None = None,
            verbose=True,

            val_size: float = 0.15,
            early_stopping: bool = True,
            patience: int = 10,
            min_delta: float = 1e-4,
            restore_best_weights: bool = True,
            random_state: int | None = 42
    ):
        self.model = model
        self.task = task
        self.threshold = threshold
        self.move_mode = move_mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # NOWE:
        self.val_size = val_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.random_state = random_state

        self.move2id = None
        self._trained = False

    def train(self, X_train_df, y_train):
        self._prepare_encoder(X_train_df)

        X_tr_df, y_tr, X_val_df, y_val = self._split_train_val(X_train_df, y_train)

        X_tr_states, X_tr_moves = self._encode(X_tr_df)
        ds_tr = _CubeDataset(X_tr_states, X_tr_moves, y_tr)
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=False)

        X_val_states, X_val_moves = self._encode(X_val_df)
        ds_val = _CubeDataset(X_val_states, X_val_moves, y_val)
        dl_val = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.model.to(self.device)
        loss_fn = self._make_loss()
        # if self.task == "binary":
        #     n_pos = int((y_tr >= 0.5).sum())
        #     n_neg = int((y_tr < 0.5).sum())
        #     pos_weight = n_neg / max(n_pos, 1)
        #
        #     if self.verbose:
        #         print(f"[train] n_pos={n_pos}, n_neg={n_neg}, pos_weight={pos_weight:.4f}")
        #
        #     pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=self.device)
        #     loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
        # else:
        #     loss_fn = self._make_loss()

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        best_val_loss = np.inf
        best_state = None
        bad_epochs = 0

        for epoch in range(1, self.epochs + 1):
            # ---- TRAIN ----
            self.model.train()
            train_loss_sum = 0.0
            train_n = 0

            for states, moves, y in dl_tr:
                states = states.to(self.device)
                moves = moves.to(self.device)
                y = y.to(self.device)

                opt.zero_grad(set_to_none=True)
                out = self.model(states, moves).view(-1, 1)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()

                bs = y.size(0)
                train_loss_sum += loss.item() * bs
                train_n += bs

            train_loss = train_loss_sum / max(train_n, 1)

            # ---- VAL ----
            self.model.eval()
            val_loss_sum = 0.0
            val_n = 0
            with torch.no_grad():
                for states, moves, y in dl_val:
                    states = states.to(self.device)
                    moves = moves.to(self.device)
                    y = y.to(self.device)

                    out = self.model(states, moves).view(-1, 1)
                    loss = loss_fn(out, y)

                    bs = y.size(0)
                    val_loss_sum += loss.item() * bs
                    val_n += bs

            val_loss = val_loss_sum / max(val_n, 1)

            if self.verbose:
                print(f"Epoch {epoch:03d}/{self.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

            # EARLY STOPPING na val_loss
            if self.early_stopping:
                improved = val_loss < (best_val_loss - self.min_delta)
                if improved:
                    best_val_loss = val_loss
                    bad_epochs = 0
                    if self.restore_best_weights:
                        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    bad_epochs += 1

                if bad_epochs >= self.patience:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch}, best val_loss={best_val_loss:.6f}")
                    break
            else:
                if self.verbose:
                    print(f"Epoch {epoch:03d}/{self.epochs} | train_loss={train_loss:.6f}")

        # restore best
        if self.early_stopping and self.restore_best_weights and best_state is not None:
            self.model.load_state_dict(best_state)

        self._trained = True
        return self

    # -------- prediction --------

    @torch.no_grad()
    def predict_scores(self, X_test_df):
        if not self._trained:
            raise RuntimeError("Call train() before predict().")

        X_states, X_moves = self._encode(X_test_df)
        dummy_y = np.zeros((len(X_states),), dtype=np.float32)
        ds = _CubeDataset(X_states, X_moves, dummy_y)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.model.eval()

        outs = []
        for states, moves, _ in dl:
            states = states.to(self.device)
            moves = moves.to(self.device)
            out = self.model(states, moves).squeeze(1)  # (B,)
            outs.append(out.detach().cpu().numpy())

        raw = np.concatenate(outs, axis=0)

        if self.task == "binary":
            # zwracamy prawdopodobieństwo klasy 1
            return 1.0 / (1.0 + np.exp(-raw))
        else:
            # regresja: surowa wartość
            return raw.astype(np.float32)

    def predict(self, X_test_df):
        scores = self.predict_scores(X_test_df)
        if self.task == "binary":
            return (scores >= self.threshold).astype(np.int32)
        return scores

    def score_samples(self, X_test_df):
        scores = self.predict_scores(X_test_df)
        return scores

    # -------- encoding --------

    def _prepare_encoder(self, X_train_df):
        if self.move_mode == "vocab":
            self.move2id = CubeEncoder.build_move_vocab_from_X(X_train_df)
        else:
            self.move2id = None

    def _encode(self, X_df):
        X_states, X_moves = CubeEncoder.encode_windows_df(
            X_df, move_mode=self.move_mode, move2id=self.move2id
        )
        return X_states, X_moves

    # -------- training --------

    def _make_loss(self):
        if self.task == "binary":
            return nn.BCEWithLogitsLoss()
        if self.task == "regression":
            return nn.MSELoss()
        raise ValueError("task must be 'binary' or 'regression'")

    @torch.no_grad()
    def _eval_binary_on_train(self, X_states, X_moves, y_true_np):
        """Szybka ewaluacja na train: acc + f1."""
        self.model.eval()

        ds = _CubeDataset(X_states, X_moves, y_true_np)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        probs = []
        for states, moves, _ in dl:
            states = states.to(self.device)
            moves = moves.to(self.device)
            logits = self.model(states, moves).squeeze(1)  # (B,)
            p = torch.sigmoid(logits)
            probs.append(p.cpu().numpy())

        probs = np.concatenate(probs, axis=0)
        y_pred = (probs >= self.threshold).astype(np.int32)
        y_true = y_true_np.astype(np.int32)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average=self.f1_average)
        return acc, f1

    def _split_train_val(self, X_df, y_np):
        n = len(X_df)

        if self.val_size <= 0.0 or self.val_size >= 1.0 or n < 2:
            return X_df.reset_index(drop=True), y_np, None, None

        n_val = int(np.floor(n * self.val_size))
        n_val = max(1, min(n_val, n - 1))

        rng = np.random.default_rng(self.random_state)
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        X_tr = X_df.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_np[tr_idx]
        X_val = X_df.iloc[val_idx].reset_index(drop=True)
        y_val = y_np[val_idx]
        return X_tr, y_tr, X_val, y_val