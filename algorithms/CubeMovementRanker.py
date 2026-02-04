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
            task: str = "binary",  # "binary" lub "regression"
            threshold: float = 0.5,  # tylko binary
            move_mode: str = "18",  # "18" lub "vocab"
            epochs: int = 10,
            batch_size: int = 256,
            lr: float = 1e-3,
            device: str | None = None,
            verbose=True,
            f1_average: str = "binary"  # <-- dla binary
    ):
        self.model = model
        self.task = task
        self.threshold = threshold
        self.move_mode = move_mode

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        self.f1_average = f1_average

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ustawiane w train() jeśli move_mode="vocab"
        self.move2id = None

        self._trained = False

    def train(self, X_train_df, y_train):
        self._prepare_encoder(X_train_df)

        X_states, X_moves = self._encode(X_train_df)
        y_train = np.asarray(y_train, dtype=np.float32)

        ds = _CubeDataset(X_states, X_moves, y_train)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model.to(self.device)
        loss_fn = self._make_loss()
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)

        for epoch in range(1, self.epochs + 1):
            self.model.train()

            total_loss = 0.0
            n = 0

            for states, moves, y in dl:
                states = states.to(self.device)
                moves = moves.to(self.device)
                y = y.to(self.device)

                opt.zero_grad(set_to_none=True)
                out = self.model(states, moves)  # (B,1)
                loss = loss_fn(out, y)
                loss.backward()
                opt.step()

                bs = y.size(0)
                total_loss += loss.item() * bs
                n += bs

            avg_loss = total_loss / max(n, 1)

            # --- log / metryki ---
            if self.verbose:
                if self.task == "binary":
                    train_acc, train_f1 = self._eval_binary_on_train(X_states, X_moves, y_train)
                    print(
                        f"Epoch {epoch:03d}/{self.epochs} | loss={avg_loss:.6f} | acc={train_acc:.4f} | f1={train_f1:.4f}")
                else:
                    # regresja: na razie tylko loss (możesz dodać MAE/RMSE)
                    print(f"Epoch {epoch:03d}/{self.epochs} | loss={avg_loss:.6f}")

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
