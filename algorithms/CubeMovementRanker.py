import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    roc_auc_score,
    matthews_corrcoef,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from utils.BestThreshold import BestThreshold
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
            weight_decay: float = 1e-2,
            device: str | None = None,
            verbose=True,

            val_size: float = 0.15,
            early_stopping: bool = True,
            patience: int = 10,
            min_delta: float = 1e-4,
            restore_best_weights: bool = True,
            random_state: int | None = 42,
            stop_metric: str = "loss",  # np. "loss", "f1", "pr_auc", "roc_auc"
            threshold_technique: str = "max_f1",  # "max_f1" | "youden" | "dist01" | "fixed"
            metrics_to_log: tuple[str, ...] = ("f1", "pr_auc", "roc_auc"),
    ):
        self.model = model
        self.task = task
        self.threshold = threshold
        self.move_mode = move_mode
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # NOWE:
        self.val_size = val_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.random_state = random_state

        self.stop_metric = stop_metric.lower().strip()
        self.threshold_technique = threshold_technique.lower().strip()
        self.metrics_to_log = tuple(m.lower().strip() for m in metrics_to_log)


        self.move2id = None
        self._trained = False


    def train(self, X_train_df, y_train, X_test_wyciek=None, y_test_wyciek=None):
        self._prepare_encoder(X_train_df)

        X_tr_df, y_tr, X_val_df, y_val = self._split_train_val(X_train_df, y_train)
        if X_val_df is None or y_val is None:
            raise ValueError("Dla early stopping/model selection ustaw val_size w (0,1).")

        # train / val loaders
        X_tr_states, X_tr_moves = self._encode(X_tr_df)
        ds_tr = _CubeDataset(X_tr_states, X_tr_moves, y_tr)
        dl_tr = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=True, drop_last=False)

        X_val_states, X_val_moves = self._encode(X_val_df)
        ds_val = _CubeDataset(X_val_states, X_val_moves, y_val)
        dl_val = DataLoader(ds_val, batch_size=self.batch_size, shuffle=False, drop_last=False)

        # optional test loader do diagnostyki
        dl_test = None
        if X_test_wyciek is not None and y_test_wyciek is not None:
            X_test_states, X_test_moves = self._encode(X_test_wyciek)
            ds_test = _CubeDataset(X_test_states, X_test_moves, y_test_wyciek)
            dl_test = DataLoader(ds_test, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.model.to(self.device)
        loss_fn = self._make_loss(y_train=y_tr)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # inicjalizacja best
        if self._metric_direction(self.stop_metric) == "max":
            best_monitor = -np.inf
        else:
            best_monitor = np.inf

        best_state = None
        best_threshold = float(self.threshold)
        bad_epochs = 0


        for epoch in range(1, self.epochs + 1):
            # ---------- TRAIN ----------
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

            # ---------- VAL ----------
            val_loss, y_val_true, y_val_score = self._eval_loader(
                dl_val, loss_fn, collect_scores=(self.task == "binary")
            )

            epoch_thr = None
            val_metrics = {}

            if self.task == "binary":
                epoch_thr = self._pick_threshold(y_val_true, y_val_score)
                val_metrics = self._compute_binary_metrics(y_val_true, y_val_score, epoch_thr)

            # monitor
            if self.stop_metric == "loss":
                monitor = float(val_loss)
            else:
                monitor = float(val_metrics.get(self.stop_metric, np.nan))

            improved = self._is_improved(monitor, best_monitor, self.stop_metric)

            # ---------- TEST (diagnostycznie) ----------
            test_loss = None
            test_metrics = None

            if dl_test is not None:
                test_loss, y_test_true, y_test_score = self._eval_loader(
                    dl_test, loss_fn, collect_scores=(self.task == "binary")
                )
                if self.task == "binary":
                    thr_for_test = epoch_thr if epoch_thr is not None else self.threshold
                    test_metrics = self._compute_binary_metrics(y_test_true, y_test_score, thr_for_test)

            # log
            if self.verbose:
                if self.task == "binary":
                    log_main = (
                            f"Epoch {epoch:03d}/{self.epochs} | "
                            f"train_loss={self._fmt(train_loss)} | val_loss={self._fmt(val_loss)} | "
                            f"val_thr={self._fmt(epoch_thr)} | "
                            f"monitor({self.stop_metric})={self._fmt(monitor)}"
                        )

                    val_parts = [f"val_{m}={self._fmt(val_metrics.get(m))}" for m in self.metrics_to_log]
                    log_val = " | " + " | ".join(val_parts) if val_parts else ""

                    if test_metrics is not None:
                        test_parts = [f"test_{m}={self._fmt(test_metrics.get(m))}" for m in self.metrics_to_log]
                        log_test = f" | test_loss={self._fmt(test_loss)} | " + " | ".join(test_parts)
                    else:
                        log_test = ""

                    print(log_main + log_val + log_test)
                else:
                    print(
                            f"Epoch {epoch:03d}/{self.epochs} | "
                            f"train_loss={self._fmt(train_loss)} | val_loss={self._fmt(val_loss)}"
                        )

            # model selection
            if improved:
                best_monitor = monitor
                bad_epochs = 0

                if self.task == "binary" and epoch_thr is not None:
                    best_threshold = float(epoch_thr)

                if self.restore_best_weights:
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad_epochs += 1


            # early stopping
            if self.early_stopping and bad_epochs >= self.patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch}, best {self.stop_metric}={self._fmt(best_monitor)}")
                break

        # restore best checkpoint
        if self.restore_best_weights and best_state is not None:
            self.model.load_state_dict(best_state)

        # final threshold
        if self.task == "binary":
            # po restore best checkpoint najlepiej policzyć próg ponownie na val lub moze na train + val ale to juz do zbadania
            if self.restore_best_weights:
                _, y_val_true, y_val_score = self._eval_loader(
                    dl_val, loss_fn, collect_scores=True
                )
                self.threshold = self._pick_threshold(y_val_true, y_val_score)
            else:
                self.threshold = best_threshold

        # ===== FINAL SUMMARY =====
        # train eval loader bez shuffle (żeby summary było stabilne)
        dl_tr_eval = DataLoader(ds_tr, batch_size=self.batch_size, shuffle=False, drop_last=False)

        summary = {
            "best_monitor_metric": self.stop_metric,
            "best_monitor_value": float(best_monitor) if np.isfinite(best_monitor) else np.nan,
            "final_threshold": float(self.threshold) if self.task == "binary" else None,
        }

        if self.task == "binary":
            # TRAIN
            train_loss_f, y_tr_true_f, y_tr_score_f = self._eval_loader(dl_tr_eval, loss_fn, collect_scores=True)
            train_metrics_f = self._compute_binary_metrics(y_tr_true_f, y_tr_score_f, self.threshold)

            # VAL
            val_loss_f, y_val_true_f, y_val_score_f = self._eval_loader(dl_val, loss_fn, collect_scores=True)
            val_metrics_f = self._compute_binary_metrics(y_val_true_f, y_val_score_f, self.threshold)

            # TEST (opcjonalnie)
            test_loss_f, test_metrics_f = None, None
            if dl_test is not None:
                test_loss_f, y_test_true_f, y_test_score_f = self._eval_loader(dl_test, loss_fn, collect_scores=True)
                test_metrics_f = self._compute_binary_metrics(y_test_true_f, y_test_score_f, self.threshold)

            summary.update({
                "train_loss": float(train_loss_f),
                "val_loss": float(val_loss_f),
                "test_loss": None if test_loss_f is None else float(test_loss_f),
                "train_metrics": train_metrics_f,
                "val_metrics": val_metrics_f,
                "test_metrics": test_metrics_f,
            })

            if self.verbose:
                print("\n================ FINAL SUMMARY ================")
                print(
                    f"best_{self.stop_metric}={self._fmt(summary['best_monitor_value'])} | "
                    f"final_thr={self._fmt(summary['final_threshold'])}"
                )
                print(f"TRAIN | loss={self._fmt(train_loss_f)} | {self._format_metrics_all(train_metrics_f)}")
                print(f"VAL   | loss={self._fmt(val_loss_f)} | {self._format_metrics_all(val_metrics_f)}")
                if test_metrics_f is not None:
                    print(f"TEST  | loss={self._fmt(test_loss_f)} | {self._format_metrics_all(test_metrics_f)}")
                print("==============================================\n")

        else:
            # regression: tylko losses
            train_loss_f, _, _ = self._eval_loader(dl_tr_eval, loss_fn, collect_scores=False)
            val_loss_f, _, _ = self._eval_loader(dl_val, loss_fn, collect_scores=False)
            test_loss_f = None
            if dl_test is not None:
                test_loss_f, _, _ = self._eval_loader(dl_test, loss_fn, collect_scores=False)

            summary.update({
                "train_loss": float(train_loss_f),
                "val_loss": float(val_loss_f),
                "test_loss": None if test_loss_f is None else float(test_loss_f),
            })

            if self.verbose:
                print("\n================ FINAL SUMMARY ================")
                print(f"best_{self.stop_metric}={self._fmt(summary['best_monitor_value'])}")
                print(f"TRAIN | loss={self._fmt(train_loss_f)}")
                print(f"VAL   | loss={self._fmt(val_loss_f)}")
                if test_loss_f is not None:
                    print(f"TEST  | loss={self._fmt(test_loss_f)}")
                print("==============================================\n")

        # zapisz summary do obiektu (przyda się poza printem)
        self.summary_ = summary

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

        raw = np.concatenate(outs, axis=0).astype(np.float32)
        return raw

    def predict(self, X_test_df):
        scores = self.predict_scores(X_test_df)
        if self.task == "binary":
            return (scores >= self.threshold).astype(np.int32)
        return scores

    def score_samples(self, X_test_df):
        return self.predict_scores(X_test_df)

    def _metric_direction(self, metric_name: str) -> str:
        # "max" albo "min"
        if metric_name == "loss":
            return "min"
        return "max"

    @staticmethod
    def _fmt(x):
        if x is None:
            return "NA"
        try:
            xf = float(x)
            if np.isnan(xf):
                return "NA"
            return f"{xf:.6f}"
        except Exception:
            return "NA"

    def _is_improved(self, current: float, best: float, metric_name: str) -> bool:
        if current is None or np.isnan(current):
            return False
        direction = self._metric_direction(metric_name)
        if direction == "max":
            return current > (best + self.min_delta)
        else:
            return current < (best - self.min_delta)

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

    def _make_loss(self, y_train=None):
        if self.task == "binary":
            # opcjonalnie możesz tu odkomentować pos_weight:
            if y_train is not None:
                yb = (np.asarray(y_train).reshape(-1) >= 0.5).astype(np.int32)
                n_pos = int(yb.sum()); n_neg = int((1 - yb).sum())
                pos_weight = n_neg / max(n_pos, 1)
                pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=self.device)
                return nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
            return nn.BCEWithLogitsLoss()
        return nn.MSELoss()

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

    def _eval_loader(self, dl, loss_fn, collect_scores: bool):
        self.model.eval()

        loss_sum = 0.0
        n_all = 0
        all_scores = []
        all_y = []

        with torch.no_grad():
            for states, moves, y in dl:
                states = states.to(self.device)
                moves = moves.to(self.device)
                y = y.to(self.device)

                out = self.model(states, moves).view(-1, 1)
                loss = loss_fn(out, y)

                bs = y.size(0)
                loss_sum += loss.item() * bs
                n_all += bs

                if collect_scores:
                    all_scores.append(out.squeeze(1).detach().cpu().numpy())
                    all_y.append(y.squeeze(1).detach().cpu().numpy())

        avg_loss = loss_sum / max(n_all, 1)

        if collect_scores:
            y_true = np.concatenate(all_y, axis=0).astype(np.int32)
            y_score = np.concatenate(all_scores, axis=0).astype(np.float32)
        else:
            y_true, y_score = None, None

        return avg_loss, y_true, y_score


    def _pick_threshold(self, y_true, y_score):
        if self.threshold_technique == "fixed":
            return float(self.threshold)

        # bez obu klas nie stroimy progu
        if y_true is None or np.unique(y_true).size < 2:
            return float(self.threshold)

        thr, _ = BestThreshold._best_threshold(
                y_true=y_true, y_score=y_score, technique=self.threshold_technique
            )
        return float(thr)

    def _format_metrics_all(self, metrics: dict) -> str:
        return " | ".join(
            f"{k}={self._fmt(metrics.get(k))}" for k in self.metrics_to_log
        )

    def _compute_binary_metrics(self, y_true, y_score, thr: float):
        y_true = np.asarray(y_true).astype(np.int32).reshape(-1)
        y_score = np.asarray(y_score).astype(np.float32).reshape(-1)
        y_pred = (y_score >= float(thr)).astype(np.int32)

        out = {}

        out["f1"] = f1_score(y_true, y_pred, average="binary", zero_division=0)
        out["precision"] = precision_score(y_true, y_pred, zero_division=0)
        out["recall"] = recall_score(y_true, y_pred, zero_division=0)
        out["accuracy"] = accuracy_score(y_true, y_pred)
        out["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        out["mcc"] = matthews_corrcoef(y_true, y_pred)

        # threshold-independent
        if np.unique(y_true).size < 2:
            out["pr_auc"] = float("nan")
            out["roc_auc"] = float("nan")
        else:
            # average_precision_score to standardowy PR-AUC w praktyce ML
            out["pr_auc"] = average_precision_score(y_true, y_score)
            out["roc_auc"] = roc_auc_score(y_true, y_score)

        return out

