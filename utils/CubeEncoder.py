import numpy as np
import re


class CubeEncoder:
    FACE_TO_ID = {'U': 0, 'R': 1, 'F': 2, 'D': 3, 'L': 4, 'B': 5}
    MOVE_RE = re.compile(r"^([URFDLB])([2']?)$")  # R, R2, R'

    # ---------- STATES ----------

    @staticmethod
    def encode_facelet(facelet: str) -> np.ndarray:
        """facelet: string długości 54 z literami URFDLB -> int[54] 0..5"""
        arr = np.fromiter((CubeEncoder.FACE_TO_ID[ch] for ch in facelet),
                          dtype=np.int8, count=54)
        if arr.shape[0] != 54:
            raise ValueError(f"Facelet length != 54, got {arr.shape[0]}")
        return arr

    @staticmethod
    def encode_states_ctx(states_ctx) -> np.ndarray:
        """states_ctx: lista długości (W+1) -> (W+1, 54)"""
        return np.stack([CubeEncoder.encode_facelet(s) for s in states_ctx], axis=0)

    # ---------- MOVES (18-class) ----------

    @staticmethod
    def encode_move_18(move: str) -> int:
        """
        18 klas: face*3 + power, power: 0=cw, 1=double, 2=ccw
        np. R -> 3, R2 -> 4, R' -> 5 (bo U=0.., R=1..)
        """
        m = CubeEncoder.MOVE_RE.match(move)
        if not m:
            return -1
        face, suf = m.group(1), m.group(2)
        face_id = "URFDLB".index(face)
        power = 0 if suf == "" else (1 if suf == "2" else 2)
        return face_id * 3 + power  # 0..17

    @staticmethod
    def encode_moves_ctx_18(moves_ctx) -> np.ndarray:
        ids = []
        for mv in moves_ctx:
            v = CubeEncoder.encode_move_18(str(mv))
            if v < 0:
                raise ValueError(f"Move '{mv}' not supported by 18-class encoder.")
            ids.append(v)
        return np.array(ids, dtype=np.int8)

    # ---------- MOVES (vocab) ----------

    @staticmethod
    def build_move_vocab_from_X(X_df, extra_tokens=("PAD", "UNK")) -> dict:
        """Buduje move2id z X_train (DataFrame z kolumną moves_ctx)."""
        uniq = set()
        for row in X_df.itertuples(index=False):
            for mv in row.moves_ctx:
                uniq.add(str(mv))
        tokens = list(extra_tokens) + sorted(uniq)
        return {t: i for i, t in enumerate(tokens)}

    @staticmethod
    def encode_moves_ctx_vocab(moves_ctx, move2id: dict) -> np.ndarray:
        unk = move2id.get("UNK", 1)
        return np.array([move2id.get(str(m), unk) for m in moves_ctx], dtype=np.int16)

    # ---------- BATCH / WINDOWS ----------

    @staticmethod
    def encode_windows_df(
        X_df,
        move_mode: str = "18",    # "18" albo "vocab"
        move2id: dict | None = None,
    ):
        """
        X_df: DataFrame z kolumnami states_ctx, moves_ctx (po Twoich dropach).
        Zwraca:
          X_states: (N, W+1, 54) int
          X_moves : (N, W) int
        """
        N = len(X_df)
        if N == 0:
            return None, None

        X_states = []
        X_moves = []

        for row in X_df.itertuples(index=False):
            X_states.append(CubeEncoder.encode_states_ctx(row.states_ctx))

            if move_mode == "18":
                X_moves.append(CubeEncoder.encode_moves_ctx_18(row.moves_ctx))
            elif move_mode == "vocab":
                if move2id is None:
                    raise ValueError("move2id is required for move_mode='vocab'")
                X_moves.append(CubeEncoder.encode_moves_ctx_vocab(row.moves_ctx, move2id))
            else:
                raise ValueError("move_mode must be '18' or 'vocab'")

        return np.stack(X_states, axis=0), np.stack(X_moves, axis=0)

    # ---------- OPTIONAL: one-hot + flatten (np. pod sklearn) ----------

    @staticmethod
    def one_hot(x: np.ndarray, depth: int) -> np.ndarray:
        return np.eye(depth, dtype=np.float32)[x]

    @staticmethod
    def featurize_flat_onehot(
        X_df,
        move_mode: str = "18",
        move2id: dict | None = None,
    ) -> np.ndarray:
        """
        Zwraca macierz cech (N, D) dla sklearn:
        - states: onehot 6 i flatten
        - moves : onehot (18 lub vocab_size) i flatten
        """
        X_states, X_moves = CubeEncoder.encode_windows_df(X_df, move_mode=move_mode, move2id=move2id)

        if move_mode == "18":
            mv_depth = 18
        else:
            mv_depth = len(move2id)

        s_feat = CubeEncoder.one_hot(X_states, depth=6).reshape(len(X_df), -1)
        m_feat = CubeEncoder.one_hot(X_moves, depth=mv_depth).reshape(len(X_df), -1)
        return np.concatenate([s_feat, m_feat], axis=1)
