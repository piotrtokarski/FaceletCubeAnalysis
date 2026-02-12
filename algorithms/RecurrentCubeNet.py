import torch
import torch.nn as nn


class RecurrentCubeNet(nn.Module):
    def __init__(
        self,
        move_vocab_size: int = 18,
        d_state: int = 24,
        d_move: int = 24,
        d_state_token: int = 96,
        d_model: int = 192,
        rnn_hidden: int = 192,
        rnn_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # embeddingi wejścia
        self.state_emb = nn.Embedding(6, d_state)

        # +1 na token PAD (używany dla ostatniego stanu bez ruchu)
        self.move_pad_idx = move_vocab_size
        self.move_emb = nn.Embedding(move_vocab_size + 1, d_move, padding_idx=self.move_pad_idx)

        # pozycja stickerów na kostce: (1, 1, 54, d_state)
        self.sticker_pos = nn.Parameter(torch.randn(1, 1, 54, d_state) * 0.02)

        # kompresja jednego stanu (54 stickerów) -> token stanu
        self.state_proj = nn.Sequential(
            nn.Linear(54 * d_state, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_state_token),
            nn.GELU(),
        )

        # fuzja tokenu stanu + tokenu ruchu
        self.input_proj = nn.Sequential(
            nn.Linear(d_state_token + d_move, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.rnn = nn.GRU(
            input_size=d_model,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            dropout=dropout if rnn_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        self.attn = nn.Sequential(
            nn.Linear(2 * rnn_hidden, 2 * rnn_hidden),
            nn.Tanh(),
            nn.Linear(2 * rnn_hidden, 1),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(4 * rnn_hidden),
            nn.Linear(4 * rnn_hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, states: torch.Tensor, moves: torch.Tensor) -> torch.Tensor:
        # states: (B, T, 54), moves: (B, T-1)
        bsz, time_steps, _ = states.shape

        # stan -> token sekwencyjny
        s = self.state_emb(states) + self.sticker_pos
        s = s.reshape(bsz, time_steps, -1)
        s = self.state_proj(s)

        # ruchy wyrównane do T (ostatni krok to PAD)
        padded_moves = torch.full(
            (bsz, time_steps),
            self.move_pad_idx,
            dtype=moves.dtype,
            device=moves.device,
        )
        padded_moves[:, : moves.shape[1]] = moves
        m = self.move_emb(padded_moves)

        x = torch.cat([s, m], dim=-1)
        x = self.input_proj(x)

        rnn_out, _ = self.rnn(x)

        # attention pooling + ostatni krok daje lepszą stabilność niż sam last hidden
        attn_logits = self.attn(rnn_out).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(rnn_out * attn_weights.unsqueeze(-1), dim=1)

        last = rnn_out[:, -1, :]
        features = torch.cat([pooled, last], dim=-1)
        return self.head(features)
