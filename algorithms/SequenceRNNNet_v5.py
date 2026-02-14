import torch
import torch.nn as nn


class SequenceRNNNet_v5(nn.Module):
    def __init__(
        self,
        move_vocab_size=18,
        d_state=24,
        d_move=24,
        d_state_token=96,
        rnn_hidden=160,
        dropout=0.30,
        rnn_type="gru",  # "gru" albo "lstm"
    ):
        super().__init__()

        self.rnn_type = rnn_type.lower()

        # --- embeddings ---
        self.state_emb = nn.Embedding(6, d_state)
        self.move_emb = nn.Embedding(move_vocab_size, d_move)
        self.sticker_pos = nn.Parameter(torch.randn(1, 1, 54, d_state) * 0.02)

        # mniejsza projekcja stanu niż v4 (szybciej + mniej overfitu)
        self.state_proj = nn.Sequential(
            nn.Linear(54 * d_state, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, d_state_token),
            nn.LayerNorm(d_state_token),
        )

        self.time_token_dim = d_state_token + d_move
        self.pre_rnn = nn.Sequential(
            nn.LayerNorm(self.time_token_dim),
            nn.Linear(self.time_token_dim, rnn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=rnn_hidden,
                hidden_size=rnn_hidden,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
                bidirectional=False,
            )
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=rnn_hidden,
                hidden_size=rnn_hidden,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
                bidirectional=False,
            )
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        # łączymy różne poolingi czasowe (last + mean + max)
        pooled_dim = rnn_hidden * 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, states, moves):
        """
        states: (B, T+1, 54)
        moves:  (B, T)
        """
        bsz, t_plus_1, _ = states.shape

        s = self.state_emb(states) + self.sticker_pos
        s = s.reshape(bsz, t_plus_1, -1)
        s = self.state_proj(s)

        m = self.move_emb(moves)
        zero_move = torch.zeros(bsz, 1, m.size(-1), device=m.device, dtype=m.dtype)
        m_padded = torch.cat([m, zero_move], dim=1)

        x = torch.cat([s, m_padded], dim=2)
        x = self.pre_rnn(x)

        out, _ = self.rnn(x)

        last = out[:, -1, :]
        mean_pool = out.mean(dim=1)
        max_pool = out.max(dim=1).values

        pooled = torch.cat([last, mean_pool, max_pool], dim=1)
        return self.classifier(pooled)
