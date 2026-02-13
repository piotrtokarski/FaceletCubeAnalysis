import torch
import torch.nn as nn


class SequenceRNNNet_v4(nn.Module):
    def __init__(
        self,
        move_vocab_size=18,
        d_state=32,
        d_move=32,
        d_state_token=128,
        rnn_hidden=192,
        rnn_layers=2,
        dropout=0.25,
        rnn_type="gru",  # "gru" albo "lstm"
        bidirectional=True,
    ):
        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.bidirectional = bool(bidirectional)

        # --- embeddings ---
        self.state_emb = nn.Embedding(6, d_state)
        self.move_emb = nn.Embedding(move_vocab_size, d_move)
        self.sticker_pos = nn.Parameter(torch.randn(1, 1, 54, d_state) * 0.02)

        # kompresja 54*d_state -> d_state_token
        self.state_proj = nn.Sequential(
            nn.Linear(54 * d_state, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_state_token),
            nn.LayerNorm(d_state_token),
            nn.GELU(),
        )

        self.time_token_dim = d_state_token + d_move

        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM if self.rnn_type == "lstm" else None
        if rnn_cls is None:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        self.rnn = rnn_cls(
            input_size=self.time_token_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
            bidirectional=self.bidirectional,
        )

        d_out = rnn_hidden * (2 if self.bidirectional else 1)

        # atencja po czasie, zamiast brania tylko ostatniego kroku
        self.attn = nn.Sequential(
            nn.Linear(d_out, d_out // 2),
            nn.Tanh(),
            nn.Linear(d_out // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_out),
            nn.Linear(d_out, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
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

        if self.rnn_type == "lstm":
            out, _ = self.rnn(x)
        else:
            out, _ = self.rnn(x)

        # attention pooling po czasie: (B,T+1,D) -> (B,D)
        attn_logits = self.attn(out)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(out * attn_weights, dim=1)

        logits = self.classifier(pooled)
        return logits
