import torch
import torch.nn as nn


class SequenceRNNNet_v6(nn.Module):
    def __init__(
        self,
        move_vocab_size=18,
        d_state=28,
        d_move=28,
        d_state_token=112,
        rnn_hidden=192,
        dropout=0.25,
        rnn_type="gru",  # "gru" albo "lstm"
    ):
        super().__init__()
        self.rnn_type = rnn_type.lower()

        self.state_emb = nn.Embedding(6, d_state)
        self.move_emb = nn.Embedding(move_vocab_size, d_move)
        self.sticker_pos = nn.Parameter(torch.randn(1, 1, 54, d_state) * 0.02)

        self.state_proj = nn.Sequential(
            nn.Linear(54 * d_state, 224),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(224, d_state_token),
            nn.LayerNorm(d_state_token),
            nn.GELU(),
        )

        token_dim = d_state_token + d_move
        self.token_gate = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, token_dim),
            nn.Sigmoid(),
        )

        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM if self.rnn_type == "lstm" else None
        if rnn_cls is None:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        self.rnn = rnn_cls(
            input_size=token_dim,
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.attn = nn.Sequential(
            nn.Linear(rnn_hidden, rnn_hidden // 2),
            nn.Tanh(),
            nn.Linear(rnn_hidden // 2, 1),
        )

        pooled_dim = rnn_hidden * 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(96, 1),
        )

    def forward(self, states, moves):
        bsz, t_plus_1, _ = states.shape

        s = self.state_emb(states) + self.sticker_pos
        s = s.reshape(bsz, t_plus_1, -1)
        s = self.state_proj(s)

        m = self.move_emb(moves)
        zero_move = torch.zeros(bsz, 1, m.size(-1), device=m.device, dtype=m.dtype)
        m = torch.cat([m, zero_move], dim=1)

        x = torch.cat([s, m], dim=2)
        g = self.token_gate(x)
        x = x * g

        out, _ = self.rnn(x)
        attn_w = torch.softmax(self.attn(out), dim=1)
        attn_pool = (out * attn_w).sum(dim=1)
        mean_pool = out.mean(dim=1)
        max_pool = out.max(dim=1).values

        feats = torch.cat([attn_pool, mean_pool, max_pool], dim=1)
        return self.classifier(feats)
