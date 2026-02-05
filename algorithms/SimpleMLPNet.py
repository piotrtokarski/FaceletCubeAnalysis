import torch
import torch.nn as nn

class SimpleMLPNet(nn.Module):
    def __init__(self, move_vocab_size=18, d_state=16, d_move=32, hidden=(256, 128), dropout=0.3):
        super().__init__()
        self.state_emb = nn.Embedding(6, d_state)
        self.move_emb = nn.Embedding(move_vocab_size, d_move)

        # pozycje stickerów (żeby model wiedział który sticker jest który)
        self.sticker_pos = nn.Parameter(torch.randn(1, 1, 54, d_state) * 0.02)

        # kompresja 54*d_state -> d_state_tok (na każdy krok T)
        d_state_tok = 64
        self.state_proj = nn.Sequential(
            nn.Linear(54 * d_state, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_state_tok),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.LazyLinear(hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1)
        )

    def forward(self, states, moves):
        # states: (B,T,54)
        B, T, _ = states.shape

        s = self.state_emb(states) + self.sticker_pos     # (B,T,54,d_state)
        s = s.reshape(B, T, -1)                           # (B,T,54*d_state)
        s = self.state_proj(s)                            # (B,T,d_state_tok)

        m = self.move_emb(moves)                          # (B,W,d_move)

        x = torch.cat([s.reshape(B, -1), m.reshape(B, -1)], dim=1)
        return self.mlp(x)
