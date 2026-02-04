import torch
import torch.nn as nn

class SimpleMLPNet(nn.Module):
    """
    Zachowuje kolejność:
      - stany: (B, T, 54) -> embed -> (B, T, d_state)  [mean po 54 stickerach]
      - ruchy: (B, W)     -> embed -> (B, W, d_move)
    Potem flatten w kolejności i MLP.
    """
    def __init__(self, move_vocab_size=18, d_state=32, d_move=32, hidden=(256, 128), dropout=0.2):
        super().__init__()
        self.state_emb = nn.Embedding(6, d_state)
        self.move_emb  = nn.Embedding(move_vocab_size, d_move)

        # LazyLinear nie wymaga znajomości (T,W) w __init__
        layers = [
            nn.LazyLinear(hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1)  # logit
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, states, moves):
        # states: (B,T,54)
        s = self.state_emb(states).mean(dim=2)   # (B,T,d_state)  <-- kolejność T zachowana
        m = self.move_emb(moves)                # (B,W,d_move)    <-- kolejność W zachowana

        x = torch.cat([s.reshape(s.size(0), -1), m.reshape(m.size(0), -1)], dim=1)
        return self.mlp(x)  # (B,1) logit
