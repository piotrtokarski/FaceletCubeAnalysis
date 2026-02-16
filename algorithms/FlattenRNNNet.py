import torch
import torch.nn as nn

# Zastosowanie:
# model_factory=lambda: FlattenRNNNet(move_vocab_size=18)

class FlattenRNNNet(nn.Module):
    """
    Model hybrydowy (Flatten + LSTM) do przewidywania redundantności okna ruchów.
    Nie modeluje zależności czasowych w sensie sekwencyjnym – całe okno
    jest najpierw spłaszczane, a następnie przetwarzane przez LSTM
    jako sekwencja długości 1. Model architektonicznie używa LSTM, ale nie wykorzystuje
    struktury czasowej okna – działa jak bogatsza, bramkowana wersja MLP.

    Zakłada takie samo wejście jak SimpleMLPNet:
    - states: tensor (B, T, 54) po zakodowaniu w CubeEncoder
    - moves:  tensor (B, W)  z indeksami ruchów

    Działanie:
    1. Każdy stan (54 naklejek) jest embedowany (6 kolorów → d_state).
    2. 54*d_state jest kompresowane do tokenu stanu.
    3. Wszystkie tokeny stanów w oknie są spłaszczane (flatten po czasie).
    4. Ruchy są embedowane i również spłaszczane.
    5. Całość jest rzutowana do przestrzeni rnn_hidden.
    6. Tak powstały wektor traktowany jest jako sekwencja długości 1 i przetwarzany przez LSTM.
    7. Hidden state LSTM trafia do klasyfikatora.
    """

    def __init__(
        self,
        move_vocab_size=18,
        d_state=16,
        d_move=32,
        rnn_hidden=512,
        rnn_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        # --- embeddings ---
        self.state_emb = nn.Embedding(6, d_state)
        self.move_emb = nn.Embedding(move_vocab_size, d_move)

        # pozycje stickerów
        self.sticker_pos = nn.Parameter(
            torch.randn(1, 1, 54, d_state) * 0.02
        )

        # kompresja 54*d_state -> d_state_tok
        d_state_tok = 64
        self.state_proj = nn.Sequential(
            nn.Linear(54 * d_state, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_state_tok),
            nn.ReLU(),
        )

        # wymiar po spłaszczeniu będzie dynamiczny -> używamy Lazy
        self.flatten_proj = nn.LazyLinear(rnn_hidden)

        # --- RNN jako "głębsza warstwa" ---
        self.rnn = nn.LSTM(
            input_size=rnn_hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # --- klasyfikator ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(rnn_hidden),
            nn.Linear(rnn_hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, states, moves):
        """
        states: (B, T+1, 54)
        moves:  (B, T)
        """

        B, T_plus_1, _ = states.shape

        # --- embedding stanów ---
        s = self.state_emb(states) + self.sticker_pos   # (B,T+1,54,d_state)
        s = s.reshape(B, T_plus_1, -1)                  # (B,T+1,54*d_state)
        s = self.state_proj(s)                          # (B,T+1,d_state_tok)

        # --- embedding ruchów ---
        m = self.move_emb(moves)                        # (B,T,d_move)

        # flatten dokładnie jak w MLP
        s_flat = s.reshape(B, -1)
        m_flat = m.reshape(B, -1)

        x = torch.cat([s_flat, m_flat], dim=1)         # (B, D_total)

        # projekcja do wymiaru RNN
        x = self.flatten_proj(x)                       # (B, rnn_hidden)

        # sztuczna sekwencja długości 1
        x = x.unsqueeze(1)                             # (B,1,rnn_hidden)

        out, (h_n, _) = self.rnn(x)

        last_hidden = h_n[-1]                          # (B,rnn_hidden)

        logits = self.classifier(last_hidden)          # (B,1)
        return logits
