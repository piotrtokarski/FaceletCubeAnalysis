import torch
import torch.nn as nn

# Zastosowanie:
# model_factory=lambda: SequenceRNNNet(move_vocab_size=18)

class SequenceRNNNet(nn.Module):
    """
    Model sekwencyjny (LSTM) do przewidywania redundantności okna ruchów.
    Modeluje zależności czasowe pomiędzy kolejnymi stanami i ruchami.

    Zakłada takie samo wejście jak SimpleMLPNet:
    - states: tensor (B, T+1, 54) po zakodowaniu w CubeEncoder
    - moves:  tensor (B, W)      z indeksami ruchów

    Działanie:
    1. Każdy stan (54 naklejek) jest embedowany (6 kolorów → d_state).
    2. 54*d_state jest kompresowane do tokenu stanu.
    3. Każdy ruch jest embedowany do wektora d_move.
    4. Dla każdego kroku czasu tworzony jest token:
           [state_t, move_t]
       (dla ostatniego stanu stosowany jest padding ruchu zerem).
    5. Sekwencja tokenów trafia do LSTM.
    6. Ostatni hidden state jest używany do klasyfikacji.

    Model wykorzystuje pełną strukturę czasową okna i pozwala
    uczyć się zależności między kolejnymi stanami oraz ruchami.
    """
    def __init__(
        self,
        move_vocab_size=18,
        d_state=32,
        d_move=32,
        d_state_token=128,
        rnn_hidden=256,
        rnn_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        # --- embeddings ---
        self.state_emb = nn.Embedding(6, d_state)
        self.move_emb = nn.Embedding(move_vocab_size, d_move)

        # pozycja stickerów
        self.sticker_pos = nn.Parameter(torch.randn(1, 1, 54, d_state) * 0.02)

        # kompresja 54*d_state -> d_state_token
        self.state_proj = nn.Sequential(
            nn.Linear(54 * d_state, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_state_token),
            nn.ReLU(),
        )

        # rozmiar tokenu czasowego
        self.time_token_dim = d_state_token + d_move

        # --- RNN ---
        self.rnn = nn.LSTM(
            input_size=self.time_token_dim,
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
        T = moves.shape[1]

        # --- embedding stanów ---
        s = self.state_emb(states) + self.sticker_pos  # (B,T+1,54,d_state)
        s = s.reshape(B, T_plus_1, -1)                 # (B,T+1,54*d_state)
        s = self.state_proj(s)                         # (B,T+1,d_state_token)

        # --- embedding ruchów ---
        m = self.move_emb(moves)                       # (B,T,d_move)

        # dopasowanie czasowe:
        # dla ostatniego stanu nie ma ruchu -> padding zerowy
        zero_move = torch.zeros(B, 1, m.size(-1), device=m.device)
        m_padded = torch.cat([m, zero_move], dim=1)    # (B,T+1,d_move)

        # --- token czasowy ---
        x = torch.cat([s, m_padded], dim=2)            # (B,T+1,time_token_dim)

        # --- RNN ---
        out, (h_n, c_n) = self.rnn(x)

        # użycie ostatniego hidden state
        last_hidden = h_n[-1]                          # (B,rnn_hidden)

        logits = self.classifier(last_hidden)          # (B,1)
        return logits
