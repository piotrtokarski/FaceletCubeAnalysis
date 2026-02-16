import torch
import torch.nn as nn

# Zastosowanie:
# model_factory = lambda: RecurrentNet(move_vocab_size=18)

class RecurrentNet(nn.Module):
    """
    Model sekwencyjny (LSTM) do przewidywania redundantności okna ruchów.
    Modeluje trajektorię stanów w czasie, a informację o ruchach
    traktuje jako dodatkowy kontekst.

    Zakłada takie samo wejście jak SimpleMLPNet:
    - states: tensor (B, T, 54) po zakodowaniu w CubeEncoder
    - moves:  tensor (B, W)  z indeksami ruchów

    Działanie:
    1. Każdy stan (54 naklejek) jest embedowany (6 kolorów → d_state).
    2. 54*d_state jest kompresowane do tokenu stanu.
    3. Sekwencja tokenów stanów trafia do dwukierunkowego RNN (LSTM lub GRU).
    4. Z wyjścia RNN budowana jest reprezentacja sekwencji:
           - ostatni krok (last hidden),
           - max-pooling po czasie,
       które są ze sobą łączone.
    5. Ruchy są embedowane i redukowane do jednej reprezentacji (np. ostatni ruch w oknie).
    6. Reprezentacja sekwencji stanów oraz reprezentacja ruchów są łączone i przekazywane do klasyfikatora.

    Model wykorzystuje pełną strukturę czasową stanów oraz globalną informację o całej trajektorii w oknie.
    Dwukierunkowe RNN pozwala analizować zależności zarówno w przód, jak i wstecz w obrębie okna.
    """

    def __init__(
        self,
        move_vocab_size: int = 18,
        d_state: int = 32,
        d_move: int = 32,
        rnn_hidden: int = 128,
        rnn_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Embedding naklejek stanu: mamy 6 kolorów (tak jak w SimpleMLPNet)
        self.state_emb = nn.Embedding(6, d_state)

        # Embeeding ruchów: słownik ruchów z CubeEncoder
        self.move_emb = nn.Embedding(move_vocab_size, d_move)

        # Kompresja 54*D_state do wektora tokena stanu (jak w SimpleMLPNet)
        d_state_token = 64
        self.state_proj = nn.Sequential(
            nn.Linear(54 * d_state, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_state_token),
            nn.ReLU(),
        )

        # RNN po sekwencji stanów
        rnn_input_dim = d_state_token

        self.rnn = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Prosty pooling po sekwencji: concat(last, max-pool)
        rnn_out_dim = 2 * rnn_hidden  # bidirectional
        pooled_dim = rnn_out_dim * 2  # last + max

        # Przetworzenie informacji o ruchach (skrót sekwencji ruchów)
        self.move_proj = nn.Sequential(
            nn.Linear(d_move, d_move),
            nn.ReLU(),
        )

        # Klasyfikator
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim + d_move, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),  # wyjście: logit
        )

    def forward(self, states, moves):
        """
        states: (B, T, 54) – zakodowane stany okna
        moves:  (B, W)     – indeksy ruchów okna
        """
        B, T, F = states.shape
        assert F == 54, "states powinno mieć wymiar 54 na ostatniej osi."

        # Zakładamy, że states to już ID kolorów 0..5 na 54 naklejkach
        s = self.state_emb(states)          # (B, T, 54, d_state)
        s = s.reshape(B, T, 54 * s.size(-1))  # (B, T, 54*d_state)
        s_tok = self.state_proj(s)          # (B, T, d_state_token)

        # Przepuszczamy sekwencję stanów przez RNN
        rnn_out, _ = self.rnn(s_tok)        # (B, T, 2*rnn_hidden)

        # Last hidden z ostatniego kroku (w obu kierunkach)
        last = rnn_out[:, -1, :]            # (B, 2*rnn_hidden)

        # Max-pooling po całym T
        max_pool, _ = torch.max(rnn_out, dim=1)  # (B, 2*rnn_hidden)

        seq_rep = torch.cat([last, max_pool], dim=-1)  # (B, 4*rnn_hidden)

        # Embedding tylko ostatniego ruchu w oknie
        m = self.move_emb(moves)            # (B, W, d_move)
        m_last = m[:, -1, :]                # (B, d_move)
        m_rep = self.move_proj(m_last)      # (B, d_move)

        # Połączenie reprezentacji sekwencji stanów + ruchów
        x = torch.cat([seq_rep, m_rep], dim=-1)  # (B, 4*rnn_hidden + d_move)

        logits = self.classifier(x)         # (B, 1)
        return logits
