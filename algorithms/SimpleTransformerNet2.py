import torch
import torch.nn as nn


class SimpleTransformerNet2(nn.Module):
    def __init__(
        self,
        move_vocab_size=18,
        d_state=16,
        d_move=32,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_window=20,
    ):
        super().__init__()

        # === Embedding stickerów (6 kolorów) ===
        self.state_emb = nn.Embedding(6, d_state)

        # embedding pozycji stickerów (54)
        self.sticker_pos = nn.Parameter(
            torch.randn(1, 1, 54, d_state) * 0.02
        )

        # kompresja 54*d_state -> d_model
        self.state_proj = nn.Sequential(
            nn.Linear(54 * d_state, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, d_model),
        )

        # === Embedding ruchów ===
        self.move_emb = nn.Embedding(move_vocab_size, d_move)

        self.move_proj = nn.Linear(d_move, d_model)

        # === Pozycje czasowe ===
        self.time_pos_emb = nn.Parameter(
            torch.randn(1, max_window, d_model) * 0.02
        )

        # === Transformer encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # === Head klasyfikacyjny ===
        self.cls_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, states, moves):
        """
        states: (B, T, 54)
        moves:  (B, W)  gdzie W = T-1
        """

        B, T, _ = states.shape

        # ===== 1. State embedding =====
        s = self.state_emb(states) + self.sticker_pos  # (B,T,54,d_state)
        s = s.reshape(B, T, -1)                        # (B,T,54*d_state)
        s = self.state_proj(s)                         # (B,T,d_model)

        # ===== 2. Move embedding =====
        # dodajemy padding ruchu dla pierwszego stanu
        if moves is not None:
            m = self.move_emb(moves)                   # (B,T-1,d_move)
            m = self.move_proj(m)                      # (B,T-1,d_model)

            pad = torch.zeros(B, 1, m.size(-1), device=m.device)
            m = torch.cat([pad, m], dim=1)             # (B,T,d_model)
        else:
            m = torch.zeros_like(s)

        # ===== 3. Token = state + move =====
        x = s + m

        # ===== 4. Dodaj pozycje czasowe =====
        x = x + self.time_pos_emb[:, :T, :]

        # ===== 5. Transformer =====
        x = self.transformer(x)  # (B,T,d_model)

        # ===== 6. Klasyfikujemy ostatni token =====
        last_token = x[:, -1, :]  # (B,d_model)

        return self.cls_head(last_token)
