# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 14:23:31 2026

@author: mp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CubeCNN1D(nn.Module):
    def __init__(
        self,
        move_vocab_size=18,
        d_embed=32,
        d_state=64,
        d_time=128,
        dropout=0.2,
    ):
        super().__init__()

        # ===== Embeddings =====
        self.state_emb = nn.Embedding(6, d_embed)
        self.move_emb = nn.Embedding(move_vocab_size, d_embed)

        # ===== CNN po stickerach =====
        self.sticker_conv = nn.Sequential(
            nn.Conv1d(d_embed, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(64, d_state, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # ===== CNN po czasie =====
        self.time_conv = nn.Sequential(
            nn.Conv1d(d_state, d_time, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_time, d_time, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # ===== Głowa =====
        self.head = nn.Sequential(
            nn.Linear(d_time + d_embed, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, states, moves):
        # states: (B,T,54)
        # moves:  (B,W)

        B, T, _ = states.shape

        # =====================
        # 1️⃣ Sticker CNN
        # =====================
        s = self.state_emb(states)          # (B,T,54,d_embed)

        s = s.view(B*T, 54, -1)             # (B*T,54,d)
        s = s.permute(0, 2, 1)              # (B*T,d,54)

        s = self.sticker_conv(s)            # (B*T,d_state,54)

        s = F.adaptive_avg_pool1d(s, 1)     # global pooling
        s = s.squeeze(-1)                   # (B*T,d_state)

        s = s.view(B, T, -1)                # (B,T,d_state)

        # =====================
        # 2️⃣ Time CNN
        # =====================
        s = s.permute(0, 2, 1)              # (B,d_state,T)

        s = self.time_conv(s)               # (B,d_time,T)

        s = F.adaptive_avg_pool1d(s, 1)     # (B,d_time,1)
        s = s.squeeze(-1)                   # (B,d_time)

        # =====================
        # 3️⃣ Move embedding
        # =====================
        m = self.move_emb(moves)            # (B,W,d_embed)
        m = m.mean(dim=1)                   # średnia ruchów → (B,d_embed)

        # =====================
        # 4️⃣ Final MLP
        # =====================
        x = torch.cat([s, m], dim=1)

        return self.head(x)
