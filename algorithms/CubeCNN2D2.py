# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 15:14:24 2026

@author: mp
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================
# Residual Block 2D
# ==========================
class ResBlock2D(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


# ==========================
# Cube 2D CNN
# ==========================
class CubeCNN2D(nn.Module):
    def __init__(
        self,
        move_vocab_size=18,
        d_embed=8,        # embedding koloru
        base_channels=64,
        num_res_blocks=3,
        dropout=0.2,
    ):
        super().__init__()

        # ===== Embedding kolorów =====
        self.color_emb = nn.Embedding(6, d_embed)
        self.move_emb = nn.Embedding(move_vocab_size, 32)

        in_channels = 6 * d_embed

        # ===== Przestrzenny CNN =====
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU()
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock2D(base_channels) for _ in range(num_res_blocks)]
        )

        # ===== Agregacja czasu (Conv1D) =====
        self.time_conv = nn.Sequential(
            nn.Conv1d(base_channels, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.GELU()
        )

        # ===== Head =====
        self.head = nn.Sequential(
            nn.Linear(128 + 32, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, states, moves):
        # states: (B,T,54)
        # moves:  (B,W)

        B, T, _ = states.shape

        # =====================
        # 1️⃣ Reshape do (6,3,3)
        # =====================
        s = states.view(B, T, 6, 3, 3)

        # =====================
        # 2️⃣ Embedding kolorów
        # =====================
        s = self.color_emb(s)                # (B,T,6,3,3,d_embed)
        s = s.permute(0,1,2,5,3,4)           # (B,T,6,d_embed,3,3)
        s = s.reshape(B*T, -1, 3, 3)         # (B*T, 6*d_embed,3,3)

        # =====================
        # 3️⃣ Spatial CNN
        # =====================
        s = self.input_conv(s)
        s = self.res_blocks(s)

        # global average pooling
        s = F.adaptive_avg_pool2d(s, 1)
        s = s.view(B, T, -1)                 # (B,T,base_channels)

        # =====================
        # 4️⃣ Time Conv
        # =====================
        s = s.permute(0,2,1)                 # (B,C,T)
        s = self.time_conv(s)

        s = F.adaptive_avg_pool1d(s, 1)
        s = s.squeeze(-1)                    # (B,128)

        # =====================
        # 5️⃣ Move embedding
        # =====================
        m = self.move_emb(moves)             # (B,W,32)
        m = m.mean(dim=1)                    # (B,32)

        # =====================
        # 6️⃣ Head
        # =====================
        x = torch.cat([s, m], dim=1)

        return self.head(x)
