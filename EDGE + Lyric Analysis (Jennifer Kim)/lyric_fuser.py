import torch
import torch.nn as nn

class LyricFuser(nn.Module):
    def __init__(self, lyric_dim=384, music_dim=4800):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(lyric_dim, music_dim),
            nn.GELU(),
            nn.Linear(music_dim, music_dim)
        )
        # Gate starts at 0 so EDGE behaves exactly as original until you tune it
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, music_emb, lyric_emb):
        lyric_proj = self.proj(lyric_emb)
        return music_emb + torch.sigmoid(self.gate) * lyric_proj