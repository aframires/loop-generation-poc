from typing import Tuple
import math


import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    """
    Generates rotary positional embeddings (RoPE) for sequence tokens.
    Computes frequencies dynamically based on sequence length to allow extrapolation.
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        # persistent=False means it won't be saved in the state_dict, but still moves to GPU
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # [SeqLen]
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # [SeqLen, Dim // 2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # [1, 1, SeqLen, Dim // 2]
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to queries or keys.
    x: [Batch, NumHeads, SeqLen, HeadDim]
    cos, sin: [1, 1, SeqLen, HeadDim // 2]
    """
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    rotated = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return rotated


class SinusoidalEmbedding(nn.Module):
    """
    Standard sinusoidal continuous time embedding for diffusion/flow timesteps.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [Batch]
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        
        # [Batch, Dim // 2]
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        # [Batch, Dim]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class TimestepEmbedder(nn.Module):
    """
    Combines the SinusoidalEmbedding with an MLP.
    """
    def __init__(self, hidden_dim: int, frequency_embedding_dim: int = 256):
        super().__init__()
        self.sinusoidal = SinusoidalEmbedding(frequency_embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [Batch] -> output: [Batch, HiddenDim]
        x = self.sinusoidal(t)
        return self.mlp(x)