import torch
import torch.nn as nn

from loop_generation_poc.models.components.embeddings import RotaryEmbedding, TimestepEmbedder
from loop_generation_poc.models.components.attention import CrossAttention, SelfAttentionWithRoPE

class AdaLNZero(nn.Module):
    """
    Predicts the scale, shift, and gate parameters for both the 
    Self-Attention and MLP blocks from the timestep embedding.
    """
    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(emb_dim, 6 * hidden_dim)
        
        # Initialize linear layer to output zeros.
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, t_emb: torch.Tensor) -> tuple:
        """
        Returns 6 chunks of shape [Batch, 1, hidden_dim]
        """
        # [Batch, emb_dim] -> [Batch, 6 * hidden_dim]
        emb_out = self.linear(self.silu(t_emb))
        
        # Add sequence dimension: [Batch, 1, 6 * hidden_dim]
        emb_out = emb_out.unsqueeze(1)
        
        # Return unpacked tuple of 6 tensors
        return emb_out.chunk(6, dim=-1)
    
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Applies AdaLN scaling and shifting."""
    return x * (1 + scale) + shift

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, context_dim: int, num_heads: int, emb_dim: int):
        super().__init__()
        # 1. Global AdaLNZero Conditioning (Timestep)
        self.adaln = AdaLNZero(emb_dim, hidden_dim)
        
        # 2. Self-Attention (Relative Temporal Conditioning)
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = SelfAttentionWithRoPE(hidden_dim, num_heads)
        
        # 3. Cross-Attention (Text Conditioning)
        self.norm2 = nn.LayerNorm(hidden_dim) # Standard LayerNorm for context
        self.cross_attn = CrossAttention(hidden_dim, context_dim, num_heads)
        
        # 4. Feed Forward
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(
        self, 
        x: torch.Tensor, 
        t_emb: torch.Tensor, 
        context: torch.Tensor, 
        cos: torch.Tensor, 
        sin: torch.Tensor
    ) -> torch.Tensor:
        
        # 1. Generate all global conditioning parameters for this layer
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln(t_emb)
        
        # 2. Self-Attention with AdaLN and gating
        x_mod = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa * self.attn(x_mod, cos, sin)
        
        # 3. Text Cross-Attention 
        # (Standard residual connection, no timestep gating needed here)
        x = x + self.cross_attn(self.norm2(x), context)
        
        # 4. MLP with AdaLN and gating
        x_mod = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_mod)
        
        return x


class Transformer(nn.Module):
    """
    Main Transformer backbone.
    Instantiates the independent embedder classes and passes their outputs through the blocks.
    """
    def __init__(
        self, 
        latent_dim: int = 64,
        text_dim: int = 768, 
        hidden_dim: int = 256, 
        depth: int = 4, 
        num_heads: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Standard DiT 4x expansion for the timestep embedding
        self.emb_dim = hidden_dim * 4 
        
        self.latent_proj_in = nn.Linear(latent_dim, hidden_dim)
        self.latent_proj_out = nn.Linear(hidden_dim, latent_dim)
        
        # Timestep Embedder and Positional Embedder
        self.time_embedder = TimestepEmbedder(self.emb_dim) 
        self.rope_embedder = RotaryEmbedding(dim=hidden_dim // num_heads)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, text_dim, num_heads, self.emb_dim) 
            for _ in range(depth)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.emb_dim, 2 * hidden_dim) 
        )
        
        # Zero-initialize the final AdaLNZero layer
        nn.init.zeros_(self.final_adaln[-1].weight)
        nn.init.zeros_(self.final_adaln[-1].bias)
    
    def forward(
        self, 
        latents: torch.Tensor, 
        timesteps: torch.Tensor, 
        text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        latents:     [Batch, SeqLen, LatentDim]
        timesteps:   [Batch]
        text_embeds: [Batch, TextSeqLen, TextDim]
        """
        B, S, _ = latents.shape
        
        x = self.latent_proj_in(latents)
        t_emb = self.time_embedder(timesteps)
        
        cos, sin = self.rope_embedder(seq_len=S)
        
        for block in self.blocks:
            x = block(x, t_emb, text_embeds, cos, sin)
            
        shift, scale = self.final_adaln(t_emb).unsqueeze(1).chunk(2, dim=-1)
        
        x = modulate(self.final_norm(x), shift, scale)
        
        return self.latent_proj_out(x)