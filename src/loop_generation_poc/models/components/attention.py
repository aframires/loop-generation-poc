import torch
import torch.nn as nn
import torch.nn.functional as F

from loop_generation_poc.models.components.embeddings import apply_rope
class SelfAttentionWithRoPE(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [Batch, SeqLen, HiddenDim]
        B, S, C = x.shape
        
        # [Batch, SeqLen, 3 * HiddenDim]
        qkv = self.qkv(x)
        
        # [3, Batch, NumHeads, SeqLen, HeadDim]
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # [Batch, NumHeads, SeqLen, HeadDim]
        x_attn = F.scaled_dot_product_attention(q, k, v)
        
        # [Batch, SeqLen, HiddenDim]
        x_attn = x_attn.permute(0, 2, 1, 3).reshape(B, S, C)
        
        return self.proj(x_attn)


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim: int, context_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kv = nn.Linear(context_dim, hidden_dim * 2, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [Batch, SeqLen, HiddenDim]
        # context: [Batch, ContextSeqLen, ContextDim]
        B, S, C = x.shape
        _, S_ctx, _ = context.shape
        
        # [Batch, NumHeads, SeqLen, HeadDim]
        q = self.q(x).view(B, S, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # [2, Batch, NumHeads, ContextSeqLen, HeadDim]
        kv = self.kv(context).view(B, S_ctx, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # [Batch, NumHeads, SeqLen, HeadDim]
        x_attn = F.scaled_dot_product_attention(q, k, v)
        
        # [Batch, SeqLen, HiddenDim]
        x_attn = x_attn.permute(0, 2, 1, 3).reshape(B, S, C)
        
        return self.proj(x_attn)