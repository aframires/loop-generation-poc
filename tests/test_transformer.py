import pytest
import torch

from loop_generation_poc.models.components.transformer import (
    Transformer,
    TransformerBlock,
)

from loop_generation_poc.models.components.embeddings import (
    RotaryEmbedding,
    TimestepEmbedder,
    apply_rope,
)

from loop_generation_poc.models.components.attention import CrossAttention


def test_rotary_embedding_and_apply_rope() -> None:
    """
    Test that RoPE generates cos/sin correctly and applies them to 
    attention representations without altering magnitudes.
    """
    batch_size, seq_len, heads, head_dim = 2, 16, 4, 32
    
    # [Batch, NumHeads, SeqLen, HeadDim]
    q = torch.randn(batch_size, heads, seq_len, head_dim)
    
    rope = RotaryEmbedding(dim=head_dim)
    
    # cos, sin: [1, 1, SeqLen, HeadDim // 2]
    cos, sin = rope(seq_len)
    
    q_rot = apply_rope(q, cos, sin)
    
    assert q_rot.shape == q.shape
    
    # RoPE should strictly be a rotation, preserving the L2 norm
    assert torch.allclose(torch.norm(q, dim=-1), torch.norm(q_rot, dim=-1), atol=1e-4)


def test_timestep_embedder() -> None:
    """
    Test the timestep embedder mapping continuous timesteps to hidden dimension.
    """
    batch_size, hidden_dim = 2, 128
    
    # [Batch]
    t = torch.rand(batch_size)
    
    embedder = TimestepEmbedder(hidden_dim=hidden_dim)
    t_emb = embedder(t)
    
    # [Batch, HiddenDim]
    assert t_emb.shape == (batch_size, hidden_dim)


def test_cross_attention_shapes() -> None:
    """
    Test Cross-Attention block with differing sequence lengths for 
    latents and text embeddings.
    """
    batch_size, seq_len, text_seq_len, dim, text_dim = 2, 16, 8, 64, 768
    
    # [Batch, SeqLen, Dim]
    x = torch.randn(batch_size, seq_len, dim)
    # [Batch, TextSeqLen, TextDim]
    context = torch.randn(batch_size, text_seq_len, text_dim)
    
    cross_attn = CrossAttention(hidden_dim=dim, context_dim=text_dim, num_heads=4)
    out = cross_attn(x, context)
    
    assert out.shape == x.shape


def test_transformer_block() -> None:
    """
    Test a single Transformer block integrating AdaLN (via Mod), Self-Attention (with RoPE),
    Cross-Attention, and the MLP.
    """
    batch_size, seq_len, text_seq_len, dim, text_dim = 2, 16, 8, 64, 768
    num_heads = 4
    
    emb_dim = dim * 4

    # [Batch, SeqLen, Dim]
    x = torch.randn(batch_size, seq_len, dim)
    # [Batch, Dim]
    t_emb = torch.randn(batch_size, emb_dim)
    # [Batch, TextSeqLen, TextDim]
    context = torch.randn(batch_size, text_seq_len, text_dim)
    
    rope = RotaryEmbedding(dim=dim // num_heads)
    cos, sin = rope(seq_len)
    
    block = TransformerBlock(hidden_dim=dim, context_dim=text_dim, num_heads=num_heads,emb_dim=emb_dim)
    out = block(x, t_emb, context, cos, sin)
    
    assert out.shape == x.shape


@pytest.fixture
def transformer_model() -> Transformer:
    """
    Fixture to instantiate the backbone Transformer model.
    """
    return Transformer(
        latent_dim=64, 
        text_dim=768, 
        hidden_dim=128, 
        depth=2, 
        num_heads=4
    )


def test_transformer_end_to_end(transformer_model: Transformer) -> None:
    """Test the full Transformer backbone end-to-end to verify the strict shape contract."""
    batch_size, seq_len, latent_dim, text_seq_len, text_dim = 2, 16, 64, 8, 768
    
    x = torch.randn(batch_size, seq_len, latent_dim)         # [Batch, SeqLen, LatentDim]
    t = torch.rand(batch_size)                               # [Batch]
    context = torch.randn(batch_size, text_seq_len, text_dim)# [Batch, TextSeqLen, TextDim]
    
    out = transformer_model(x, t, context)
    
    assert out.shape == x.shape