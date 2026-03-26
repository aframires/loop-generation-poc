from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from loop_generation_poc.models.diffusion import ConditionalDiffusion, LinearWarmupCosineAnnealingLR


def test_linear_warmup_cosine_annealing_lr() -> None:
    """
    Test the custom learning rate scheduler for correct linear warmup 
    and cosine annealing decay logic.
    """
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer, warmup_steps=10, max_steps=100, eta_min=0.0
    )
    
    # Initial LR during warmup (epoch 0)
    assert scheduler.get_lr()[0] == 0.0
    
    # Halfway through warmup
    scheduler.last_epoch = 5
    assert scheduler.get_lr()[0] == pytest.approx(0.05)
    
    # End of warmup (peak LR)
    scheduler.last_epoch = 10
    assert scheduler.get_lr()[0] == pytest.approx(0.1)
    
    # End of training (minimum LR)
    scheduler.last_epoch = 100
    assert scheduler.get_lr()[0] == pytest.approx(0.0)


@pytest.fixture
def mock_backbone() -> Any:
    """
    Creates a mock backbone network to test the Diffusion LightningModule independently.
    """
    backbone = MagicMock()
    backbone.return_value = torch.randn(2, 16, 64)
    return backbone


@pytest.fixture
def dummy_batch() -> Dict[str, torch.Tensor]:
    """
    Returns a dummy batch matching the datamodule contract defined in agents.md.
    """
    # [Batch, SeqLen, LatentDim]
    latents = torch.randn(2, 16, 64)
    # [Batch, ContextSeqLen, ContextDim]
    text_embeds = torch.randn(2, 8, 768)
    return {"latents": latents, "text_embeds": text_embeds}


def test_conditional_diffusion_init(mock_backbone: Any) -> None:
    """Test LightningModule initialization and hyperparameter saving."""
    model = ConditionalDiffusion(
        backbone=mock_backbone,
        optimizer=MagicMock(),
        loss_fn=nn.MSELoss(),
        diffusion_objective="v_prediction"
    )
    assert model.hparams.diffusion_objective == "v_prediction"


def test_conditional_diffusion_forward(mock_backbone: Any) -> None:
    """Test the raw forward pass routes correctly to the underlying net."""
    model = ConditionalDiffusion(
        backbone=mock_backbone,
        optimizer=MagicMock(),
        loss_fn=nn.MSELoss()
    )
    
    # [Batch, SeqLen, LatentDim]
    x = torch.randn(2, 16, 64)
    # [Batch]
    t = torch.rand(2)
    # [Batch, ContextSeqLen, ContextDim]
    context = torch.randn(2, 8, 768)
    
    out = model(x, t, context)
    
    # Output should match the patched Transformer's mock output shape
    assert out.shape == (2, 16, 64)


def test_conditional_diffusion_v_prediction_step(mock_backbone: Any, dummy_batch: Dict[str, torch.Tensor]) -> None:
    """Test training_step correctly computes loss for the v_prediction objective."""
    model = ConditionalDiffusion(
        backbone=mock_backbone,
        optimizer=MagicMock(),
        loss_fn=nn.MSELoss(),
        diffusion_objective="v_prediction"
    )
    
    # Ensure the objective runs end-to-end without shape errors and returns a scalar loss
    loss = model.training_step(dummy_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_conditional_diffusion_rectified_flow_step(mock_backbone: Any, dummy_batch: Dict[str, torch.Tensor]) -> None:
    """Test training_step correctly computes loss for the rectified_flow objective."""
    model = ConditionalDiffusion(
        backbone=mock_backbone,
        optimizer=MagicMock(),
        loss_fn=nn.MSELoss(),
        diffusion_objective="rectified_flow"
    )
    
    loss = model.training_step(dummy_batch, batch_idx=0)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0


def test_conditional_diffusion_invalid_objective(mock_backbone: Any, dummy_batch: Dict[str, torch.Tensor]) -> None:
    """Ensure the module strictly validates the supported objectives."""
    model = ConditionalDiffusion(
        backbone=mock_backbone,
        optimizer=MagicMock(),
        loss_fn=nn.MSELoss(),
        diffusion_objective="invalid_objective"
    )
    
    with pytest.raises(ValueError, match="Unknown diffusion_objective"):
        model.training_step(dummy_batch, batch_idx=0)


def test_configure_optimizers_no_scheduler(mock_backbone: Any) -> None:
    """Test Hydra optimizer mapping without a learning rate scheduler."""
    model = ConditionalDiffusion(
        backbone=mock_backbone,
        optimizer=MagicMock(),
        loss_fn=nn.MSELoss()
    )
    opt_config = model.configure_optimizers()
    assert "optimizer" in opt_config
    assert "lr_scheduler" not in opt_config


def test_configure_optimizers_with_scheduler(mock_backbone: Any) -> None:
    """Test Hydra optimizer mapping with a learning rate scheduler."""
    model = ConditionalDiffusion(
        backbone=mock_backbone,
        optimizer=MagicMock(),
        loss_fn=nn.MSELoss(),
        scheduler=MagicMock()
    )
    opt_config = model.configure_optimizers()
    assert "lr_scheduler" in opt_config
    assert opt_config["lr_scheduler"]["interval"] == "step"
