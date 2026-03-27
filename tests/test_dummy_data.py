from typing import Dict

import pytest
import torch
from torch.utils.data import DataLoader

from loop_generation_poc.data.dummy_data_module import DummyDataModule


@pytest.fixture
def data_module() -> DummyDataModule:
    """
    Fixture to instantiate the DummyDataModule with standard dimensions
    to verify the batch schema defined in the architectural guidelines.
    """
    return DummyDataModule(
        batch_size=4,
        seq_len=16,
        latent_dim=64,
        text_seq_len=8,
        text_dim=768,
        num_samples=20,
    )


def test_dummy_data_module_init(data_module: DummyDataModule) -> None:
    """Test initialization and hyperparameter saving of the DataModule."""
    assert data_module.hparams.batch_size == 4
    assert data_module.hparams.seq_len == 16
    assert data_module.hparams.latent_dim == 64
    assert data_module.hparams.text_seq_len == 8
    assert data_module.hparams.text_dim == 768


def test_dummy_data_module_dataloaders(data_module: DummyDataModule) -> None:
    """Test that dataloaders are correctly instantiated."""
    data_module.setup(stage="fit")
    
    train_dl = data_module.train_dataloader()
    val_dl = data_module.val_dataloader()
    
    assert isinstance(train_dl, DataLoader)
    assert isinstance(val_dl, DataLoader)


def test_dummy_batch_schema(data_module: DummyDataModule) -> None:
    """
    Test that yielded batches adhere strictly to the shapes of t5 text encoder and audio embeddings.
    Expects 'latents' and 'text_embeds' with proper shapes and dtypes.
    """
    data_module.setup(stage="fit")
    train_dl = data_module.train_dataloader()
    
    batch: Dict[str, torch.Tensor] = next(iter(train_dl))
    
    assert "latents" in batch
    assert "text_embeds" in batch
    
    # [Batch, SeqLen, LatentDim]
    assert batch["latents"].shape == (4, 16, 64)
    
    # [Batch, TextSeqLen, TextDim]
    assert batch["text_embeds"].shape == (4, 8, 768)
