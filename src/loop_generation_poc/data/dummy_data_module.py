from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Subset

class DummyLoopDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    A minimal dataset generating random tensors to simulate pre-computed
    audio latents and text embeddings. T5 dimensions are used as default.
    """
    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        latent_dim: int,
        text_seq_len: int,
        text_dim: int,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.text_seq_len = text_seq_len
        self.text_dim = text_dim
        
        # Pre-compute tensors so the model sees the same dummy data every epoch.
        self.latents_data = torch.randn(num_samples, seq_len, latent_dim)
        self.text_embeds_data = torch.randn(num_samples, text_seq_len, text_dim)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "latents": self.latents_data[idx],
            "text_embeds": self.text_embeds_data[idx]
        }

class DummyDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the DummyLoopDataset.
    Exposes parameters for Hydra configuration and handles dataloaders.
    """
    def __init__(
        self,
        batch_size: int = 16,
        num_samples: int = 1000,
        train_val_split: float = 0.8,  # percentage of training data
        seq_len: int = 256,
        latent_dim: int = 64,
        text_seq_len: int = 77,  # Parameters for the t5-base used in SAO
        text_dim: int = 768,
        num_workers: int = 0,
        pin_memory: bool = True, # makes it faster for the data to go from CPU to GPU
    ) -> None:
        super().__init__()
        if not 0.0 < train_val_split < 1.0:
            raise ValueError("train_val_split must be between 0 and 1 (exclusive).")

        # This saves all __init__ args to self.hparams
        self.save_hyperparameters()
        self.train_dataset: Optional[Dataset[Dict[str, torch.Tensor]]] = None
        self.val_dataset: Optional[Dataset[Dict[str, torch.Tensor]]] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            full_dataset = DummyLoopDataset(
                num_samples=self.hparams.num_samples,
                seq_len=self.hparams.seq_len,
                latent_dim=self.hparams.latent_dim,
                text_seq_len=self.hparams.text_seq_len,
                text_dim=self.hparams.text_dim,
            )

            # Use the full dataset for training
            self.train_dataset = full_dataset
            
            # Validation is a subset of the training data to evaluate memorization
            val_size = max(1, self.hparams.num_samples - int(self.hparams.train_val_split * self.hparams.num_samples))
            self.val_dataset = Subset(full_dataset, indices=list(range(val_size)))

    def train_dataloader(self) -> DataLoader[Dict[str, torch.Tensor]]:
        """
        Returns the training dataloader.
        Expected batch shapes yielded:
        - "latents":   [batch_size, seq_len, latent_dim]
        - "text_embeds": [batch_size, text_seq_len, text_dim]
        """
        if self.train_dataset is None:
            raise RuntimeError("DummyDataModule.setup() must be called before train_dataloader().")

        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory, 
        )

    def val_dataloader(self) -> DataLoader[Dict[str, torch.Tensor]]:
        """
        Returns the validation dataloader.
        Expected batch shapes yielded:
        - "latents":   [batch_size, seq_len, latent_dim]
        - "text_embeds": [batch_size, text_seq_len, text_dim]
        """
        if self.val_dataset is None:
            raise RuntimeError("DummyDataModule.setup() must be called before val_dataloader().")

        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
