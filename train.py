import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from hydra.utils import instantiate

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set the seed for pt,np and random
    pl.seed_everything(cfg.seed, workers=True)

    # Instantiate DataModule
    print(f"Instantiating datamodule {cfg.data._target_}")
    datamodule: pl.LightningDataModule = instantiate(cfg.data)

    # Instantiate Model
    print(f"Instantiating model {cfg.model._target_}")
    model: pl.LightningModule = instantiate(cfg.model)

    # Instantiate Callbacks
    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if cb_conf is not None:
                callbacks.append(instantiate(cb_conf))

    # Instantiate Logger
    logger = None
    if "logger" in cfg:
        logger = instantiate(cfg.logger)

    # Instantiate Trainer
    print(f"Instantiating trainer {cfg.trainer._target_}")
    trainer: pl.Trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # 7. Start Training!
    print("Starting training")
    trainer.fit(model=model, datamodule=datamodule)

if __name__ == "__main__":
    main()