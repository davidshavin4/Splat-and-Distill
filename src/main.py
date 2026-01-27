import os
from pathlib import Path
import warnings

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.dataset.utils import expand_tensors
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper
    from src.model.vit import get_vit

seed_everything(42, workers=True)

def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


class ValidateEveryNSteps(pl.Callback):
    def __init__(self, every_n_steps: int):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.global_validation_step = 0
        self._last_validated = 0        

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called at the end of every training batch."""
        if trainer.global_step % self.every_n_steps == 1 and self._last_validated<trainer.global_step:# and trainer.global_step > 0:
            scene_idx = int(self.global_validation_step / len(trainer.datamodule.val_dataloader().dataset.dataset))
            print(f"validation step {self.global_validation_step}, scene number {scene_idx}")
            # batch = next(trainer.datamodule.val_dataloader().dataset.dataset.__iter__())            
            try:
                # Try to get the batch via __getitem__
                batch = trainer.datamodule.val_dataloader().dataset.dataset[scene_idx]
            except:
                try:
                    # Fallback to iterator-based retrieval
                    batch = next(iter(trainer.datamodule.val_dataloader().dataset.dataset))
                except Exception as e:
                    raise RuntimeError("Could not retrieve batch from dataset using either __getitem__ or iterator") from e            
            batch = expand_tensors(batch, device='cuda')
            self.global_validation_step += 1
            self._last_validated = trainer.global_step
            # trainer.validate(model=pl_module, ckpt_path=None)

            pl_module.validation_step(batch, scene_idx)


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))
    latest_run = output_dir.parents[1] / "latest-run"
    os.system(f"rm {latest_run}")
    os.system(f"ln -s {output_dir} {latest_run}")

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,
                                       'resume': "must"})
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",  # save the lastest k ckpt, can do offline test later
        )
    )
    callbacks.append(ValidateEveryNSteps(every_n_steps=2500))
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()
    
    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        # val_check_interval=cfg.trainer.val_check_interval,
        limit_val_batches=0,  # ✅ Disables automatic validation
        enable_progress_bar=cfg.mode == "test",
        # gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        # deterministic=True,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_visualizer,
        "decoder": get_decoder(cfg.model.decoder, cfg.dataset),
        "vit": get_vit(cfg.model.vit),
        "losses": get_losses(cfg.loss),
        "step_tracker": step_tracker,
    }
    if cfg.mode == "train" and checkpoint_path is not None and not cfg.checkpointing.resume:
        # Just load model weights, without optimizer states
        # e.g., fine-tune from the released weights on other datasets
        model_wrapper = ModelWrapper.load_from_checkpoint(
            checkpoint_path, **model_kwargs, strict=False)
        print(cyan(f"Loaded weigths from {checkpoint_path}."))
    else:
        model_wrapper = ModelWrapper(**model_kwargs)

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=(
            checkpoint_path if cfg.checkpointing.resume else None))
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    train()
