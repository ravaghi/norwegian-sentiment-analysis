from lightning.pytorch import seed_everything
from hydra import initialize, compose
from omegaconf import DictConfig
from omegaconf import OmegaConf
import wandb
import os


def init_run(config_name: str, run_name: str) -> DictConfig:
    with initialize(version_base=None, config_path="../configs"):
        config = compose(config_name)
        
    if not os.path.exists(config.general.log_dir):
        os.makedirs(config.general.log_dir)

    seed_everything(config.general.seed)
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        config=OmegaConf.to_container(config, resolve=True),
        name=run_name,
        dir=config.general.log_dir
    )

    print("-" * 30 + " config " + "-" * 30)
    print(OmegaConf.to_yaml(config))
    print("-" * 30 + " config " + "-" * 30)

    return config
