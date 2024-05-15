import typing as tp
from dataclasses import dataclass

import yaml


@dataclass
class TrainingConfig:
    early_stopping_rounds: int
    target_scale_pos_weight: float = -1
    train_size: int = -1


def load_training_config_from_yaml_path(yaml_path: str) -> TrainingConfig:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return create_training_config_from_dict(config)


def create_training_config_from_dict(config: dict) -> TrainingConfig:
    training_config_dict = config["train"]
    config = TrainingConfig(**training_config_dict)
    return config

