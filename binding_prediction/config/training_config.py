import typing as tp
from dataclasses import dataclass

import yaml


@dataclass
class TrainingConfig:
    early_stopping_rounds: int
    target_scale_pos_weight: float = -1
    train_size: int = -1


def load_training_config_from_yaml_path(yaml_path: str, max_train_size: int) -> TrainingConfig:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return create_training_config_from_dict(config, max_train_size)


def create_training_config_from_dict(config: dict, max_train_size: int) -> TrainingConfig:
    training_config_dict = config["train"]
    training_config_dict["train_size"] = min(max_train_size, training_config_dict["train_size"])
    config = TrainingConfig(**training_config_dict)
    return config

