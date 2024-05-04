from dataclasses import dataclass

import yaml

from binding_prediction.utils import FeaturizerTypes, ModelTypes


@dataclass
class XGBoostModelConfig:
    name: str
    max_depth: int
    objective: str
    eval_metric: str
    verbosity: int
    nthread: int
    tree_method: str
    grow_policy: str
    subsample: float
    colsample_bytree: float
    num_boost_round: int
    scale_pos_weight: float


def load_xgboost_model_config_from_yaml(yaml_path: str,
                                        scale_pos_weight=1.0) -> XGBoostModelConfig:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)["model"]
    name = config['name']
    if name not in ModelTypes.__dict__.values():
        raise ValueError(f"Model {name} is not supported")
    config['scale_pos_weight'] = scale_pos_weight
    return XGBoostModelConfig(**config)


@dataclass
class FeaturizerConfig:
    name: str
    radius: int
    length: int


def load_featurizer_config_from_yaml(yaml_path: str) -> FeaturizerConfig:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)["featurizer"]

    name = config['name']
    if name not in FeaturizerTypes.__dict__.values():
        raise ValueError(f"Featurizer {name} is not supported")
    return FeaturizerConfig(**config)


@dataclass
class TrainingConfig:
    early_stopping_rounds: int


def load_training_config_from_yaml(yaml_path: str) -> TrainingConfig:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)["train"]
    return TrainingConfig(**config)


@dataclass
class Config:
    train_file_path: str
    test_file_path: str
    logs_dir: str
    neg_samples: int
    pos_samples: int
    featurizer_config: FeaturizerConfig
    model_config: XGBoostModelConfig
    training_config: TrainingConfig
    protein_map_path: str = None


def create_training_config(train_file_path: str, test_file_path: str,
                           logs_dir: str,
                           neg_samples: int, pos_samples: int,
                           config_yaml_path: str) -> Config:
    featurizer_config = load_featurizer_config_from_yaml(config_yaml_path)
    xgboost_model_config = load_xgboost_model_config_from_yaml(config_yaml_path,
                                                               scale_pos_weight=neg_samples / pos_samples)
    training_config = load_training_config_from_yaml(config_yaml_path)
    return Config(train_file_path=train_file_path, test_file_path=test_file_path,
                  logs_dir=logs_dir,
                  neg_samples=neg_samples, pos_samples=pos_samples,
                  featurizer_config=featurizer_config,
                  model_config=xgboost_model_config,
                  training_config=training_config)
