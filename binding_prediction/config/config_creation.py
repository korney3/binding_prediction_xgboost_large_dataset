from dataclasses import dataclass

import yaml

from binding_prediction.config.featurizer_config import CircularFingerprintFeaturizerConfig, \
    load_circular_fingerprint_featurizer_config_from_yaml, load_maccs_fingerprint_featurizer_config_from_yaml, \
    MACCSFingerprintFeaturizerConfig
from binding_prediction.config.model_config import XGBoostModelConfig, load_xgboost_model_config_from_yaml
import typing as tp

from binding_prediction.const import ModelTypes, FeaturizerTypes
import pyarrow.parquet as pq


@dataclass
class TrainingConfig:
    early_stopping_rounds: int
    pq_groups_numbers: tp.Optional[tp.List[int]] = None
    target_scale_pos_weight: float = -1
    train_size: int = -1


def load_training_config_from_yaml(yaml_path: str) -> TrainingConfig:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)["train"]
    config = TrainingConfig(**config)
    if config.pq_groups_numbers is not None:
        config.pq_groups_numbers = sorted(config.pq_groups_numbers)
    return config


@dataclass
class Config:
    train_file_path: str
    test_file_path: str
    logs_dir: str
    neg_samples: int
    pos_samples: int
    featurizer_config: tp.Union[CircularFingerprintFeaturizerConfig, MACCSFingerprintFeaturizerConfig]
    model_config: tp.Union[XGBoostModelConfig]
    training_config: TrainingConfig
    protein_map_path: str = None


def create_training_config(train_file_path: str, test_file_path: str,
                           logs_dir: str,
                           neg_samples: int, pos_samples: int,
                           config_yaml_path: str) -> Config:
    with open(config_yaml_path, 'r') as file:
        config_yaml = yaml.safe_load(file)
        featurizer_name = config_yaml["featurizer"]["name"]
        model_name = config_yaml["model"]["name"]

    if featurizer_name == FeaturizerTypes.CIRCULAR:
        featurizer_config = load_circular_fingerprint_featurizer_config_from_yaml(config_yaml_path)

    elif featurizer_name == FeaturizerTypes.MACCS:
        featurizer_config = load_maccs_fingerprint_featurizer_config_from_yaml(config_yaml_path)
    else:
        raise ValueError(f"Featurizer {featurizer_name} is not supported")
    if model_name == ModelTypes.XGBOOST:
        model_config = load_xgboost_model_config_from_yaml(config_yaml_path,
                                                           scale_pos_weight=neg_samples / pos_samples)
    else:
        raise ValueError(f"Model {model_name} is not supported")

    training_config = load_training_config_from_yaml(config_yaml_path)
    if training_config.pq_groups_numbers is not None:
        num_pq_groups_in_train = pq.ParquetFile(train_file_path).num_row_groups
        training_config.pq_groups_numbers = list(
            filter(lambda x: x < num_pq_groups_in_train, training_config.pq_groups_numbers))
    return Config(train_file_path=train_file_path, test_file_path=test_file_path,
                  logs_dir=logs_dir,
                  neg_samples=neg_samples, pos_samples=pos_samples,
                  featurizer_config=featurizer_config,
                  model_config=model_config,
                  training_config=training_config)
