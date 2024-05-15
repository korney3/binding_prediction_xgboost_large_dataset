from dataclasses import dataclass

import yaml

from binding_prediction.const import FeaturizerTypes


@dataclass
class CircularFingerprintFeaturizerConfig:
    name: str
    radius: int
    length: int


def load_circular_fingerprint_featurizer_config_from_yaml_path(yaml_path: str) -> CircularFingerprintFeaturizerConfig:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return create_circular_fingerprint_featurizer_config_from_dict(config)


def create_circular_fingerprint_featurizer_config_from_dict(config: dict) -> CircularFingerprintFeaturizerConfig:
    featurizer_config_dict = config["featurizer"]
    name = featurizer_config_dict['name']
    if name not in FeaturizerTypes.__dict__.values():
        raise ValueError(f"Featurizer {name} is not supported")
    return CircularFingerprintFeaturizerConfig(**featurizer_config_dict)


@dataclass
class MACCSFingerprintFeaturizerConfig:
    name: str


def load_maccs_fingerprint_featurizer_config_from_yaml_path(yaml_path: str) -> MACCSFingerprintFeaturizerConfig:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return create_maccs_fingerprint_featurizer_config_from_dict(config)


def create_maccs_fingerprint_featurizer_config_from_dict(config: dict) -> MACCSFingerprintFeaturizerConfig:
    featurizer_config_dict = config["featurizer"]
    name = featurizer_config_dict['name']
    if name != FeaturizerTypes.MACCS:
        raise ValueError(f"Featurizer {name} is not supported")
    return MACCSFingerprintFeaturizerConfig(**featurizer_config_dict)


@dataclass
class EnsemblePredictionsFeaturizerConfig:
    name: str


def load_ensemble_predictions_featurizer_config_from_yaml_path(yaml_path: str) -> EnsemblePredictionsFeaturizerConfig:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return create_ensemble_predictions_featurizer_config_from_dict(config)


def create_ensemble_predictions_featurizer_config_from_dict(config: dict) -> EnsemblePredictionsFeaturizerConfig:
    featurizer_config_dict = config["featurizer"]
    name = featurizer_config_dict['name']
    if name != FeaturizerTypes.ENSEMBLE_PREDICTIONS:
        raise ValueError(f"Featurizer {name} is not supported")
    return EnsemblePredictionsFeaturizerConfig(**featurizer_config_dict)
