from dataclasses import dataclass

import yaml

from binding_prediction.utils import FeaturizerTypes


@dataclass
class CircularFingerprintFeaturizerConfig:
    name: str
    radius: int
    length: int


def load_circular_fingerprint_featurizer_config_from_yaml(yaml_path: str) -> CircularFingerprintFeaturizerConfig:
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)["featurizer"]

    name = config['name']
    if name not in FeaturizerTypes.__dict__.values():
        raise ValueError(f"Featurizer {name} is not supported")
    return CircularFingerprintFeaturizerConfig(**config)
