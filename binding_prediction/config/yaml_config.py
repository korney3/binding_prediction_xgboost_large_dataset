import typing as tp
from dataclasses import dataclass

from binding_prediction.config.config_creator import get_featurizer_config, get_model_config
from binding_prediction.config.featurizer_config import CircularFingerprintFeaturizerConfig, \
    MACCSFingerprintFeaturizerConfig, EnsemblePredictionsFeaturizerConfig
from binding_prediction.config.model_config import XGBoostModelConfig, XGBoostEnsembleModelConfig
from binding_prediction.config.training_config import TrainingConfig, create_training_config_from_dict


@dataclass
class YamlConfig:
    featurizer_config: tp.Union[
        CircularFingerprintFeaturizerConfig,
        MACCSFingerprintFeaturizerConfig,
        EnsemblePredictionsFeaturizerConfig]
    model_config: tp.Union[XGBoostModelConfig, XGBoostEnsembleModelConfig]
    training_config: TrainingConfig


def create_yaml_config_from_dict(config: dict) -> YamlConfig:
    featurizer_config_name = config['featurizer']["name"]
    featurizer_config = get_featurizer_config(config['featurizer'], featurizer_config_name)
    model_config_name = config['model']["name"]
    model_config = get_model_config(config['model'], model_config_name)
    training_config = create_training_config_from_dict(config['training'])
    return YamlConfig(
        featurizer_config=featurizer_config,
        model_config=model_config,
        training_config=training_config
    )
