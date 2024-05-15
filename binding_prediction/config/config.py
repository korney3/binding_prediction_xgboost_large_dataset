import typing as tp
from dataclasses import dataclass, field

import yaml

from binding_prediction.config.config_creator import get_model_config, get_featurizer_config, \
    get_training_config
from binding_prediction.config.featurizer_config import CircularFingerprintFeaturizerConfig, \
    EnsemblePredictionsFeaturizerConfig, MACCSFingerprintFeaturizerConfig
from binding_prediction.config.model_config import XGBoostModelConfig, XGBoostEnsembleModelConfig
from binding_prediction.config.training_config import TrainingConfig
from binding_prediction.config.yaml_config import YamlConfig
from binding_prediction.const import PROTEIN_MAP_JSON_PATH


@dataclass
class Config:
    train_file_path: str
    test_file_path: str
    logs_dir: str
    neg_samples: int
    pos_samples: int
    yaml_config: YamlConfig
    protein_map_path: str


def create_config(train_file_path: str, test_file_path: str,
                  logs_dir: str,
                  neg_samples: int, pos_samples: int,
                  config: tp.Union[str, dict]) -> Config:
    if isinstance(config, str):
        with open(config, 'r') as file:
            config_yaml = yaml.safe_load(file)
            featurizer_name = config_yaml["featurizer"]["name"]
            model_name = config_yaml["model"]["name"]
    else:
        featurizer_name = config["featurizer"]["name"]
        model_name = config["model"]["name"]

    featurizer_config = get_featurizer_config(config, featurizer_name)
    model_config = get_model_config(config, model_name, neg_samples, pos_samples)

    training_config = get_training_config(config)
    config = Config(train_file_path=train_file_path, test_file_path=test_file_path, logs_dir=logs_dir,
                    neg_samples=neg_samples, pos_samples=pos_samples,
                    yaml_config=YamlConfig(featurizer_config=featurizer_config,
                                           model_config=model_config,
                                           training_config=training_config),
                    protein_map_path=PROTEIN_MAP_JSON_PATH)
    return config


def construct_config_dataclass(loader, node, dataclass_type: tp.Type):
    init_values = loader.construct_mapping(node)
    return dataclass_type(**init_values)


def yaml_config_constructor_factory(dataclass_type):
    def constructor(loader, node):
        return construct_config_dataclass(loader, node, dataclass_type)

    return constructor


yaml.add_constructor(u'tag:yaml.org,2002:python/object:binding_prediction.config.training_config.TrainingConfig',
                     yaml_config_constructor_factory(TrainingConfig), Loader=yaml.SafeLoader)
yaml.add_constructor(u'tag:yaml.org,2002:python/object:binding_prediction.config.model_config.XGBoostModelConfig',
                     yaml_config_constructor_factory(XGBoostModelConfig), Loader=yaml.SafeLoader)
yaml.add_constructor(
    u'tag:yaml.org,2002:python/object:binding_prediction.config.model_config.XGBoostEnsembleModelConfig',
    yaml_config_constructor_factory(XGBoostEnsembleModelConfig), Loader=yaml.SafeLoader)
yaml.add_constructor(
    u'tag:yaml.org,2002:python/object:binding_prediction.config.featurizer_config.CircularFingerprintFeaturizerConfig',
    yaml_config_constructor_factory(CircularFingerprintFeaturizerConfig), Loader=yaml.SafeLoader)
yaml.add_constructor(
    u'tag:yaml.org,2002:python/object:binding_prediction.config.featurizer_config.MACCSFingerprintFeaturizerConfig',
    yaml_config_constructor_factory(MACCSFingerprintFeaturizerConfig), Loader=yaml.SafeLoader)
yaml.add_constructor(
    u'tag:yaml.org,2002:python/object:binding_prediction.config.featurizer_config.EnsemblePredictionsFeaturizerConfig',
    yaml_config_constructor_factory(EnsemblePredictionsFeaturizerConfig), Loader=yaml.SafeLoader)
yaml.add_constructor(u'tag:yaml.org,2002:python/object:binding_prediction.config.yaml_config.YamlConfig',
                     yaml_config_constructor_factory(YamlConfig), Loader=yaml.SafeLoader)
yaml.add_constructor(u'tag:yaml.org,2002:python/object:binding_prediction.config.config.Config',
                     yaml_config_constructor_factory(Config), Loader=yaml.SafeLoader)
