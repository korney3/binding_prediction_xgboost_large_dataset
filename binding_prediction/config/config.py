from dataclasses import dataclass
import typing as tp
import pyarrow.parquet as pq
import yaml

from binding_prediction.config.config_creator import get_model_config, get_featurizer_config, \
    get_training_config
from binding_prediction.config.yaml_config import YamlConfig


@dataclass
class Config:
    train_file_path: str
    test_file_path: str
    logs_dir: str
    neg_samples: int
    pos_samples: int
    yaml_config: YamlConfig
    protein_map_path: str = None


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

    return collect_config(model_config, training_config, featurizer_config,
                          train_file_path, test_file_path,
                          pos_samples, neg_samples, logs_dir)


def collect_config(model_config, training_config, featurizer_config, train_file_path, test_file_path, pos_samples,
                   neg_samples, logs_dir):
    if training_config.pq_groups_numbers is not None:
        num_pq_groups_in_train = pq.ParquetFile(train_file_path).num_row_groups
        training_config.pq_groups_numbers = list(
            filter(lambda x: x < num_pq_groups_in_train, training_config.pq_groups_numbers))
    return Config(train_file_path=train_file_path, test_file_path=test_file_path,
                  logs_dir=logs_dir,
                  neg_samples=neg_samples, pos_samples=pos_samples,
                  yaml_config=YamlConfig(featurizer_config=featurizer_config,
                                         model_config=model_config,
                                         training_config=training_config))
