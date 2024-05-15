import os
import tempfile
import time

import pyarrow.parquet as pq
import pytest
import yaml

from binding_prediction.config.config import create_config
from binding_prediction.utils import calculate_number_of_neg_and_pos_samples, save_config, load_config

current_file_path = os.path.abspath(__file__)
TEST_DATA_PATH = os.path.join(os.path.dirname(current_file_path), 'test_data')
CONFIGS_PATH = os.path.join(os.path.dirname(current_file_path), '..', 'binding_prediction', 'config', 'yamls')
CONFIGS = os.listdir(CONFIGS_PATH)

TRAIN_PARQUET_PATH = os.path.join(TEST_DATA_PATH, 'train.parquet')
TEST_PARQUET_PATH = os.path.join(TEST_DATA_PATH, 'test.parquet')


class TestConfig:
    @pytest.mark.parametrize("config_filename", CONFIGS)
    def test_create_config(self, config_filename):
        train_val_pq = pq.ParquetFile(TRAIN_PARQUET_PATH)

        with tempfile.TemporaryDirectory() as tmp_dir:
            current_date = time.strftime("%Y-%m-%d_%H-%M-%S")
            logs_dir = os.path.join(tmp_dir, 'logs', current_date)
            os.makedirs(logs_dir, exist_ok=True)

            config_path = os.path.join(os.path.dirname(current_file_path),
                                       '../binding_prediction/config/yamls',
                                       config_filename)

            neg_samples, pos_samples = calculate_number_of_neg_and_pos_samples(train_val_pq)
            config = create_config(train_file_path=TRAIN_PARQUET_PATH, test_file_path=TEST_PARQUET_PATH,
                                   logs_dir=logs_dir, neg_samples=neg_samples, pos_samples=pos_samples,
                                   config=config_path)
            with open(os.path.join(CONFIGS_PATH, config_path), 'r') as file:
                config_dict = yaml.safe_load(file)

            yaml_config = config.yaml_config

            self._compare_internal_level_config_parts(config_dict["model"], yaml_config.model_config)
            self._compare_internal_level_config_parts(config_dict["featurizer"], yaml_config.featurizer_config)
            self._compare_internal_level_config_parts(config_dict["train"], yaml_config.training_config)

    @staticmethod
    def _compare_internal_level_config_parts(config_dict_key, config_obj):
        for key, value in config_dict_key.items():
            assert getattr(config_obj, key) == value, (f"Key: {key}, Value: {value} is not equal to "
                                                       f"{getattr(config_obj, key)}")

    @pytest.mark.parametrize("config_filename", CONFIGS)
    def test_saving_and_loading_config(self, config_filename):
        train_val_pq = pq.ParquetFile(TRAIN_PARQUET_PATH)

        with tempfile.TemporaryDirectory() as tmp_dir:
            current_date = time.strftime("%Y-%m-%d_%H-%M-%S")
            logs_dir = os.path.join(tmp_dir, 'logs', current_date)
            os.makedirs(logs_dir, exist_ok=True)

            config_path = os.path.join(os.path.dirname(current_file_path),
                                       '../binding_prediction/config/yamls',
                                       config_filename)

            neg_samples, pos_samples = calculate_number_of_neg_and_pos_samples(train_val_pq)
            config = create_config(train_file_path=TRAIN_PARQUET_PATH, test_file_path=TEST_PARQUET_PATH,
                                   logs_dir=logs_dir, neg_samples=neg_samples, pos_samples=pos_samples,
                                   config=config_path)

            save_config(config)

            loaded_config = load_config(logs_dir)

            assert loaded_config == config
