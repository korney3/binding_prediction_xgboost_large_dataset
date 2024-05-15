import os
import tempfile
import time

import numpy as np
import pyarrow.parquet as pq
import pytest

from binding_prediction.config.config import create_config
from binding_prediction.const import FeaturizerTypes, PROTEIN_MAP_JSON_PATH
from binding_prediction.training.training_pipeline import TrainingPipeline
from binding_prediction.utils import calculate_number_of_neg_and_pos_samples

current_file_path = os.path.abspath(__file__)
TEST_DATA_PATH = os.path.join(os.path.dirname(current_file_path), 'test_data')
CONFIGS_PATH = os.path.join(os.path.dirname(current_file_path), '..', 'binding_prediction', 'config', 'yamls')

PROTEIN_MAP_JSON_PATH_TEST = os.path.join(os.path.dirname(current_file_path), '..', PROTEIN_MAP_JSON_PATH)

TRAIN_PARQUET_PATH = os.path.join(TEST_DATA_PATH, 'train.parquet')
TEST_PARQUET_PATH = os.path.join(TEST_DATA_PATH, 'test.parquet')

XGBOOST_MODELS_CONFIGS = ['xgboost_config.yaml', 'xgboost_maccs_config.yaml']
XGBOOST_ENSEMBLE_MODELS_CONFIGS = ['xgboost_ensemble_config.yaml']


class TestPipeline:
    def test_calculate_pos_neg_samples(self):
        train_val_pq = pq.ParquetFile(TRAIN_PARQUET_PATH)
        neg_samples, pos_samples = calculate_number_of_neg_and_pos_samples(train_val_pq)
        assert neg_samples == 3000
        assert pos_samples == 1589

    @pytest.mark.parametrize("config_filename", XGBOOST_MODELS_CONFIGS)
    def test_xgboost_training_pipeline(self, config_filename):
        train_val_pq = pq.ParquetFile(TRAIN_PARQUET_PATH)

        with tempfile.TemporaryDirectory() as tmp_dir:
            current_date = time.strftime("%Y-%m-%d_%H-%M-%S")
            logs_dir = os.path.join(tmp_dir, 'logs', current_date)
            os.makedirs(logs_dir, exist_ok=True)

            config_path = os.path.join(CONFIGS_PATH,
                                       config_filename)

            neg_samples, pos_samples = calculate_number_of_neg_and_pos_samples(train_val_pq)
            config = create_config(train_file_path=TRAIN_PARQUET_PATH, test_file_path=TEST_PARQUET_PATH,
                                   logs_dir=logs_dir, neg_samples=neg_samples, pos_samples=pos_samples,
                                   config=config_path)

            self._update_xgboost_config_parameters_for_test(config)

            training_pipeline = TrainingPipeline(config,
                                                 debug=False,
                                                 rng=np.random.default_rng(seed=42))

            training_pipeline.run()

            assert os.path.exists(os.path.join(logs_dir, 'train_indices.npy'))
            assert os.path.exists(os.path.join(logs_dir, 'val_indices.npy'))
            assert os.path.exists(os.path.join(logs_dir, 'model.pkl'))
            assert os.path.exists(os.path.join(logs_dir, 'model_1.pkl'))

    def _update_xgboost_config_parameters_for_test(self, config):
        config.protein_map_path = PROTEIN_MAP_JSON_PATH_TEST
        config.yaml_config.model_config.max_depth = 3
        config.yaml_config.model_config.num_boost_round = 3
        config.yaml_config.training_config.early_stopping_rounds = 2
        config.yaml_config.training_config.train_size = 2000
        if config.yaml_config.featurizer_config.name == FeaturizerTypes.CIRCULAR:
            config.yaml_config.featurizer_config.radius = 2
            config.yaml_config.featurizer_config.length = 256

    @pytest.mark.parametrize("config_filename", XGBOOST_ENSEMBLE_MODELS_CONFIGS)
    def test_xgboost_ensemble_training_pipeline(self, config_filename):
        train_val_pq = pq.ParquetFile(TRAIN_PARQUET_PATH)

        with tempfile.TemporaryDirectory() as tmp_dir:
            current_date = time.strftime("%Y-%m-%d_%H-%M-%S")
            logs_dir = os.path.join(tmp_dir, 'logs', current_date)
            os.makedirs(logs_dir, exist_ok=True)

            config_path = os.path.join(CONFIGS_PATH,
                                       config_filename)

            neg_samples, pos_samples = calculate_number_of_neg_and_pos_samples(train_val_pq)
            ensemble_config = create_config(train_file_path=TRAIN_PARQUET_PATH, test_file_path=TEST_PARQUET_PATH,
                                   logs_dir=logs_dir, neg_samples=neg_samples, pos_samples=pos_samples,
                                   config=config_path)

            ensemble_config.protein_map_path = PROTEIN_MAP_JSON_PATH_TEST


            ensemble_config.yaml_config.model_config.max_depth = 3
            ensemble_config.yaml_config.model_config.num_boost_round = 3
            ensemble_config.yaml_config.training_config.early_stopping_rounds = 2
            ensemble_config.yaml_config.training_config.train_size = 1000
            ensemble_config.yaml_config.model_config.weak_learner_config["model"]["max_depth"] = 3
            ensemble_config.yaml_config.model_config.weak_learner_config["model"]["num_boost_round"] = 3
            ensemble_config.yaml_config.model_config.weak_learner_config["model"]["early_stopping_rounds"] = 2
            ensemble_config.yaml_config.model_config.weak_learner_config["model"]["train_size"] = 1000


            if ensemble_config.yaml_config.model_config.weak_learner_config["featurizer"]["name"] == FeaturizerTypes.CIRCULAR:
                ensemble_config.yaml_config.model_config.weak_learner_config["featurizer"]["radius"] = 2
                ensemble_config.yaml_config.model_config.weak_learner_config["featurizer"]["length"] = 256

            training_pipeline = TrainingPipeline(ensemble_config,
                                                 debug=False,
                                                 rng=np.random.default_rng(seed=42))

            training_pipeline.run()

            assert os.path.exists(os.path.join(logs_dir, 'train_indices.npy'))
            assert os.path.exists(os.path.join(logs_dir, 'val_indices.npy'))
            assert os.path.exists(os.path.join(logs_dir, 'model.pkl'))
            assert os.path.exists(os.path.join(logs_dir, 'model_1.pkl'))