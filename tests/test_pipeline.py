import os
import tempfile

import pyarrow.parquet as pq
import pytest
import yaml
from mock import patch

from binding_prediction.const import FeaturizerTypes, WEAK_LEARNER_ARTIFACTS_NAME_PREFIX
from binding_prediction.runner import Runner
from binding_prediction.utils import calculate_number_of_neg_and_pos_samples

current_file_path = os.path.abspath(__file__)
TEST_DATA_PATH = os.path.join(os.path.dirname(current_file_path), 'test_data')
CONFIGS_PATH = os.path.join(os.path.dirname(current_file_path), '..', 'binding_prediction', 'config', 'yamls')

PROTEIN_MAP_JSON_PATH_TEST = os.path.join(os.path.dirname(current_file_path), '..', 'data/processed/protein_map.json')

TRAIN_PARQUET_PATH = os.path.join(TEST_DATA_PATH, 'train.parquet')
TEST_PARQUET_PATH = os.path.join(TEST_DATA_PATH, 'test.parquet')

XGBOOST_MODELS_CONFIGS = ['xgboost_config.yaml', 'xgboost_maccs_config.yaml']
XGBOOST_ENSEMBLE_MODELS_CONFIGS = ['xgboost_ensemble_config.yaml']


@patch('binding_prediction.config.config.PROTEIN_MAP_JSON_PATH', PROTEIN_MAP_JSON_PATH_TEST)
class TestPipeline:
    def teardown_class(self):
        print("Cleaning up cache files")
        cache_files = list(filter(lambda x: x.startswith("cache-0") and x.endswith(".page"),
                                  os.listdir(os.path.dirname(current_file_path))))
        cache_files_paths = list(map(lambda x: os.path.join(os.path.dirname(current_file_path), x), cache_files))
        for cache_file in cache_files_paths:
            os.remove(cache_file)

    def test_calculate_pos_neg_samples(self):
        train_val_pq = pq.ParquetFile(TRAIN_PARQUET_PATH)
        neg_samples, pos_samples = calculate_number_of_neg_and_pos_samples(train_val_pq)
        assert neg_samples == 1500
        assert pos_samples == 789

    @pytest.mark.parametrize("config_filename", XGBOOST_MODELS_CONFIGS + XGBOOST_ENSEMBLE_MODELS_CONFIGS)
    def test_xgboost_training_pipeline(self, config_filename):
        with tempfile.TemporaryDirectory() as tmp_dir:
            if config_filename in XGBOOST_MODELS_CONFIGS:
                self._update_xgboost_config_parameters_for_test(os.path.join(CONFIGS_PATH,
                                                                             config_filename), tmp_dir)
            elif config_filename in XGBOOST_ENSEMBLE_MODELS_CONFIGS:
                self._update_xgboost_ensemble_config_parameters_for_test(os.path.join(CONFIGS_PATH,
                                                                                      config_filename), tmp_dir)
            else:
                raise NotImplementedError(f"Config file {config_filename} is not supported.")

            config_path = os.path.join(tmp_dir, 'config.yaml')

            runner = Runner(train_parquet_path=TRAIN_PARQUET_PATH,
                            test_parquet_path=TEST_PARQUET_PATH,
                            config_path=config_path, debug=False,
                            logs_dir_location=tmp_dir, seed=42)
            runner.run()

            files_to_check_existence = ['train_indices.npy', 'val_indices.npy', 'model.pkl', 'model_1.pkl',
                                        'submission.csv', 'train_metrics.csv']

            if config_filename in XGBOOST_ENSEMBLE_MODELS_CONFIGS:
                weak_learner_dir = os.path.join(runner.logs_dir, f'{WEAK_LEARNER_ARTIFACTS_NAME_PREFIX}0')
                assert os.path.exists(weak_learner_dir)
                for file in files_to_check_existence:
                    assert os.path.exists(os.path.join(weak_learner_dir, file)), f"File {file} does not exist in " \
                                                                                 f"{weak_learner_dir}"

            for file in files_to_check_existence:
                assert os.path.exists(os.path.join(runner.logs_dir, file)), f"File {file} does not exist in " \
                                                                            f"{runner.logs_dir}"

    @staticmethod
    def _update_xgboost_config_parameters_for_test(config_path, tmp_dir):
        config = TestPipeline._update_top_model_xgboost_parameters(config_path)
        config["train"]["train_size"] = 2000
        if config["featurizer"]["name"] == FeaturizerTypes.CIRCULAR:
            config["featurizer"]["radius"] = 2
            config["featurizer"]["length"] = 256

        with open(os.path.join(tmp_dir, 'config.yaml'), 'w') as file:
            yaml.dump(config, file)

    @staticmethod
    def _update_xgboost_ensemble_config_parameters_for_test(config_path, tmp_dir):
        config = TestPipeline._update_top_model_xgboost_parameters(config_path)
        config["train"]["train_size"] = 600
        config["model"]["weak_learner_config"]["model"]["max_depth"] = 3
        config["model"]["weak_learner_config"]["model"]["num_boost_round"] = 3
        config["model"]["weak_learner_config"]["train"]["early_stopping_rounds"] = 2
        config["model"]["weak_learner_config"]["train"]["train_size"] = 700

        if config["model"]["weak_learner_config"]["featurizer"][
            "name"] == FeaturizerTypes.CIRCULAR:
            config["model"]["weak_learner_config"]["featurizer"]["radius"] = 2
            config["model"]["weak_learner_config"]["featurizer"]["length"] = 256

        with open(os.path.join(tmp_dir, 'config.yaml'), 'w') as file:
            yaml.dump(config, file)

    @staticmethod
    def _update_top_model_xgboost_parameters(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        config["model"]["max_depth"] = 3
        config["model"]["num_boost_round"] = 3
        config["train"]["early_stopping_rounds"] = 2
        return config
