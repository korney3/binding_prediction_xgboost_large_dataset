import logging
import os
from logging import Logger

import numpy as np
import typing as tp
import yaml

from binding_prediction.config.config import Config
from binding_prediction.const import ModelTypes, WEAK_LEARNER_ARTIFACTS_NAME_PREFIX, \
    FINAL_ENSEMBLE_MODEL_ARTIFACTS_NAME_PREFIX
from binding_prediction.evaluation.utils import evaluate_validation_set, evaluate_test_set
from binding_prediction.training.training_pipeline import TrainingPipeline
from binding_prediction.utils import create_logs_dir, pretty_print_text, \
    get_config, save_weak_learners_data_indices


class Runner:
    def __init__(self, train_parquet_path: str,
                 test_parquet_path: str,
                 config_path: str,
                 debug: bool = False,
                 logs_dir_location: str = 'logs',
                 seed: int = 42,
                 log_level: int = logging.INFO):
        self.seed = seed
        self.logs_dir_location = logs_dir_location
        self.debug = debug
        self.config_path = config_path
        self.test_parquet_path = test_parquet_path
        self.train_parquet_path = train_parquet_path

        self.rng = np.random.default_rng(seed=seed)
        self.logs_dir = None

        self.logger = logging.getLogger(name=__name__)
        self.logger.setLevel(log_level)

    def run(self):
        self.logs_dir = create_logs_dir(self.logs_dir_location)
        config = get_config(self.train_parquet_path,
                            self.test_parquet_path,
                            self.config_path,
                            self.logs_dir)

        if config.yaml_config.model_config.name == ModelTypes.XGBOOST_ENSEMBLE:
            save_weak_learners_data_indices(self.train_parquet_path, config,
                                            self.logs_dir, self.rng)

        train_and_evaluate(config, self.config_path, self.debug, self.rng, logger=self.logger)


def train_and_evaluate(config: Config, config_path: str, debug: bool = False,
                       rng: np.random.Generator = np.random.default_rng(seed=42),
                       train_val_indices: tp.Optional[list[int]] = None,
                       logger: tp.Optional[Logger] = None):
    if config.yaml_config.model_config.name == ModelTypes.XGBOOST_ENSEMBLE:
        for i in range(config.yaml_config.model_config.num_weak_learners):
            pretty_print_text(f"Training weak learner {i}")
            weak_train_val_indices = np.load(os.path.join(config.logs_dir,
                                                          f'{WEAK_LEARNER_ARTIFACTS_NAME_PREFIX}{i}_indices.npy'))
            logs_dir = os.path.join(config.logs_dir, f'{WEAK_LEARNER_ARTIFACTS_NAME_PREFIX}{i}')
            os.makedirs(logs_dir, exist_ok=True)
            with open(config_path, 'r') as file:
                weak_learner_config_dict = yaml.safe_load(file)["model"]["weak_learner_config"]
                weak_learner_config = get_config(config.train_file_path,
                                                 config.test_file_path,
                                                 weak_learner_config_dict, logs_dir,
                                                 weak_train_val_indices)
            train_and_evaluate(weak_learner_config,
                               config_path, debug,
                               rng, weak_train_val_indices)
        train_val_indices = np.load(os.path.join(config.logs_dir,
                                                 f'{FINAL_ENSEMBLE_MODEL_ARTIFACTS_NAME_PREFIX}indices.npy'))
    training_pipeline = TrainingPipeline(config,
                                         debug=debug,
                                         rng=rng,
                                         train_val_indices=train_val_indices,
                                         logger=logger)
    training_pipeline.run()
    evaluate_validation_set(config, config.train_file_path, debug,
                            logger=logger)
    evaluate_test_set(config, config.test_file_path, debug,
                      logger=logger)
