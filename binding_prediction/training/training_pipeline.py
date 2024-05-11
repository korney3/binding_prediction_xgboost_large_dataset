import os

import numpy as np
import pyarrow.parquet as pq
import xgboost
import yaml

from binding_prediction.config.config_creation import Config
from binding_prediction.const import TARGET_COLUMN, PROTEIN_MAP_JSON_PATH
from binding_prediction.datasets.xgboost_iterator import SmilesIterator
from binding_prediction.evaluation.kaggle_submission_creation import get_submission_test_predictions_for_xgboost_model
from binding_prediction.models.xgboost_model import XGBoostModel
from binding_prediction.utils import timing_decorator, pretty_print_text
from binding_prediction.const import ModelTypes


class TrainingPipeline:
    def __init__(self, config: Config,
                 debug: bool = False,
                 rng: np.random.Generator = np.random.default_rng(seed=42)):
        self.config = config
        self.save_config()

        self.debug = debug
        self.rng = rng

        self.model = None

        self.protein_map_path = PROTEIN_MAP_JSON_PATH

    def run(self):
        if self.config.model_config.name == ModelTypes.XGBOOST:
            train_Xy, val_Xy = self.prepare_train_val_data()
            self.save_config()
            self.model = XGBoostModel(self.config)
            self.train(train_Xy, val_Xy)
            self.evaluate()
        else:
            raise ValueError(f"Model type {self.config.model_config.name} is not supported")

    def save_config(self):
        with open(os.path.join(self.config.logs_dir, 'config.yaml'), 'w') as file:
            yaml.dump(self.config.__dict__, file)

    @timing_decorator
    def train(self, train_dataset, val_dataset):
        pretty_print_text("Training model")
        if self.config.model_config.name == ModelTypes.XGBOOST:
            eval_list = [(train_dataset, 'train'), (val_dataset, 'eval')]
            self.model.train(train_dataset, eval_list)
            self.model.save(os.path.join(self.config.logs_dir, 'model.pkl'))
        else:
            raise ValueError(f"Model type {self.config.model_config.name} is not supported")

    @timing_decorator
    def evaluate(self):
        pretty_print_text("Testing model")
        if self.config.model_config.name == ModelTypes.XGBOOST:
            test_dataset, test_Xy = self.prepare_test_data()
            get_submission_test_predictions_for_xgboost_model(test_dataset, test_Xy,
                                                              self.model, self.config.logs_dir)
        else:
            raise ValueError(f"Model type {self.config.model_config.name} is not supported")

    @timing_decorator
    def prepare_train_val_data(self):
        pretty_print_text("Preparing train and validation data")
        train_val_pq = pq.ParquetFile(self.config.train_file_path)
        if self.debug:
            train_size = 50000
            print(f"DEBUG MODE: Using only {train_size} samples for training")
            train_val_indices = self.rng.choice(train_val_pq.metadata.num_rows,
                                                train_size,
                                                replace=False)
        else:
            if self.config.training_config.pq_groups_numbers is not None:
                train_size = 0
                train_val_indices = []
                shard_size = train_val_pq.metadata.row_group(0).num_rows
                for group_number in self.config.training_config.pq_groups_numbers:
                    train_size += train_val_pq.metadata.row_group(group_number).num_rows
                    start_index = shard_size * group_number
                    train_val_indices.extend(
                        range(start_index, start_index + train_val_pq.metadata.row_group(group_number).num_rows))
            else:
                train_size = train_val_pq.metadata.num_rows
                train_val_indices = self.rng.choice(train_val_pq.metadata.num_rows,
                                                    train_size,
                                                    replace=False)
        if self.config.training_config.train_size != -1 and self.config.training_config.train_size < train_size:
            train_val_indices = self.rng.choice(train_val_indices,
                                                self.config.training_config.train_size, replace=False)
        if (0 <
                self.config.training_config.target_scale_pos_weight <
                self.config.neg_samples / self.config.pos_samples):
            pretty_print_text("Adding positive samples to training set")
            pos_samples_indices = self.sample_positive_indexes_to_add_to_train(train_val_pq, train_val_indices)
            print(f"Got {len(pos_samples_indices)} indices")
            train_val_indices = np.concatenate([train_val_indices, pos_samples_indices])
            train_val_indices = self.rng.permutation(train_val_indices)

            self.config.pos_samples += len(pos_samples_indices)
            self.config.model_config.scale_pos_weight = self.config.neg_samples / self.config.pos_samples

        train_indices, val_indices = self.get_train_val_indices(train_val_indices)

        self.save_train_val_indices(train_indices, val_indices)

        if self.config.model_config.name == ModelTypes.XGBOOST:
            train_dataset = SmilesIterator(self.config, self.config.train_file_path,
                                           indicies=train_indices,
                                           shuffle=True)

            self.config.protein_map_path = self.protein_map_path

            val_dataset = SmilesIterator(self.config, self.config.train_file_path,
                                         indicies=val_indices,
                                         shuffle=True)

            train_Xy = xgboost.DMatrix(train_dataset)
            val_Xy = xgboost.DMatrix(val_dataset)

            return train_Xy, val_Xy
        else:
            raise ValueError(f"Model type {self.config.model_config.name} is not supported")

    def sample_positive_indexes_to_add_to_train(self, train_val_pq, train_val_indices):
        pos_samples_indexes = []
        group_size = train_val_pq.metadata.row_group(0).num_rows
        last_index = 0
        for group_number in range(train_val_pq.num_row_groups):
            row_group = train_val_pq.read_row_group(group_number).to_pandas()
            pos_samples_indexes.extend([x + last_index for x in row_group[row_group[TARGET_COLUMN] == 1].index])
            last_index += group_size
        pos_samples_indexes = np.array(pos_samples_indexes)
        print(f"Got {len(pos_samples_indexes)} positive samples")
        pos_samples_to_sample = int(
            self.config.neg_samples / self.config.training_config.target_scale_pos_weight - self.config.pos_samples)
        print(f"Sampling {pos_samples_to_sample} positive samples")
        pos_samples = self.rng.choice(pos_samples_indexes, pos_samples_to_sample, replace=False)
        pos_samples_not_in_train_val = np.setdiff1d(pos_samples, train_val_indices)
        print(f"Got {len(pos_samples_not_in_train_val)} positive samples not in validation set")
        if self.debug:
            pos_samples_not_in_train_val = pos_samples_not_in_train_val[:10000]
        return pos_samples_not_in_train_val

    def prepare_test_data(self):
        if self.config.model_config.name == ModelTypes.XGBOOST:

            test_dataset = SmilesIterator(self.config, self.config.test_file_path,
                                          shuffle=False)
            test_Xy = xgboost.DMatrix(test_dataset)
            return test_dataset, test_Xy
        else:
            raise ValueError(f"Model type {self.config.model_config.name} is not supported")

    def get_train_val_indices(self, train_val_indices):
        train_size = len(train_val_indices)
        train_indices = self.rng.choice(train_val_indices, int(0.8 * train_size), replace=False)
        val_indices = np.setdiff1d(train_val_indices, train_indices)
        return train_indices, val_indices

    def save_train_val_indices(self, train_indices, val_indices):
        with open(os.path.join(self.config.logs_dir, 'train_indices.npy'), 'wb') as f:
            np.save(f, train_indices)
        with open(os.path.join(self.config.logs_dir, 'val_indices.npy'), 'wb') as f:
            np.save(f, val_indices)
