import os

import numpy as np
import pyarrow.parquet as pq
import xgboost

import binding_prediction.runner
import binding_prediction.training.utils
from binding_prediction.config.config import Config
from binding_prediction.const import TARGET_COLUMN
from binding_prediction.datasets.xgboost_iterator import SmilesIterator
from binding_prediction.models.xgboost_model import XGBoostModel
from binding_prediction.utils import timing_decorator, pretty_print_text, save_config
from binding_prediction.const import ModelTypes


class TrainingPipeline:
    def __init__(self, config: Config,
                 debug: bool = False,
                 rng: np.random.Generator = np.random.default_rng(seed=42),
                 train_val_indices=None):
        self.config = config
        save_config(self.config)

        self.debug = debug
        self.rng = rng

        self.model = None

        self.train_val_pq = pq.ParquetFile(self.config.train_file_path)
        if train_val_indices is not None:
            self.train_val_indices = train_val_indices
        else:
            self.train_val_indices = np.arange(self.train_val_pq.metadata.num_rows)

    def run(self):
        if (self.config.yaml_config.model_config.name == ModelTypes.XGBOOST or
                self.config.yaml_config.model_config.name == ModelTypes.XGBOOST_ENSEMBLE):
            train_Xy, val_Xy = self.prepare_train_val_data()
            save_config(self.config)
            self.model = XGBoostModel(self.config)
            self.train(train_Xy, val_Xy)
        else:
            raise ValueError(f"Model type {self.config.yaml_config.model_config.name} is not supported")

    @timing_decorator
    def train(self, train_dataset, val_dataset):
        pretty_print_text("Training model")
        if (self.config.yaml_config.model_config.name == ModelTypes.XGBOOST or
                self.config.yaml_config.model_config.name == ModelTypes.XGBOOST_ENSEMBLE):
            eval_list = [(train_dataset, 'train'), (val_dataset, 'eval')]
            self.model.train(train_dataset, eval_list)
            self.model.save(os.path.join(self.config.logs_dir, 'model.pkl'))
        else:
            raise ValueError(f"Model type {self.config.yaml_config.model_config.name} is not supported")

    @timing_decorator
    def prepare_train_val_data(self):
        pretty_print_text("Preparing train and validation data")
        train_size = len(self.train_val_indices)
        if self.debug:
            train_size = 50000
            print(f"DEBUG MODE: Using only {train_size} samples for training")
            self.train_val_indices = self.rng.choice(self.train_val_indices,
                                                     train_size,
                                                     replace=False)

        if self.config.yaml_config.training_config.train_size != -1 and self.config.yaml_config.training_config.train_size < train_size:
            self.train_val_indices = self.rng.choice(self.train_val_indices,
                                                     self.config.yaml_config.training_config.train_size, replace=False)
        else:
            self.train_val_indices = self.rng.permutation(self.train_val_indices)
        if (0 <
                self.config.yaml_config.training_config.target_scale_pos_weight <
                self.config.neg_samples / self.config.pos_samples):
            pretty_print_text("Adding positive samples to training set")
            pos_samples_indices = self.sample_positive_indexes_to_add_to_train(self.train_val_pq,
                                                                               self.train_val_indices)
            print(f"Got {len(pos_samples_indices)} indices")
            self.train_val_indices = np.concatenate([self.train_val_indices, pos_samples_indices])
            self.train_val_indices = self.rng.permutation(self.train_val_indices)

            self.config.pos_samples += len(pos_samples_indices)
            self.config.yaml_config.model_config.scale_pos_weight = self.config.neg_samples / self.config.pos_samples

        train_indices, val_indices = self.get_train_val_indices(self.train_val_indices)

        self.save_train_val_indices(train_indices, val_indices)

        if (self.config.yaml_config.model_config.name == ModelTypes.XGBOOST or
                self.config.yaml_config.model_config.name == ModelTypes.XGBOOST_ENSEMBLE):

            train_dataset = SmilesIterator(self.config, self.config.train_file_path,
                                           indicies=train_indices,
                                           shuffle=True)

            val_dataset = SmilesIterator(self.config, self.config.train_file_path,
                                         indicies=val_indices, shuffle=True)

            train_Xy = xgboost.DMatrix(train_dataset)
            val_Xy = xgboost.DMatrix(val_dataset)

            return train_Xy, val_Xy
        else:
            raise ValueError(f"Model type {self.config.yaml_config.model_config.name} is not supported")

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
            self.config.neg_samples / self.config.yaml_config.training_config.target_scale_pos_weight - self.config.pos_samples)
        print(f"Sampling {pos_samples_to_sample} positive samples")
        pos_samples = self.rng.choice(pos_samples_indexes, pos_samples_to_sample, replace=False)
        pos_samples_not_in_train_val = np.setdiff1d(pos_samples, train_val_indices)
        print(f"Got {len(pos_samples_not_in_train_val)} positive samples not in validation set")
        if self.debug:
            pos_samples_not_in_train_val = pos_samples_not_in_train_val[:10000]
        return pos_samples_not_in_train_val

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
